# Experiment 6.2 Specification & Implementation Report
## Pinned Host Memory & Asynchronous CUDA Streams — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming
**Workspace:** `D:\FraudGraph Intelligence`
**Baseline Commit:** `8d6f46e` (Phase 6.1 Complete)
**Target Hardware:** NVIDIA Tesla T4 (`sm_75`, 40 SMs, 14.9 GB VRAM, 320.0 GB/s Peak Bandwidth)
**Driver & Compiler:** NVIDIA Driver 580.82.07, CUDA Toolkit 13.0, V13.0.88 (`-O3 -std=c++17`)
**Status:** PASSED (Complete & Verified on GPU)

---

## 1. Executive Summary & Objective

In Experiment 6.1 (Warp-Coalesced Aggregation), the pure CUDA kernel achieved a $\sim 3.7\times - 4.1\times$ acceleration by coalescing DRAM transactions into contiguous 128-byte cache lines. However, end-to-end wall-clock speedup was severely constrained to **$0.99\times - 1.09\times$** because synchronous PCIe host-to-device (H2D) and device-to-host (D2H) memory copies accounted for **$92.9\% - 98.1\%$** of total execution time. Under Amdahl's Law, optimizing the kernel in isolation could never yield major whole-pipeline speedups without addressing host-device data movement.

**Experiment 6.2** directly targets this transfer bottleneck through two architectural techniques:
1. **Page-Locked (Pinned) Host Memory (`cudaMallocHost` / `cudaFreeHost`):** Eliminates operating system paging and CPU bounce-buffer staging, enabling the GPU's Direct Memory Access (DMA) engine to read and write host memory across PCIe at maximum physical bus throughput.
2. **Asynchronous Stream Execution (`cudaStreamNonBlocking`):** Dispatches asynchronous memory copies (`cudaMemcpyAsync`) and kernel launches on a dedicated non-default stream, ensuring decoupled, deterministic hardware execution and establishing the architectural foundation for multi-batch pipelining.

### Critical Implementation Boundaries
1. **Baseline & Phase 6.1 Preservation:**
   - Baseline sources [`backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu) and [`main_baseline.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_baseline.cu) remain **strictly untouched**.
   - Phase 6.1 sources [`backend/gpu_engine/cuda/src/fgi_coalesced_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_coalesced_kernels.cu) and [`main_coalesced.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_coalesced.cu) remain **strictly untouched**.
2. **Dedicated Candidate Implementation:**
   - Authored in [`backend/gpu_engine/cuda/src/main_streams.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_streams.cu) reusing the verified `fgi::launch_coalesced_aggregation` kernel.
3. **Multi-Target Build:** [`backend/gpu_engine/cuda/Makefile`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/Makefile) builds all three standalone binaries concurrently:
   - `bin/fgi_baseline_runner`
   - `bin/fgi_coalesced_runner`
   - `bin/fgi_streams_runner`

---

## 2. Architecture & Memory Management

### 2.1 The Pinned Host Memory DMA Mechanism

In standard CUDA programming (used in Phase 5 and Phase 6.1), host arrays allocated with standard C++ `malloc`, `std::vector`, or pageable memory cannot be accessed directly by the GPU DMA engine because the host operating system could page out or relocate physical memory addresses at any time. Consequently, `cudaMemcpy` must first copy the data into a temporary, driver-managed page-locked staging buffer on the CPU, and then initiate DMA transfer to GPU VRAM:
$$\text{Host Pageable RAM} \xrightarrow{\text{CPU Copy}} \text{OS Staging Buffer} \xrightarrow{\text{PCIe DMA}} \text{GPU VRAM}$$

In Experiment 6.2, all input graph structures (`row_ptr`, `col_idx`, `features`) and the output tensor (`h_pinned_output`) are allocated directly in page-locked host memory using `cudaMallocHost`:
```cpp
struct PinnedHostCSRGraph {
    int num_nodes = 0;
    int num_edges = 0;
    int feature_dim = 0;
    int* row_ptr = nullptr;
    int* col_idx = nullptr;
    float* features = nullptr;

    ~PinnedHostCSRGraph() {
        if (row_ptr) cudaFreeHost(row_ptr);
        if (col_idx) cudaFreeHost(col_idx);
        if (features) cudaFreeHost(features);
    }
};
```
This permits the GPU DMA engine to transfer directly between host physical memory and device memory across the PCIe bus without CPU intervention:
$$\text{Host Pinned RAM} \xrightarrow[\text{Direct PCIe DMA}]{\text{Max Physical Bandwidth}} \text{GPU VRAM}$$

### 2.2 Asynchronous Stream & Event Protocol

A dedicated non-default, non-blocking CUDA stream is created to sequence all asynchronous operations:
```cpp
cudaStream_t stream = nullptr;
CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
```
Decoupled hardware timing is captured using CUDA events recorded strictly on this dedicated stream:
1. **Timed Asynchronous H2D Transfer:**
   ```cpp
   CUDA_CHECK(cudaEventRecord(start_event, stream));
   CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, graph.row_ptr, bytes_row, cudaMemcpyHostToDevice, stream));
   CUDA_CHECK(cudaMemcpyAsync(d_col_idx, graph.col_idx, bytes_col, cudaMemcpyHostToDevice, stream));
   CUDA_CHECK(cudaMemcpyAsync(d_features, graph.features, bytes_feat, cudaMemcpyHostToDevice, stream));
   CUDA_CHECK(cudaEventRecord(stop_event, stream));
   CUDA_CHECK(cudaEventSynchronize(stop_event));
   CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start_event, stop_event));
   ```
2. **Timed Kernel Execution:**
   The warp-coalesced kernel is launched asynchronously on `stream`:
   ```cpp
   CUDA_CHECK(cudaEventRecord(start_event, stream));
   fgi::launch_coalesced_aggregation(
       d_row_ptr, d_col_idx, d_features, d_output,
       graph.num_nodes, graph.feature_dim, op, block_size, stream
   );
   CUDA_CHECK(cudaEventRecord(stop_event, stream));
   CUDA_CHECK(cudaEventSynchronize(stop_event));
   CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start_event, stop_event));
   ```
3. **Timed Asynchronous D2H Transfer:**
   ```cpp
   CUDA_CHECK(cudaEventRecord(start_event, stream));
   CUDA_CHECK(cudaMemcpyAsync(h_pinned_output, d_output, bytes_out, cudaMemcpyDeviceToHost, stream));
   CUDA_CHECK(cudaEventRecord(stop_event, stream));
   CUDA_CHECK(cudaEventSynchronize(stop_event));
   CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start_event, stop_event));
   ```
4. **Host Synchronization:**
   `cudaStreamSynchronize(stream)` is called to guarantee all device work has concluded before CPU reference verification or output serialization.

---

## 3. Measured Google Colab Tesla T4 Results & Audit

The empirical benchmark run was executed on an **NVIDIA Tesla T4** (`sm_75`, 40 SMs, 14.9 GB VRAM, Driver 580.82.07, CUDA Toolkit 13.0, V13.0.88). All 9 toy-graph correctness validations and all 9 Tier A benchmark records passed numerical verification with maximum absolute error equal to $0.00$ ($\text{atol} \le 1.0 \times 10^{-5}$).

### 3.1 Comparative Performance Matrix (Tier A: $N=10,000$, $M=94,945$, $D=32$, 20 Repetitions, 5 Warmup)
*Timestamp: `20261024_114748` on NVIDIA Tesla T4 / CUDA 13.0*

| Operator | Step / Phase | H2D Transfer (ms) | Kernel Latency (ms) | D2H Transfer (ms) | **End-to-End Latency (ms)** | **Pure Kernel Speedup** | **End-to-End Speedup** | PCIe H2D BW | PCIe D2H BW | Validation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SUM** | **Baseline (Phase 5)** | $0.5513\text{ ms}$ | $0.0679\text{ ms}$ | $0.3170\text{ ms}$ | **$0.9362\text{ ms}$** | $1.00\times$ (Ref) | $1.00\times$ (Ref) | N/A (Pageable) | N/A (Pageable) | $\text{err}=0.00$ (`PASSED`) |
| | **Coalesced (Phase 6.1)** | $0.5760\text{ ms}$ | $0.0169\text{ ms}$ | $0.3140\text{ ms}$ | **$0.9069\text{ ms}$** | $4.02\times$ vs Base | $1.03\times$ vs Base | N/A (Pageable) | N/A (Pageable) | $\text{err}=0.00$ (`PASSED`) |
| | **Streams (Phase 6.2)** | **$0.1563\text{ ms}$** | **$0.0167\text{ ms}$** | **$0.1051\text{ ms}$** | **$0.2782\text{ ms}$** | **$4.06\times$ vs Base**<br>($1.01\times$ vs 6.1) | **$3.37\times$ vs Base**<br>**$3.26\times$ vs 6.1** | $10.87\text{ GB/s}$ | $12.18\text{ GB/s}$ | $\text{err}=0.00$ (`PASSED`) |
| **MEAN** | **Baseline (Phase 5)** | $0.5377\text{ ms}$ | $0.0678\text{ ms}$ | $0.3363\text{ ms}$ | **$0.9418\text{ ms}$** | $1.00\times$ (Ref) | $1.00\times$ (Ref) | N/A (Pageable) | N/A (Pageable) | $\text{err}=0.00$ (`PASSED`) |
| | **Coalesced (Phase 6.1)** | $0.5596\text{ ms}$ | $0.0176\text{ ms}$ | $0.3284\text{ ms}$ | **$0.9056\text{ ms}$** | $3.85\times$ vs Base | $1.04\times$ vs Base | N/A (Pageable) | N/A (Pageable) | $\text{err}=0.00$ (`PASSED`) |
| | **Streams (Phase 6.2)** | **$0.1577\text{ ms}$** | **$0.0175\text{ ms}$** | **$0.1057\text{ ms}$** | **$0.2809\text{ ms}$** | **$3.88\times$ vs Base**<br>($1.01\times$ vs 6.1) | **$3.35\times$ vs Base**<br>**$3.22\times$ vs 6.1** | $10.78\text{ GB/s}$ | $12.11\text{ GB/s}$ | $\text{err}=0.00$ (`PASSED`) |
| **MAX** | **Baseline (Phase 5)** | $0.5503\text{ ms}$ | $0.0664\text{ ms}$ | $0.3503\text{ ms}$ | **$0.9671\text{ ms}$** | $1.00\times$ (Ref) | $1.00\times$ (Ref) | N/A (Pageable) | N/A (Pageable) | $\text{err}=0.00$ (`PASSED`) |
| | **Coalesced (Phase 6.1)** | $0.5532\text{ ms}$ | $0.0166\text{ ms}$ | $0.3485\text{ ms}$ | **$0.9183\text{ ms}$** | $4.00\times$ vs Base | $1.05\times$ vs Base | N/A (Pageable) | N/A (Pageable) | $\text{err}=0.00$ (`PASSED`) |
| | **Streams (Phase 6.2)** | **$0.1595\text{ ms}$** | **$0.0165\text{ ms}$** | **$0.1049\text{ ms}$** | **$0.2809\text{ ms}$** | **$4.02\times$ vs Base**<br>($1.00\times$ vs 6.1) | **$3.44\times$ vs Base**<br>**$3.27\times$ vs 6.1** | $10.66\text{ GB/s}$ | $12.20\text{ GB/s}$ | $\text{err}=0.00$ (`PASSED`) |

---

## 4. In-Depth Technical Analysis

### 4.1 Transfer Acceleration via Direct DMA
- **Host-to-Device (H2D) Transfer:**
  - In baseline and coalesced engines (pageable memory), transferring the graph input data ($1.70\text{ MB}$ total: `row_ptr` $40.0\text{ KB}$, `col_idx` $379.8\text{ KB}$, `features` $1.28\text{ MB}$) required $0.538 - 0.576\text{ ms}$ (effective bandwidth $\sim 2.95 - 3.16\text{ GB/s}$).
  - With pinned memory DMA in Experiment 6.2, H2D transfer time dropped to **$0.156 - 0.160\text{ ms}$** (**$3.41\times - 3.53\times$ speedup** over baseline).
  - Effective PCIe H2D payload bandwidth reached **$10.66 - 10.87\text{ GB/s}$**, representing a $3.5\times$ increase in host-to-device data ingestion rate.
- **Device-to-Host (D2H) Transfer:**
  - Transferring the output feature matrix ($1.28\text{ MB}$) dropped from $0.314 - 0.350\text{ ms}$ to **$0.105\text{ ms}$** (**$3.02\times - 3.34\times$ speedup** over baseline).
  - Effective PCIe D2H payload bandwidth reached **$12.11 - 12.20\text{ GB/s}$**, compared to $3.65 - 4.08\text{ GB/s}$ under pageable memory.
- **Methodological Context on PCIe Bandwidth:**
  - Effective PCIe bandwidth is computed directly from application payload bytes divided by transfer duration: $\text{BW} = V_{\text{bytes}} / (t_{\text{transfer}} \times 10^9)$.
  - The theoretical raw signaling limit of a PCIe Gen3 x16 interconnect is $15.75\text{ GB/s}$ ($16 \text{ lanes} \times 8\text{ GT/s} \times \frac{128}{130}$).
  - For small tensor buffers ($1.28 - 1.70\text{ MB}$), transfer latency encompasses driver invocation, transaction-layer packet (TLP) header framing, flow control, and PCIe link ramp-up. We report the measured $10.66 - 12.20\text{ GB/s}$ as effective payload transfer throughput, and explicitly do not claim full physical bus saturation, which would require dedicated large-buffer bandwidth sweep microbenchmarks.

### 4.2 Kernel Invariance & Reproducibility
- Because `main_streams.cu` dispatches the exact same `fgi::launch_coalesced_aggregation` kernel used in Phase 6.1, pure kernel latency remained mathematically invariant:
  - **Sum:** $0.016869\text{ ms}$ (Coalesced 6.1) vs $0.016731\text{ ms}$ (Streams 6.2) — delta $0.000138\text{ ms}$ ($0.82\%$, $1.008\times$ ratio).
  - **Mean:** $0.017590\text{ ms}$ (Coalesced 6.1) vs $0.017486\text{ ms}$ (Streams 6.2) — delta $0.000104\text{ ms}$ ($0.59\%$, $1.006\times$ ratio).
  - **Max:** $0.016558\text{ ms}$ (Coalesced 6.1) vs $0.016512\text{ ms}$ (Streams 6.2) — delta $0.000046\text{ ms}$ ($0.28\%$, $1.003\times$ ratio).
  - Across all three operators, the empirical kernel latency delta between Phase 6.1 and Phase 6.2 is $\le 0.82\%$ (absolute delta $\le 0.00014\text{ ms}$ / $0.14\ \mu\text{s}$), confirming execution invariance within hardware runtime noise.
- Pure kernel throughput reached **$5.43\times 10^9 - 5.75\times 10^9\text{ edges/sec}$**, retaining the full $\sim 4.0\times$ acceleration over the baseline kernel ($1.40\times 10^9 - 1.43\times 10^9\text{ edges/sec}$).

### 4.3 Amdahl's Law Resolution

In Phase 6.1, the kernel accounted for only $\sim 7\%$ of end-to-end execution time ($0.068\text{ ms}$ out of $0.94\text{ ms}$), with transfers dominating $93\%$. Amdahl's Law states:
$$S_{\text{overall}} = \frac{1}{(1 - p) + \frac{p}{s}}$$
For Phase 6.1 where $p \approx 0.07$ and $s \approx 4.0$:
$$S_{\text{overall}} \approx \frac{1}{0.93 + \frac{0.07}{4.0}} = \frac{1}{0.9475} \approx 1.055\times$$
This explains why Phase 6.1 achieved only $1.03\times - 1.05\times$ end-to-end speedup despite a $4\times$ kernel improvement.

In Experiment 6.2, pinned host memory DMA directly compressed the non-kernel portion ($1 - p$) from $\sim 0.89\text{ ms}$ down to $\sim 0.26\text{ ms}$ (a $3.42\times$ reduction in data transfer time). Combined with the $4\times$ kernel acceleration:
$$t_{\text{e2e}} = 0.158\text{ ms (H2D)} + 0.017\text{ ms (Kernel)} + 0.105\text{ ms (D2H)} = 0.280\text{ ms}$$
$$S_{\text{overall}} = \frac{0.95\text{ ms}}{0.28\text{ ms}} \approx \mathbf{3.39\times}$$
This matches the measured end-to-end speedups of **$3.35\times - 3.44\times$ vs Baseline** and **$3.22\times - 3.27\times$ vs Coalesced 6.1**.

### 4.4 Scientific Boundaries & Non-Overlap Disclosure
1. **Sequential Stream Pipeline (No Concurrent Overlap Claimed):**
   - The benchmark pipeline processes a single graph partition sequentially: `H2D -> Kernel -> D2H` within a single non-blocking stream.
   - Pinned memory alone is responsible for the $3.4\times$ PCIe transfer acceleration through direct DMA.
   - We explicitly **do not claim** concurrent transfer/kernel overlap for this single-graph aggregation benchmark, as such overlap requires multiple independent graph sub-batches.
2. **Analytical Bandwidth Disclosures:**
   - Both `estimated_compulsory_bandwidth_gbps` ($170.41 - 180.46\text{ GB/s}$) and `estimated_algorithmic_bandwidth_gbps` ($792.20 - 838.95\text{ GB/s}$) remain analytical calculations, not hardware counter measurements.
   - As in Phase 6.1, the algorithmic bandwidth rate exceeds the T4's physical DRAM peak ($320\text{ GB/s}$) because the $1.28\text{ MB}$ feature table fits entirely in the $4.0\text{ MB}$ L2 cache.
3. **Application Domain:**
   - This experiment strictly evaluates the memory transfer and graph aggregation pipeline. Downstream fraud classification performance (GNN inference, PR-AUC, F1-score) is evaluated in Phase 7.

### 4.5 Timing Boundaries and Synchronization Comparability

To ensure equitable scientific comparison, all three runners enforce identical timing boundaries and measurement protocols:
- **Baseline (`main_baseline.cu`) & Coalesced (`main_coalesced.cu`):**
  - Input arrays reside in standard C++ pageable memory (`std::vector`).
  - H2D transfer: Bracketed by CUDA events (`start_event`, `stop_event`) on default stream (0) using synchronous `cudaMemcpy`.
  - Kernel execution: Isolated from transfers; bracketed by CUDA events on stream 0 across 20 repetitions following 5 warmups. The mean kernel latency is recorded.
  - D2H transfer: Bracketed by CUDA events on default stream (0) using synchronous `cudaMemcpy`.
  - End-to-End latency: Formulated as $t_{\text{e2e}} = t_{\text{H2D}} + t_{\text{kernel\_mean}} + t_{\text{D2H}}$.
- **Streams Runner (`main_streams.cu`):**
  - Input and output buffers are allocated in page-locked host memory (`cudaMallocHost`).
  - Operations execute on a dedicated non-default stream created with `cudaStreamNonBlocking`.
  - H2D transfer: Bracketed by CUDA events recorded on `stream` around `cudaMemcpyAsync(..., stream)` and synchronized with `cudaEventSynchronize(stop_event)`.
  - Kernel execution: Bracketed by CUDA events recorded on `stream` around `fgi::launch_coalesced_aggregation(..., stream)` across 20 repetitions following 5 warmups. The mean kernel latency is recorded.
  - D2H transfer: Bracketed by CUDA events recorded on `stream` around `cudaMemcpyAsync(..., stream)` and synchronized with `cudaEventSynchronize(stop_event)`.
  - Synchronization barrier: An explicit `cudaStreamSynchronize(stream)` is executed prior to host validation or file export.
  - End-to-End latency: Formulated as $t_{\text{e2e}} = t_{\text{H2D}} + t_{\text{kernel\_mean}} + t_{\text{D2H}}$.
- **Comparability Disclosure:**
  - In all three runners, disk file I/O, CSR parsing, JSON serialization, and CPU numerical verification are strictly excluded from the benchmark timing.
  - The definition of $t_{\text{e2e}}$ as the sum of decoupled transfer and mean kernel times is identical across all three runners.
  - The measured speedup is therefore isolated entirely to: (a) 128-byte warp coalescing for the kernel stage, and (b) direct DMA via page-locked host memory for the transfer stages.

---

## 5. Artifacts and Raw Evidence Preservation

All raw measurement files are preserved in `experiments/results/`:
- **Tier A Benchmarks ($N=10,000, M=94,945, D=32$):**
  - [`exp_gpu_baseline_tier_a_{mean,sum,max}_20261024_114748.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/)
  - [`exp_gpu_coalesced_tier_a_{mean,sum,max}_20261024_114748.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/)
  - [`exp_gpu_streams_tier_a_{mean,sum,max}_20261024_114748.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/)
- **Toy Graph Correctness Validations ($N=5, M=8, D=4$):**
  - `val_toy_baseline_{mean,sum,max}_20261024_114748.json`
  - `val_toy_coalesced_{mean,sum,max}_20261024_114748.json`
  - `val_toy_streams_{mean,sum,max}_20261024_114748.json`
- **Compressed Archives:**
  - Active Archive: [`experiments/fgi_phase6_gpu_results.zip`](file:///D:/FraudGraph%20Intelligence/experiments/fgi_phase6_gpu_results.zip)
  - Dedicated Phase 6.2 Archive: [`experiments/fgi_phase6_streams_gpu_results_20261024_114748.zip`](file:///D:/FraudGraph%20Intelligence/experiments/fgi_phase6_streams_gpu_results_20261024_114748.zip)
  - Prior Phase 6.1 Revalidation Archive: [`experiments/fgi_phase6_revalidation_gpu_results_20261019_175441.zip`](file:///D:/FraudGraph%20Intelligence/experiments/fgi_phase6_revalidation_gpu_results_20261019_175441.zip)

---

## 6. Phase Gate Acceptance

- **Candidate Implementation:** **COMPLETE** ([`backend/gpu_engine/cuda/src/main_streams.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_streams.cu))
- **Correctness Verification:** **PASSED** (100% agreement, max absolute error = 0.00 across all 9 configurations)
- **Measured End-to-End Speedup:** **$3.35\times - 3.44\times$ vs Baseline**, **$3.22\times - 3.27\times$ vs Coalesced 6.1**
- **Phase 6.2 Status:** **`PASSED (Experiment 6.2 Complete & Verified)`**
