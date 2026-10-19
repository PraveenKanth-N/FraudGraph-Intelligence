# Baseline CUDA C++ Implementation & Google Colab Workflow — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Course Code:** CSI4007  
**Project:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Module:** `backend/gpu_engine/cuda/` & `backend/gpu_engine/colab_runner/`  
**Phase:** Phase 5 Deliverable  

---

## 1. Executive Summary & Purpose of Baseline

The **Baseline CUDA Implementation** establishes the initial parallel GPU acceleration benchmark for neighbor aggregation in FraudGraph Intelligence. In strict accordance with the project roadmap:
- Phase 5 implements a straightforward, auditable **one-thread-per-node** mapping.
- It provides complete host orchestration, 64-bit safe allocations, hardware CUDA event timing, and numerical validation against the Phase 4 CPU reference.
- It intentionally omits advanced memory optimizations (such as warp-level reductions, shared-memory tiling, and degree-aware binning), which are reserved exclusively for Phase 6.
- It implements a turnkey **Google Colab Workflow** to compile and benchmark on actual cloud GPU hardware (NVIDIA T4 / V100 / A100) without requiring local Windows compiler toolchains.

---

## 2. Kernel Design & Mapping

### 2.1 Thread-to-Node Mapping (1 Thread per Node)
- **Kernel Function:** [`fgi::aggregate_baseline_kernel`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu)
- **Grid Configuration:**
  - Block Dimension: 256 threads (`dim3 block(256);`).
  - Grid Dimension: $\lceil N / 256 \rceil$ blocks (`dim3 grid((num_nodes + 255) / 256);`).
- **Global Index:**
  $$\text{node} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
  If $\text{node} \ge N$, the thread immediately exits.

### 2.2 CSR Traversal & Neighborhood Extraction
Each active thread reads the node's slice from `row_ptr`:
$$\text{start} = \text{row\_ptr}[\text{node}], \quad \text{end} = \text{row\_ptr}[\text{node} + 1], \quad d_i = \text{end} - \text{start}$$

### 2.3 Mathematical Semantics & Operator Support
1. **Sum Aggregation (`OP_SUM`):**
   $$Y_{i, k} = \sum_{j \in \mathcal{N}(i)} X_{j, k}$$
2. **Mean Aggregation (`OP_MEAN`):**
   $$Y_{i, k} = \frac{1}{d_i} \sum_{j \in \mathcal{N}(i)} X_{j, k}$$
3. **Max Aggregation (`OP_MAX`):**
   $$Y_{i, k} = \max_{j \in \mathcal{N}(i)} X_{j, k}$$

### 2.4 Strict Boundary & Sentinel Policies
- **Empty Neighborhoods ($d_i = 0$):**
  If $d_i \le 0$ (isolated node), the thread immediately writes $0.0\text{f}$ to all $D$ output columns:
  $$Y_{i, k} = 0.0\text{f} \quad \forall k \in [0, D-1]$$
  This eliminates division-by-zero for Mean aggregation and prevents $-\infty$ propagation for Max aggregation.
- **Negative-Only Neighborhoods in Max Aggregation:**
  To prevent invalid zero-clamping when all adjacent neighbors carry negative features (e.g. $[-50.0, -20.0]$), `max_val` is initialized directly from the first neighbor's feature:
  $$\text{max\_val} = X\Big[\text{col\_idx}[\text{start}], k\Big]$$
  and subsequently compared against remaining neighbors $\text{col\_idx}[\text{start}+1 \dots \text{end}-1]$.

---

## 3. Host-Side CUDA Orchestration

Implemented in [`backend/gpu_engine/cuda/src/main_baseline.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_baseline.cu):

### 3.1 64-bit Memory Safety
All byte allocations use 64-bit `size_t` arithmetic to prevent integer overflow:
```cpp
size_t bytes_row = (size_t)(num_nodes + 1) * sizeof(int);
size_t bytes_col = (size_t)num_edges * sizeof(int);
size_t bytes_feat = (size_t)num_nodes * (size_t)feature_dim * sizeof(float);
size_t bytes_out = bytes_feat;
```

### 3.2 High-Precision Decoupled Timing
Hardware CUDA events (`cudaEvent_t`) measure each execution phase separately:
1. **Host-to-Device Transfer ($t_{\text{H2D}}$):** Transfer of `row_ptr`, `col_idx`, and `features` to device memory.
2. **Warm-Up Phase:** 5 unrecorded kernel iterations to warm instruction caches and stabilize clock frequency.
3. **Pure Kernel Execution ($t_{\text{kernel}}$):** 20 recorded iterations measured via `cudaEventElapsedTime`.
4. **Device-to-Host Transfer ($t_{\text{D2H}}$):** Transfer of output tensor $Y$ back to host memory.
5. **End-to-End Elapsed Time ($t_{\text{e2e}}$):**
   $$t_{\text{e2e}} = t_{\text{H2D}} + \bar{t}_{\text{kernel}} + t_{\text{D2H}}$$

### 3.3 Error Handling & Cleanup
Every CUDA runtime call is wrapped in `CUDA_CHECK(call)`. Asynchronous kernel launches are verified via `CUDA_CHECK_KERNEL()`. Device pointers are safely released via `cudaFree()` across all execution paths.

---

## 4. Google Colab GPU Workflow

Due to the absence of local MSVC/`nvcc` compilers on the host workstation ([`ADR-003`](file:///D:/FraudGraph%20Intelligence/docs/DECISIONS.md)), FGI provides a turnkey cloud execution workflow:

```
+--------------------------------------------------------------------------------------------------+
|                                    COLAB WORKFLOW STEPS                                          |
+--------------------------------------------------------------------------------------------------+
| 1. Package Bundle:      python backend/gpu_engine/colab_runner/package_colab_bundle.py           |
|    Output:              experiments/colab_bundle_fgi_phase5.zip (3.55 MB)                        |
|                                                                                                  |
| 2. Open in Colab:       backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb                 |
|    Select GPU:          Runtime -> Change runtime type -> T4 GPU                                 |
|                                                                                                  |
| 3. Upload & Run:        Upload zip, execute run_colab_benchmarks.py                              |
|    Output:              fgi_phase5_gpu_results.zip                                               |
|                                                                                                  |
| 4. Ingest Results:      python backend/gpu_engine/colab_runner/ingest_colab_results.py <zip>     |
|    Output:              Validated JSON archives stored in experiments/results/                   |
+--------------------------------------------------------------------------------------------------+
```

### 4.1 Bundle Contents
The generated bundle (`experiments/colab_bundle_fgi_phase5.zip`) includes:
- Complete CUDA C++ sources: `fgi_baseline_kernels.cu`, `main_baseline.cu`, headers, and `Makefile`.
- Raw binary arrays for test fixtures:
  - `data/toy_graph_5nodes/` (5 nodes, 8 directed edges, $D=4$)
  - `data/synthetic_tier_a_10k/` (10,000 nodes, 94,945 directed edges, $D=32$)
- Precomputed CPU reference outputs (`ref_mean.bin`, `ref_sum.bin`, `ref_max.bin`).
- Automated driver: `run_colab_benchmarks.py` and notebook `fgi_colab_workflow.ipynb`.

---

## 5. Numerical Equivalence & Correctness Verification

### 5.1 Verification Threshold
$$\text{atol} = \max_{i, k} |Y_{\text{GPU}}[i, k] - Y_{\text{CPU}}[i, k]| \le 1.0 \times 10^{-5}$$

### 5.2 Correctness Gate Protocol
Before running benchmarks, the runner executes on `toy_graph_5nodes`:
1. Compares GPU float32 tensor against Phase 4 CPU reference.
2. Checks that isolated Node 4 strictly produces $\vec{0} \in \mathbb{R}^4$.
3. Verifies that negative feature values are correctly handled by Max aggregation.
4. Checks that zero NaNs or Infs exist.
5. If max absolute error $> 10^{-5}$, the run halts with exit code 2.

---

## 6. Empirical Tesla T4 GPU Benchmark Results & Speedup Analysis

The baseline CUDA engine was compiled with `nvcc` (CUDA compilation tools release 13.0, V13.0.88; flags: `-O3 -std=c++17`) and executed on a physical cloud GPU runtime via Google Colab.

### 6.1 GPU Hardware Telemetry
- **Device:** NVIDIA Tesla T4
- **CUDA Compiler:** CUDA compilation tools release 13.0, V13.0.88
- **Compute Capability:** 7.5 (Turing architecture)
- **Total VRAM:** 14,912.7 MB (~14.9 GB GDDR6)
- **Streaming Multiprocessors (SMs):** 40
- **Theoretical Peak Bandwidth:** 320.0 GB/s

### 6.2 Stage 1 Correctness Gate (`toy_graph_5nodes`, $N=5, M=8, D=4$)
Executed with 2 warmup iterations and 5 recorded repetitions:
| Operator | Status | Max Absolute Error | Tolerance Threshold | Result File |
| :--- | :--- | :--- | :--- | :--- |
| **Sum** | **PASSED** | $0.0$ | $1.0 \times 10^{-5}$ | [`val_toy_sum.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/val_toy_sum.json) |
| **Mean** | **PASSED** | $0.0$ | $1.0 \times 10^{-5}$ | [`val_toy_mean.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/val_toy_mean.json) |
| **Max** | **PASSED** | $0.0$ | $1.0 \times 10^{-5}$ | [`val_toy_max.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/val_toy_max.json) |

*Observation:* Isolated node 4 strictly evaluated to $\vec{0} \in \mathbb{R}^4$, and negative feature values in Max aggregation matched the CPU reference without numerical divergence.

### 6.3 Stage 2 Scale Benchmarks (`synthetic_tier_a_10k`, $N=10,000, M=94,945, D=32$)
Executed with 5 warmup iterations and 20 recorded repetitions:

| Operator | $t_{\text{H2D}}$ (ms) | $t_{\text{kernel}}$ Mean $\pm$ Std (ms) | $t_{\text{kernel}}$ p50 (ms) | $t_{\text{D2H}}$ (ms) | $t_{\text{e2e}}$ Mean (ms) | Throughput (edges/sec) | Effective BW (GB/s) | Max Abs Err |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sum** | 0.539 | **0.0681 $\pm$ 0.0007** | 0.0677 | 0.325 | 0.931 | 1,394,380,000 | 203.44 | $0.0$ (PASSED) |
| **Mean** | 0.558 | **0.0682 $\pm$ 0.0007** | 0.0676 | 0.329 | 0.955 | 1,393,010,000 | 203.24 | $0.0$ (PASSED) |
| **Max** | 0.584 | **0.0673 $\pm$ 0.0022** | 0.0670 | 0.354 | 1.005 | 1,410,790,000 | 205.84 | $0.0$ (PASSED) |

### 6.4 Comparative Speedup Analysis vs Phase 4 CPU Reference

Hardware comparison baseline:
- **CPU Reference:** Intel Core i5-1135G7 @ 2.40GHz (4 cores, 8 threads, AVX-512).
- **GPU Baseline:** NVIDIA Tesla T4 (40 SMs, 2,560 CUDA cores).

| Aggregation Operator | Phase 4 CPU Node Loop (ms) | Phase 4 CPU Vectorized SpMM (ms) | Phase 5 GPU Kernel Only (ms) | Phase 5 GPU End-to-End (ms) | **Pure Kernel Speedup vs CPU Ref** | **Pure Kernel Speedup vs CPU SpMM** | **End-to-End Speedup vs CPU Ref** | **End-to-End Speedup vs CPU SpMM** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sum** | 166.877 | 3.624 | **0.0681** | 0.931 | **2,450.8$\times$** | **53.2$\times$** | **179.2$\times$** | **3.89$\times$** |
| **Mean** | 133.177 | 6.325 | **0.0682** | 0.955 | **1,953.9$\times$** | **92.8$\times$** | **139.4$\times$** | **6.62$\times$** |
| **Max** | 167.173 | *N/A (unsupported)* | **0.0673** | 1.005 | **2,484.0$\times$** | *N/A* | **166.3$\times$** | *N/A* |

### 6.5 Architectural Findings from Measured Telemetry
1. **PCIe Transfer Dominance (Amdahl's Law in Practice):**
   - For a single one-off aggregation of 10K nodes, pure CUDA kernel execution ($t_{\text{kernel}} \approx 0.068\text{ ms}$) represents only **7.1%** of total end-to-end latency ($t_{\text{e2e}} \approx 0.955\text{ ms}$). The remaining **92.9%** is consumed by host-to-device ($0.55\text{ ms}$) and device-to-host ($0.33\text{ ms}$) PCIe transfers.
   - This validates the requirement in `docs/EXPERIMENT_PROTOCOL.md` and `ADR-007` to decouple kernel latency from PCIe transfer latency, as in iterative GNN training or multi-hop aggregation the graph resides persistently in GPU VRAM across layers.
2. **Effective Memory Bandwidth Utilization:**
   - The baseline 1-thread-per-node kernel achieves an estimated **203.4–205.8 GB/s** effective memory bandwidth on synthetic Tier A based on the analytical data movement model, which represents **63.6–64.3%** of the Tesla T4's theoretical peak bandwidth of 320.0 GB/s.
   - While degree skew and uncoalesced memory reads are present, launching 10,000 threads provides sufficient memory-level parallelism to achieve substantial bus utilization; however, this does not imply full hardware, compute, or memory saturation.

---

## 7. Baseline Performance Bottlenecks (Research Baseline for Phase 6)

The baseline kernel exposes fundamental GPU architectural bottlenecks:
1. **Uncoalesced Memory Reads:** Each thread iterates through its node's neighbor list sequentially. Adjacent threads within a warp read features of unrelated neighbor nodes located at non-contiguous global memory addresses, causing severe memory bus underutilization.
2. **SIMT Warp Divergence:** In power-law graphs, degree varies widely (e.g., $d_i \in [0, 433]$). Threads with low degrees complete quickly and idle while high-degree hub nodes in the same warp continue looping.
3. **Absence of Shared-Memory Caching:** High-degree hub neighbor features are repeatedly re-read from global memory rather than being cached in on-chip SRAM (`__shared__`).

*These identified bottlenecks form the precise research foundation for Phase 6 (Memory-Aware CUDA Optimization).*
