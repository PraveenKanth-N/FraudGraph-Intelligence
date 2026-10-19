# Experiment 6.1 Specification & Implementation Report
## Warp-Coalesced Neighbor Aggregation — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Workspace:** `D:\FraudGraph Intelligence`  
**Baseline Commit:** `dcebf05` (Phase 5 Complete)  
**Target Hardware:** NVIDIA Tesla T4 (`sm_75`, 40 SMs, 14.9 GB VRAM, 320.0 GB/s Peak Bandwidth)  
**Compiler Baseline:** CUDA compilation tools release 13.0, V13.0.88 (`-O3 -std=c++17`)  
**Status:** PASSED (Complete)  

---

## 1. Executive Summary & Objective

Experiment 6.1 implements the first memory-aware optimization of Phase 6: **Warp-Coalesced Neighbor Aggregation**.
Its primary objective is to eliminate the severe DRAM read transaction amplification observed in the Phase 5 baseline by restructuring the thread-to-data mapping so that memory loads from global DRAM are 100% coalesced into contiguous 128-byte cache lines.

### Critical Implementation Boundaries
1. **Baseline Preservation:** [`backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu) and its standalone runner [`main_baseline.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_baseline.cu) remain **strictly untouched**.
2. **Dedicated Candidate Files:** The warp-coalesced engine is implemented entirely in:
   - Header: [`backend/gpu_engine/cuda/include/fgi_coalesced_kernels.cuh`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/include/fgi_coalesced_kernels.cuh)
   - Kernel: [`backend/gpu_engine/cuda/src/fgi_coalesced_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_coalesced_kernels.cu)
   - Standalone CLI Driver: [`backend/gpu_engine/cuda/src/main_coalesced.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_coalesced.cu)
3. **Multi-Target Build:** [`backend/gpu_engine/cuda/Makefile`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/Makefile) compiles both binaries concurrently:
   - `bin/fgi_baseline_runner`
   - `bin/fgi_coalesced_runner`

---

## 2. Kernel Design & Theoretical Analysis

### 2.1 Thread Mapping (1 Warp per Node)
- **Grid Configuration:** Block size defaults to 256 threads (8 warps per block).
  $$\text{warps\_per\_block} = \frac{\text{block\_size}}{32} = 8$$
  $$\text{grid\_size} = \left\lceil \frac{N}{\text{warps\_per\_block}} \right\rceil = \left\lceil \frac{N}{8} \right\rceil$$
- **Global Indexing:**
  $$\text{global\_warp\_id} = \frac{\text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}}{32}$$
  $$\text{lane\_id} = \text{threadIdx.x} \pmod{32}$$
  Each warp is assigned to process output node $i = \text{global\_warp\_id}$.

### 2.2 100% Memory Coalescing Mechanism
For canonical feature dimension $D = 32$ (128 bytes per feature row):
- Lane $t \in [0, 31]$ is mapped to feature column $k = t$.
- When the warp accesses neighbor $j = \text{col\_idx}[p]$, lane $t$ reads:
  $$\text{address}_t = \text{features} + (j \times 32 + t) \times 4\text{ bytes}$$
- The 32 threads in the warp request 32 consecutive 4-byte floating-point numbers:
  $$[j \cdot 128, \quad j \cdot 128 + 4, \quad \dots, \quad j \cdot 128 + 124]\text{ bytes}$$
- Because this request is 128-byte contiguous and 128-byte aligned, the hardware memory controller services the entire warp request with **exactly one 128-byte DRAM cache-line transaction** (compared to up to 32 separate transactions in the baseline).

### 2.3 General Feature Dimension Support ($D \ne 32$)
To prevent failures on non-canonical graphs (e.g. `toy_graph_5nodes` where $D=4$, or high-dimensional embeddings where $D=128$), the kernel uses a column-strided loop:
```cpp
for (int k = lane_id; k < feature_dim; k += 32) {
    // Process column k
}
```
- For $D=4$: Lanes $0 \dots 3$ process columns $0 \dots 3$; lanes $4 \dots 31$ naturally exit the loop without memory updates.
- For $D=128$: Each lane processes 4 sequential tiles ($k, k+32, k+64, k+96$), maintaining 128-byte coalescing for every tile.

### 2.4 Strict Boundary & Correctness Handling
1. **Empty Neighborhoods ($d_i = 0$):**
   If $d_i \le 0$ (isolated node), all 32 lanes cooperate to write `0.0f` to their assigned columns:
   ```cpp
   for (int k = lane_id; k < feature_dim; k += 32) {
       output[out_node_offset + k] = 0.0f;
   }
   ```
2. **Negative-Valued Max Aggregation:**
   When $d_i > 0$, `max_val` is initialized directly from the first neighbor's feature:
   $$\text{max\_val} = X\Big[\text{col\_idx}[\text{start}], k\Big]$$
   and then iteratively compared against subsequent neighbors $p \in [\text{start}+1, \text{end})$. This prevents invalid zero-clamping when all incident neighbors carry strictly negative features.

### 2.5 Divergence Analysis (Measured Facts vs Precision)
- **Intra-Warp Divergence: ZERO.** Because all 32 lanes in warp $i$ process the *same* node $i$, they share the exact same loop bounds $[\text{start}_i, \text{end}_i)$. There is zero branch divergence across execution lanes during edge traversal.
- **Inter-Warp Divergence: STILL PRESENT.** Different warps in the same thread block process different nodes $i$ and $i'$. If warp 0 processes a node with $d_0 = 1$ and warp 1 processes a hub with $d_1 = 433$, warp 0 finishes in 1 iteration while warp 1 loops 433 times. Although the Turing hardware warp scheduler switches to other active warps, block retirement is bounded by the longest-running warp in that block. This inter-warp load imbalance remains an active subject for Candidate 4 (Degree-Aware Bucketing).

---

## 3. Two-Tier Verification & Google Colab Workflow

Since the local Windows host lacks `nvcc` and MSVC build tools, execution follows the verified Two-Tier Protocol:

### 3.1 Tier 1: Local Host Verification (Complete)
- Python regression suite: **60 passed, 1 skipped** (`test_cuda_coalesced.py` adds 5 dedicated tests).
- Frontend production build: **0 errors in 12.56s**, typecheck **0 errors**.
- Phase 6 Colab bundle: [`experiments/colab_bundle_fgi_phase6.zip`](file:///D:/FraudGraph%20Intelligence/experiments/colab_bundle_fgi_phase6.zip) (3.56 MB) packaged with both baseline and coalesced sources, headers, Makefile, raw graph arrays, and precomputed references.

### 3.2 Tier 2: Google Colab GPU Benchmark Protocol (Pending User Execution)
1. **Open Google Colab:** [colab.research.google.com](https://colab.research.google.com).
2. **Set Runtime:** `Runtime -> Change runtime type -> T4 GPU` (free tier).
3. **Upload Bundle:** Upload `experiments/colab_bundle_fgi_phase6.zip` to Colab.
4. **Open & Run Notebook:** Run cells in [`backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb):
   - Cell 1: Verifies GPU hardware via `nvidia-smi` and compiler via `nvcc --version`.
   - Cell 2: Extracts `colab_bundle_fgi_phase6.zip`.
   - Cell 3: Executes `run_colab_benchmarks.py`, compiling both binaries, running Stage 1 correctness on `toy_graph_5nodes` ($\text{atol} \le 10^{-5}$), and executing the 20-repetition benchmark on `synthetic_tier_a_10k`.
   - Cell 4: Automatically downloads `fgi_phase6_gpu_results.zip`.
5. **Local Ingestion:**
   ```powershell
   .\.venv\Scripts\python.exe backend/gpu_engine/colab_runner/ingest_colab_results.py <path_to_downloaded_fgi_phase6_gpu_results.zip>
   ```

### 3.3 CUDA 13.0 Architecture Compatibility Repair
- **Issue Encountered:** Google Colab uses CUDA compilation tools release 13.0 (V13.0.88). In CUDA 13.0, support for Volta architecture (`compute_70`) was deprecated and removed from `nvcc`, producing `nvcc fatal: Unsupported gpu architecture 'compute_70'`.
- **Resolution Applied:** Refactored `backend/gpu_engine/cuda/Makefile` to strictly target the Tesla T4 architecture:
  ```makefile
  CUDA_ARCH ?= -gencode arch=compute_75,code=sm_75
  ```
  Removed deprecated `compute_70` and extraneous `compute_80` flags.
- **Fixture Re-Verification:** Verified that the canonical Tier A benchmark fixture remains invariant with $N=10,000$ nodes, $M=94,945$ directed CSR edges, and $D=32$ features. No conflicting values (such as $M=100,000$) override the binary graph inputs.
- **Bundle Regeneration:** Rebuilt `experiments/colab_bundle_fgi_phase6.zip` with the updated Makefile, verified both `fgi_baseline_runner` and `fgi_coalesced_runner` targets, and retained the strict numerical tolerance $\text{atol} \le 1.0 \times 10^{-5}$.

### 3.4 Measured Google Colab Tesla T4 Results & Audit

The physical Google Colab revalidation run on an **NVIDIA Tesla T4** (`sm_75`, 40 SMs, 14.9 GB VRAM, Driver 580.82.07, CUDA 13.0, V13.0.88) was executed on October 19, 2026. All 6 toy-graph correctness validations and all 6 Tier A benchmarks completed with 100% numerical accuracy ($\text{max\_abs\_err} = 0.00 \le 1.0 \times 10^{-5}$).

#### Revalidation Performance Matrix (Tier A: $N=10,000$, $M=94,945$, $D=32$, 20 Repetitions, 5 Warmup)
*Timestamp: `20261019_175441` on NVIDIA Tesla T4 / CUDA 13.0*

| Operator | Baseline Kernel (ms) | Coalesced Kernel (ms) | **Pure Kernel Speedup** | Baseline E2E (ms) | Coalesced E2E (ms) | **End-to-End Speedup** | Edge Throughput (Coalesced) | Validation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MEAN** | $0.06892 \text{ ms}$ | $0.01853 \text{ ms}$ | **$3.72\times$** ($3.718\times$) | $0.96623 \text{ ms}$ | $0.97857 \text{ ms}$ | **$0.99\times$** ($0.987\times$) | $5.12\times 10^9 \text{ edges/s}$ | $\text{err}=0.00 \le 10^{-5}$ (`PASSED`) |
| **SUM**  | $0.06860 \text{ ms}$ | $0.01684 \text{ ms}$ | **$4.07\times$** ($4.072\times$) | $0.95632 \text{ ms}$ | $0.87603 \text{ ms}$ | **$1.09\times$** ($1.092\times$) | $5.64\times 10^9 \text{ edges/s}$ | $\text{err}=0.00 \le 10^{-5}$ (`PASSED`) |
| **MAX**  | $0.06726 \text{ ms}$ | $0.01670 \text{ ms}$ | **$4.03\times$** ($4.027\times$) | $0.94608 \text{ ms}$ | $0.88451 \text{ ms}$ | **$1.07\times$** ($1.070\times$) | $5.68\times 10^9 \text{ edges/s}$ | $\text{err}=0.00 \le 10^{-5}$ (`PASSED`) |

*Comparison with Initial Run (`20261019_160820`):*
- Initial run: Mean kernel speedup $3.94\times$ (E2E $1.05\times$), Sum kernel speedup $4.04\times$ (E2E $1.10\times$), Max kernel speedup $3.96\times$ (E2E $1.02\times$).
- Revalidation run: Mean kernel speedup $3.72\times$ (E2E $0.99\times$), Sum kernel speedup $4.07\times$ (E2E $1.09\times$), Max kernel speedup $4.03\times$ (E2E $1.07\times$).
- Both runs demonstrate consistent $\sim 3.7\times - 4.1\times$ pure kernel acceleration from warp coalescing across all operators.

*Raw Telemetry Files Archived in `experiments/results/`:*
- Revalidation Run 2 (`20261019_175441`): `exp_gpu_*_tier_a_*_20261019_175441.json` (6 files) + `val_toy_*_20261019_175441.json` (6 files)
- Initial Run 1 (`20261019_160820`): `exp_gpu_*_tier_a_*_20261019_160820.json` (6 files) + `val_toy_*_20261019_160820.json` (6 files)
- Archives preserved: [`experiments/fgi_phase6_gpu_results.zip`](file:///D:/FraudGraph%20Intelligence/experiments/fgi_phase6_gpu_results.zip), [`experiments/fgi_phase6_revalidation_gpu_results_20261019_175441.zip`](file:///D:/FraudGraph%20Intelligence/experiments/fgi_phase6_revalidation_gpu_results_20261019_175441.zip), and [`experiments/fgi_phase6_gpu_results_20261019_160820.zip`](file:///D:/FraudGraph%20Intelligence/experiments/fgi_phase6_gpu_results_20261019_160820.zip).

#### 3.5 Bandwidth Audit & Analytical Estimation Disclosures

The raw metrics report both compulsory and algorithmic bandwidth figures. We explicitly establish the scientific boundaries of these numbers:

1. **Analytical Estimates, Not Measured DRAM Traffic:**
   - Both `estimated_compulsory_bandwidth_gbps` and `estimated_algorithmic_bandwidth_gbps` are **analytical model calculations** ($V_{\text{bytes}} / (t_{\text{kernel}} \times 10^9)$).
   - They are **not** sampled hardware performance counters (e.g., from NVIDIA Nsight Compute, `nvprof`, or physical memory-bus probes).
   - They **do not claim or establish** physical memory-bus utilization or bus saturation.
2. **Estimated Compulsory DRAM Bandwidth:**
   - Evaluates the unique physical footprint transferred across the bus once:
     $$V_{\text{compulsory}} = (N+1)\times 4 + M \times 4 + (N \times D \times 4) + (N \times D \times 4) = 2,979,784\text{ bytes} \approx 2.98\text{ MB}$$
   - **Baseline:** $43.24 - 44.30\text{ GB/s}$ ($13.5\% - 13.8\%$ of Tesla T4 $320\text{ GB/s}$ peak).
   - **Coalesced:** $160.77 - 178.37\text{ GB/s}$ ($50.2\% - 55.7\%$ of Tesla T4 $320\text{ GB/s}$ peak).
   - **Interpretation:** Coalescing improves effective memory request density by $\sim 3.7\times - 4.1\times$, reducing uncoalesced DRAM transaction waste.
3. **Estimated Algorithmic Memory Work Rate (Un-Cached Model):**
   - Counts reading $D=32$ floats for every directed edge ($M = 94,945$):
     $$V_{\text{algo}} = (N+1)\times 4 + M \times 4 + (M \times D \times 4) + (N \times D \times 4) = 13,852,744\text{ bytes} \approx 13.85\text{ MB}$$
   - **Baseline:** $201.00 - 205.94\text{ GB/s}$.
   - **Coalesced:** $747.41 - 829.23\text{ GB/s}$.
   - **Physical Reality:** The $747 - 829\text{ GB/s}$ work rate exceeds the T4's $320\text{ GB/s}$ DRAM peak because the $1.28\text{ MB}$ feature table fits inside the Tesla T4's **$4.0\text{ MB}$ unified on-chip L2 cache**. Repeated edge lookups hit L2 cache at multi-TB/s on-chip bandwidth rather than traversing off-chip DRAM.

#### 3.6 Decoupled PCIe vs Kernel Timing Analysis (Amdahl's Law)

A rigorous comparison between kernel execution and end-to-end wall-clock time reveals the dominant role of host-to-device and device-to-host PCIe transfers:

- **Mean Operator Overhead & Transfer Jitter:**
  - Mean kernel execution time dropped from $0.06892\text{ ms}$ to $0.01853\text{ ms}$ (**$3.72\times$ acceleration**, saving $0.0504\text{ ms}$).
  - However, PCIe transfers accounted for $0.8973\text{ ms}$ ($92.9\%$) in baseline and $0.9600\text{ ms}$ ($98.1\%$) in coalesced.
  - A slight host/PCIe transmission variance of $+0.0627\text{ ms}$ on the virtualized Colab environment exceeded the $0.0504\text{ ms}$ kernel time reduction, causing end-to-end time to measure $0.97857\text{ ms}$ vs $0.96623\text{ ms}$ (**$0.99\times$ speedup**).
- **Sum and Max Operations:**
  - Sum: Kernel improved by $4.07\times$ ($0.06860\text{ ms} \rightarrow 0.01684\text{ ms}$), and steady PCIe transfers enabled an end-to-end speedup of **$1.09\times$** ($0.95632\text{ ms} \rightarrow 0.87603\text{ ms}$).
  - Max: Kernel improved by $4.03\times$ ($0.06726\text{ ms} \rightarrow 0.01670\text{ ms}$), yielding an end-to-end speedup of **$1.07\times$** ($0.94608\text{ ms} \rightarrow 0.88451\text{ ms}$).
- **Scientific Conclusion:**
  - We explicitly **do not claim that all operations improved end to end**.
  - Because un-pinned synchronous PCIe transfers consume $93\% - 98\%$ of total wall-clock time, Amdahl's Law dictates that kernel optimizations alone cannot significantly improve whole-pipeline wall-clock latency without addressing data movement.
  - This empirical evidence formally motivates Candidate 7 (Pinned Host Memory and Asynchronous CUDA Streams) in Phase 6 to overlap transfers with computation.

#### 3.7 Software Defect Repair & Format Standardization
- **Header & Namespace Resolution:** Added `#include <cstdint>` and explicit `std::uint64_t` typing to both `main_baseline.cu` and `main_coalesced.cu`, fixing GCC/CUDA 13.0 compilation errors on Google Colab.
- **Printf Format Standard:** Replaced non-standard POSIX `%12,.0f` with `format_with_commas` helper and standard `%15s edges/sec`.
- **Metric Segregation:** Separated estimated compulsory bandwidth and estimated algorithmic bandwidth in console logs and JSON outputs.

#### 3.8 Scope & Scientific Integrity Boundary
- **Neighbor Aggregation Only:** This experiment strictly measures parallel neighbor aggregation performance on CSR graph representations.
- **No Classification Claims:** This experiment does **not** evaluate machine learning fraud classification accuracy (PR-AUC, F1-score, Precision, Recall). Feature aggregation speedup does not imply changes in classification accuracy; downstream fraud detection evaluation belongs exclusively to Phase 7.

---

## 4. Phase Gate Status

- **Experiment 6.1 Implementation:** **COMPLETE**
- **GPU Revalidation Gate:** **PASSED (Verified on NVIDIA Tesla T4 / CUDA 13.0)**
- **Phase 6.1 Status:** **`PASSED (Experiment 6.1 Complete)`**
