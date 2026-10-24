# Experiment 6.3 Specification & Design Plan
## Degree-Aware Bucketing & Hybrid CUDA Dispatch — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming
**Workspace:** `D:\FraudGraph Intelligence`
**Baseline Commit:** `ff77644` (Phase 6.2 Complete)
**Target Hardware:** NVIDIA Tesla T4 (`sm_75`, 40 SMs, 14.9 GB VRAM, 320.0 GB/s Peak Bandwidth)
**Status:** COMPLETE & EMPIRICALLY VALIDATED (Stages 1–6 Complete)

---

## 1. Executive Summary & Research Hypothesis

### 1.1 Problem Statement: SIMT Inter-Warp Degree Divergence
In Experiment 6.1, the warp-coalesced aggregation kernel assigned **1 warp (32 threads) per node**. This design successfully eliminated intra-warp column divergence and achieved 100% memory coalescing for canonical $D=32$ feature matrices, accelerating pure kernel latency by $\sim 4.0\times$ over the Phase 5 baseline.

However, financial transaction networks exhibit power-law degree distributions characterized by severe structural heterogeneity:
- A large fraction of nodes have very low degree ($d_i \in [0, 8]$).
- A tiny fraction of nodes are heavy hubs ($d_i \gg 32$, up to thousands of transactions).

When a thread block (e.g. 256 threads = 8 warps) executes on an SM, block retirement is bounded by the slowest, longest-running warp in that block:
$$\text{Block Execution Time} = \max_{w \in [0, 7]} t_w$$
If warp 0 is assigned a high-degree merchant hub ($d_0 = 500$), while warps $1 \dots 7$ process sparse consumers ($d_{1 \dots 7} \le 5$), warps $1 \dots 7$ complete in 5 iterations and remain idle for 495 iterations while warp 0 completes. This creates severe **inter-warp load imbalance** and underutilizes GPU compute resources.

### 1.2 Research Hypothesis
> **Hypothesis ($H_{\text{6.3}}$):**
> *Partitioning nodes by degree into distinct topological regimes (isolated, standard, and hub) and dispatching specialized execution geometries (direct zero-initialization, 1-warp-per-node coalescing, and cooperative multi-warp block-level hub reduction) will reduce execution variance, improve SM load balance, and accelerate pure kernel and end-to-end latency on power-law graphs compared to a monolithic warp-per-node kernel.*

Treat this strictly as a research hypothesis to test, not a guaranteed speedup. If bucketing overheads or kernel launch latencies outweigh execution gains, a documented negative result is valid scientific output.

---

## 2. Degree Distribution Audit Across Fixtures

We audited the out-degree distributions ($d_i = \text{row\_ptr}[i+1] - \text{row\_ptr}[i]$) across all available project fixtures:

| Graph Metric | `toy_graph_5nodes.npz` | `synthetic_tier_a_10k.npz` (Canonical) | `synthetic_skewed_10k.npz` (Stress) |
| :--- | :--- | :--- | :--- |
| **Node Count ($N$)** | $5$ | $10,000$ | $10,000$ |
| **Edge Count ($M$)** | $8$ | $94,945$ | $94,970$ |
| **Feature Dimension ($D$)** | $4$ | $32$ | $32$ |
| **Minimum Degree** | $0$ | $0$ | $0$ |
| **Maximum Degree** | $3$ | $10$ | **$6,340$** |
| **Mean Degree ($\bar{d}$)** | $1.60$ | $9.49$ | $9.50$ |
| **Median Degree (p50)** | $2.0$ | $10.0$ | $5.0$ |
| **Degree Percentiles (p25 / p75 / p90 / p95 / p99)** | $1.0$ / $2.0$ / $2.6$ / $2.8$ / $2.96$ | $10.0$ / $10.0$ / $10.0$ / $10.0$ / $10.0$ | $5.0$ / $5.0$ / $6.0$ / $8.0$ / **$21.0$** |
| **Isolated Nodes ($d_i = 0$)** | $1$ ($20.00\%$) | $501$ ($5.01\%$) | $500$ ($5.00\%$) |
| **Low-Degree Nodes ($d_i \in [1, 4]$)** | $4$ ($80.00\%$) | $4$ ($0.04\%$) | $0$ ($0.00\%$) |
| **Medium-Degree Nodes ($d_i \in [5, 16]$)** | $0$ ($0.00\%$) | $9,495$ ($94.95\%$) | $9,362$ ($93.62\%$) |
| **Transition Nodes ($d_i \in [17, 32]$)** | $0$ ($0.00\%$) | $0$ ($0.00\%$) | $82$ ($0.82\%$) |
| **Hub Nodes ($d_i > 32$)** | $0$ ($0.00\%$) | $0$ ($0.00\%$) | **$56$ ($0.56\%$)** |
| **Extreme Hubs ($d_i > 128$)** | $0$ ($0.00\%$) | $0$ ($0.00\%$) | **$19$ ($0.19\%$, max $6,340$)** |

### Key Findings from Graph Audit:
1. **Canonical `synthetic_tier_a_10k` is Structurally Quasi-Uniform:**
   - 94.95% of nodes have degree exactly 10, and 5.01% have degree 0.
   - Because maximum degree is 10, inter-warp divergence across non-zero nodes is already near zero. The primary optimization opportunity is bypassing edge loops for the 501 isolated nodes.
2. **Stress Fixture `synthetic_skewed_10k` Exhibits Extreme Skew:**
   - Possesses a 6,340-edge super-hub and 56 nodes with degree $> 32$.
   - A single monolithic warp looping 6,340 times will bottleneck any thread block that contains it, providing the ideal stress test for degree-aware multi-warp hub processing.

---

## 3. Experimental Variants

We define three execution variants for direct comparative benchmarking under identical input arrays and measurement conditions:

### Variant 1: Monolithic Warp-per-Node Coalesced Kernel (Control)
- **Reference Implementation:** From Phase 6.1 / 6.2 ([`fgi_coalesced_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_coalesced_kernels.cu)).
- **Mechanism:** Grid maps $N$ warps linearly across nodes $i \in [0, N-1]$. Every warp executes the edge loop $[\text{start}_i, \text{end}_i)$ regardless of degree.

### Variant 2: Degree-Aware Hybrid Dispatch (Candidate 6.3)
- **Bucketing Structure:**
  During graph ingestion or preprocessing, node indices are partitioned into three mutually exclusive subsets:
  1. **Zero-Degree Bucket ($d_i = 0$):** Output tensor is initialized via fast `cudaMemsetAsync(d_output, 0, bytes_out)` on stream, setting exact FGI $\vec{0}$ sentinels with zero instruction overhead.
  2. **Standard Bucket ($1 \le d_i \le \text{THRESHOLD}$):**
     - Contains nodes with bounded degrees (default $\text{THRESHOLD} = 32$).
     - Processed by a dedicated bucket-coalesced kernel where 1 warp processes node `bucket_std[idx]`.
     - Because all nodes in this bucket share bounded loop iterations ($1 \dots 32$), warp retirement within blocks is tightly synchronized.
  3. **Hub Bucket ($d_i > \text{THRESHOLD}$):**
     - Contains dense nodes ($d_i > 32$).
     - Processed by a dedicated **Block-per-Hub Cooperative Kernel**:
       - 1 thread block (256 threads = 8 warps) is assigned to a single hub node $i = \text{bucket\_hub}[b]$.
       - The 8 warps interleave the neighbor edge list in parallel:
         $$\text{for } p = \text{start} + \text{warp\_id}; \quad p < \text{end}; \quad p += 8$$
       - Each warp accumulates 128-byte coalesced reads into private registers.
       - A fast shared-memory reduction (`extern __shared__ float s_acc[8][32]`) aggregates the 8 partial results down to 1 final row written to global memory.
       - **Theoretical Benefit:** Reduces sequential iteration count on super-hubs by **$8\times$** (e.g. $6,340 \rightarrow 793$ iterations).

### Variant 3: Monolithic 1-Thread-per-Node Baseline (Phase 5 Baseline)
- **Reference Implementation:** From Phase 5 ([`fgi_baseline_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu)).
- **Purpose:** Establishes the lower-bound reference to quantify the SIMT divergence spectrum.

---

## 4. Dispatch Thresholds & Tuning Strategy

- **Default Threshold:** $\text{THRESHOLD} = 32$.
  - Rationale: A 32-thread warp processing $D=32$ reads 1 neighbor per iteration. When $d_i \le 32$, a single warp finishes in $\le 32$ iterations. When $d_i > 32$, allocating 8 warps ensures each warp processes at least $\approx 4$ edges, amortizing the block reduction overhead.
- **Tuning Sweep:** We evaluate $\text{THRESHOLD} \in \{16, 32, 64\}$ to observe the performance boundary between 1-warp-per-node and cooperative multi-warp execution.
- **Independence:** Thresholds are derived strictly from graph topological degree distributions without reference to financial fraud labels or prediction outcomes.

---

## 5. Correctness & Precision Requirements

- **Numerical Equivalence:**
  The output tensor $Y \in \mathbb{R}^{N \times D}$ must match the CPU reference implementation ([`backend/cpu_engine/aggregators.py`](file:///D:/FraudGraph%20Intelligence/backend/cpu_engine/aggregators.py)) with absolute error:
  $$\text{max\_abs\_error} \le 1.0 \times 10^{-5}$$
- **Operator Invariance:**
  Must support Sum, Mean, and Max with exact FGI zero-sentinels for empty neighborhoods and negative-only neighborhoods for Max.
- **Index Preservation:**
  The hybrid dispatch must place outputs in exact 0-indexed node order ($Y_{i, :}$) regardless of which bucket processed node $i$.

---

## 6. Implementation & Benchmark Methodology

1. **Source Isolation:**
   - Header: `backend/gpu_engine/cuda/include/fgi_degree_aware_kernels.cuh`
   - Kernel: `backend/gpu_engine/cuda/src/fgi_degree_aware_kernels.cu`
   - Standalone CLI: `backend/gpu_engine/cuda/src/main_degree_aware.cu`
   - Makefile Target: `fgi_degree_aware_runner`
   - Existing Phase 5, 6.1, and 6.2 source files remain **100% untouched**.
2. **Benchmark Protocol:**
   - 5 warmup iterations, 20 recorded repetitions.
   - Pinned host memory (`cudaMallocHost`) and dedicated non-blocking CUDA stream (`cudaStreamNonBlocking`) to isolate algorithmic kernel speedup without transfer jitter.
   - Decoupled timing: $t_{\text{bucketing}}$, $t_{\text{H2D}}$, $t_{\text{kernel}}$, $t_{\text{D2H}}$, $t_{\text{e2e}}$.
   - Preprocessing and bucketing overheads are explicitly measured and included in end-to-end wall-clock latency.

---

## 7. Stage 3 Verification & Local Test Suite Status

- **Focused Unit Tests (`backend/tests/test_cuda_degree_aware.py`):**
  - `test_degree_aware_cuda_source_files_exist`: PASSED (verified all source files, headers, token invariants).
  - `test_degree_aware_makefile_targets`: PASSED (verified Makefile multi-target compilation integrity).
  - `test_degree_aware_runner_interface`: PASSED (verified Python wrapper `get_cuda_binary_path` support).
  - `test_package_colab_bundle_phase6_includes_degree_aware_and_skewed`: PASSED (verified bundle packaging).
  - `test_degree_aware_result_json_schema`: PASSED (validated JSON schema against `EXP-GPU-DEGREE-AWARE-OPT-6.3`).
  - `test_degree_bucketing_partition_invariants`: PASSED (mathematical partition disjointness & union completeness).
  - `test_cpu_reference_parity_on_skewed_graph`: PASSED (CPU reference numerical equivalence on 6,340 super-hub).
  - `test_extreme_hub_stress_graph`: PASSED (exact zero sentinels on isolated nodes, $\text{atol} \le 1.0 \times 10^{-5}$).
- **Full Backend Regression Suite:** 80 passed, 1 skipped (0 failures).
- **Frontend Verification:** `npm run typecheck` passed (0 errors); `npm run build` passed (0 errors).

---

## 8. Colab GPU Benchmark Execution Environment (Stage 4)

- **Target Hardware:** NVIDIA Tesla T4 (`sm_75`, Compute Capability 7.5, 40 SMs, 15,360 MiB VRAM)
- **Host & Driver Environment:** Linux, NVIDIA Driver `580.82.07`, CUDA Compiler `13.0` (`V13.0.88`, `r13.0/compiler.36424714_0`)
- **Execution Timestamp:** `20261024_142037` (2026-10-24 14:20:37 UTC / 19:50:37 IST)
- **Results Archive:** `experiments/fgi_phase6_degree_aware_gpu_results_20261024_142037.zip` (preserved and archived)
- **Measurement Protocol:** 5 warmup iterations, 20 recorded repetitions per benchmark configuration.
- **Toy Graph Correctness Gate (`toy_graph_5nodes`, 12 checks):** 100% PASSED across all 4 engines and all 3 operators (`mean`, `sum`, `max`) with `max_absolute_error == 0.00e+00`.

---

## 9. Empirical Benchmark Results & Quantitative Analysis (Stage 5)

### 9.1 Comparative Matrix: Heavy-Tailed Skewed Stress Graph (`synthetic_skewed_10k`)
$N = 10,000$ nodes, $M = 94,970$ edges, $D = 32$ features, Max Degree = **$6,340$ (Super-Hub)**:

| Operator | Engine Architecture | Kernel Latency ($t_{\text{kernel}}$) | Kernel Speedup vs Base | Transfer Latency (H2D + D2H) | End-to-End Latency ($t_{\text{e2e}}$) | E2E Speedup vs Base | E2E Speedup vs Streams | Verification ($\text{atol} \le 10^{-5}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MEAN** | Phase 5 Baseline (1 thread/node) | $6.986\text{ ms}$ | $1.00\times$ | $0.913\text{ ms}$ | $7.899\text{ ms}$ | $1.00\times$ | — | PASSED ($0.00\text{e}+00$) |
| | Phase 6.1 Coalesced (1 warp/node) | $0.220\text{ ms}$ | $31.79\times$ | $0.871\text{ ms}$ | $1.091\text{ ms}$ | $7.24\times$ | — | PASSED ($0.00\text{e}+00$) |
| | Phase 6.2 Streams (Pinned DMA) | $0.220\text{ ms}$ | $31.73\times$ | $0.263\text{ ms}$ | $0.483\text{ ms}$ | $16.34\times$ | $1.00\times$ | PASSED ($0.00\text{e}+00$) |
| | **Phase 6.3 Degree-Aware Hybrid** | **$0.065\text{ ms}$** | **$107.71\times$** | **$0.269\text{ ms}$** | **$0.396\text{ ms}$** | **$19.92\times$** | **$1.22\times$** | **PASSED ($1.19\text{e}-07$)** |
| **MAX** | Phase 5 Baseline (1 thread/node) | $7.214\text{ ms}$ | $1.00\times$ | $0.926\text{ ms}$ | $8.140\text{ ms}$ | $1.00\times$ | — | PASSED ($0.00\text{e}+00$) |
| | Phase 6.1 Coalesced (1 warp/node) | $0.233\text{ ms}$ | $30.91\times$ | $0.900\text{ ms}$ | $1.133\text{ ms}$ | $7.18\times$ | — | PASSED ($0.00\text{e}+00$) |
| | Phase 6.2 Streams (Pinned DMA) | $0.234\text{ ms}$ | $30.78\times$ | $0.271\text{ ms}$ | $0.505\text{ ms}$ | $16.11\times$ | $1.00\times$ | PASSED ($0.00\text{e}+00$) |
| | **Phase 6.3 Degree-Aware Hybrid** | **$0.111\text{ ms}$** | **$65.21\times$** | **$0.270\text{ ms}$** | **$0.443\text{ ms}$** | **$18.37\times$** | **$1.14\times$** | **PASSED ($0.00\text{e}+00$)** |
| **SUM** | Phase 5 Baseline (1 thread/node) | $6.518\text{ ms}$ | $1.00\times$ | $0.971\text{ ms}$ | $7.490\text{ ms}$ | $1.00\times$ | — | PASSED ($0.00\text{e}+00$) |
| | Phase 6.1 Coalesced (1 warp/node) | $0.212\text{ ms}$ | $30.73\times$ | $0.885\text{ ms}$ | $1.097\text{ ms}$ | $6.83\times$ | — | PASSED ($0.00\text{e}+00$) |
| | Phase 6.2 Streams (Pinned DMA) | $0.212\text{ ms}$ | $30.79\times$ | $0.286\text{ ms}$ | $0.497\text{ ms}$ | $15.05\times$ | $1.00\times$ | PASSED ($0.00\text{e}+00$) |
| | **Phase 6.3 Degree-Aware Hybrid** | **$0.464\text{ ms}$** | **$14.04\times$** | **$0.274\text{ ms}$** | **$0.802\text{ ms}$** | **$9.33\times$** | $0.62\times$ | **PASSED ($0.00\text{e}+00$)** |

*(Note: Degree-Aware End-to-End latency explicitly includes host degree scanning and bucketing partition overhead: $t_{\text{bucketing}} = 0.062\text{ ms}$ to $0.065\text{ ms}$).*

---

### 9.2 Comparative Matrix: Quasi-Uniform Canonical Tier A Graph (`synthetic_tier_a_10k`)
$N = 10,000$ nodes, $M = 94,945$ edges, $D = 32$ features, Max Degree = $10$ ($94.95\%$ of nodes have $d=10$, $5.01\%$ have $d=0$):

| Operator | Engine Architecture | Kernel Latency ($t_{\text{kernel}}$) | Kernel Speedup vs Base | Transfer Latency (H2D + D2H) | End-to-End Latency ($t_{\text{e2e}}$) | E2E Speedup vs Base | Verification ($\text{atol} \le 10^{-5}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MEAN** | Phase 5 Baseline | $0.068\text{ ms}$ | $1.00\times$ | $0.916\text{ ms}$ | $0.984\text{ ms}$ | $1.00\times$ | PASSED ($0.00\text{e}+00$) |
| | Phase 6.1 Coalesced | $0.017\text{ ms}$ | $4.00\times$ | $0.966\text{ ms}$ | $0.983\text{ ms}$ | $1.00\times$ | PASSED ($0.00\text{e}+00$) |
| | Phase 6.2 Streams | $0.017\text{ ms}$ | $3.95\times$ | $0.264\text{ ms}$ | $0.281\text{ ms}$ | $3.50\times$ | PASSED ($0.00\text{e}+00$) |
| | Phase 6.3 Degree-Aware | $0.020\text{ ms}$ | $3.38\times$ | $0.270\text{ ms}$ | $0.343\text{ ms}$ | $2.87\times$ | PASSED ($0.00\text{e}+00$) |
| **SUM** | Phase 5 Baseline | $0.069\text{ ms}$ | $1.00\times$ | $0.923\text{ ms}$ | $0.991\text{ ms}$ | $1.00\times$ | PASSED ($0.00\text{e}+00$) |
| | Phase 6.1 Coalesced | $0.017\text{ ms}$ | $3.99\times$ | $0.865\text{ ms}$ | $0.883\text{ ms}$ | $1.12\times$ | PASSED ($0.00\text{e}+00$) |
| | Phase 6.2 Streams | $0.017\text{ ms}$ | $4.11\times$ | $0.266\text{ ms}$ | $0.283\text{ ms}$ | $3.51\times$ | PASSED ($0.00\text{e}+00$) |
| | Phase 6.3 Degree-Aware | $0.020\text{ ms}$ | $3.35\times$ | $0.267\text{ ms}$ | $0.396\text{ ms}$ | $2.50\times$ | PASSED ($0.00\text{e}+00$) |
| **MAX** | Phase 5 Baseline | $0.068\text{ ms}$ | $1.00\times$ | $0.896\text{ ms}$ | $0.963\text{ ms}$ | $1.00\times$ | PASSED ($0.00\text{e}+00$) |
| | Phase 6.1 Coalesced | $0.017\text{ ms}$ | $4.01\times$ | $0.995\text{ ms}$ | $1.012\text{ ms}$ | $0.95\times$ | PASSED ($0.00\text{e}+00$) |
| | Phase 6.2 Streams | $0.016\text{ ms}$ | $4.14\times$ | $0.262\text{ ms}$ | $0.278\text{ ms}$ | $3.46\times$ | PASSED ($0.00\text{e}+00$) |
| | Phase 6.3 Degree-Aware | $0.022\text{ ms}$ | $3.14\times$ | $0.272\text{ ms}$ | $0.383\text{ ms}$ | $2.52\times$ | PASSED ($0.00\text{e}+00$) |

---

### 9.3 Key Scientific & Engineering Insights

1. **Resolution of Inter-Warp SIMT Degree Divergence for MEAN and MAX:**
   - In the monolithic warp-coalesced kernel (Phase 6.1/6.2), standard nodes ($d \le 32$) and dense hubs were co-located in the same 256-thread blocks. When processing the 6,340-edge super-hub, Warps $1 \dots 7$ finished in $\le 10$ iterations and sat completely idle for 6,330 iterations because block retirement is bounded by $\max_{w} t_w$.
   - Degree-aware partitioning eliminated this divergence entirely: `standard_degree_warp_kernel` retired all 9,444 standard nodes in $\le 32$ iterations without waiting for hubs.
   - For `MEAN`, this reduced pure kernel execution time from **$0.220\text{ ms} \rightarrow 0.065\text{ ms}$ ($3.39\times$ kernel speedup vs Streams, $107.71\times$ vs Baseline)**.
   - For `MAX`, kernel time fell from **$0.234\text{ ms} \rightarrow 0.111\text{ ms}$ ($2.12\times$ kernel speedup vs Streams, $65.21\times$ vs Baseline)**.
   - End-to-end wall-clock latency (inclusive of host bucketing overhead) improved by **$1.22\times$ on MEAN** ($0.396\text{ ms}$ vs $0.483\text{ ms}$) and **$1.14\times$ on MAX** ($0.443\text{ ms}$ vs $0.505\text{ ms}$).

2. **Topological Selectivity & Overhead Amortization:**
   - On the canonical Tier A graph, degrees are structurally quasi-uniform (no hubs with $d > 32$). The Degree-Aware kernel has no dense hubs to accelerate.
   - Host degree scanning requires $0.05\text{ ms} - 0.11\text{ ms}$ overhead, which slightly penalizes end-to-end latency ($0.34\text{ ms} - 0.40\text{ ms}$ vs $0.28\text{ ms}$ Streams).
   - This validates our research hypothesis ($H_{\text{6.3}}$): **Degree-aware hybrid dispatch is an adaptive, topology-dependent optimization**. It should be deployed when graph skew is detected ($\max d \gg \text{median } d$), whereas the monolithic streams runner remains optimal for low-degree uniform graphs.

---

### 9.4 Comprehensive Disclosure of Empirical Regressions & Negative Findings

To maintain absolute scientific integrity, we explicitly document the two performance regressions observed under Degree-Aware Hybrid Dispatch:

1. **Regression on Operator `SUM` on Skewed Graph (`synthetic_skewed_10k`):**
   - **Measured Latency:** Pure kernel latency regressed from **$0.2117\text{ ms}$ (Streams) to $0.4641\text{ ms}$ (Degree-Aware)**, representing a **$0.46\times$ speedup / $2.19\times$ slowdown**. End-to-end latency regressed from **$0.4975\text{ ms}$ to $0.8025\text{ ms}$** (**$0.62\times$ speedup / $1.61\times$ slowdown**).
   - **Observed Findings & Investigated Architectural Explanation (The Non-Associativity vs SM Occupancy Trade-off):**
     - In initial Colab Run `141134`, 8-warp interleaved cooperative reduction across the 6,340-edge super-hub ran in $0.0560\text{ ms}$ ($3.78\times$ faster than Streams). However, an observed floating-point discrepancy occurred: because IEEE 754 float32 addition is non-associative: $(a + b) + c \neq a + (b + c)$, summing 6,340 positive single-precision features out of sequential order accumulated a rounding drift of $3.97 \times 10^{-4}$ (relative error $4.56 \times 10^{-6} \le 1\text{ ULP}$ per addition), which was recorded as exceeding our project tolerance gate ($\text{atol} \le 1.0 \times 10^{-5}$, `FAILED_TOLERANCE`).
     - In `MEAN`, this rounding variance was scaled down by the node degree ($6,340$), reducing the final error to $1.19 \times 10^{-7}$, which passed verification. In `MAX`, comparison has zero precision loss ($0.00\text{e}+00$ error).
     - But in `SUM`, to ensure strict $0.00\text{e}+00$ parity with the single-threaded CPU reference without altering project tolerances, Warp 0 was assigned serial accumulation.
     - Our investigation of this subsequent latency regression indicates that because `hub_degree_block_kernel` allocates 256 threads (8 warps) per block, enforcing `if (warp_id == 0)` left 7 out of 8 warps (224 threads out of 256) inactive during the loop. With only 1 active warp per block across the 56 hub blocks, SM warp occupancy was substantially reduced, limiting the hardware warp schedulers' ability to hide global memory read latency during the 6,340 sequential iterations. In addition, the hub kernel executed sequentially after the standard-degree kernel on the same stream, together explaining the observed $0.464\text{ ms}$ kernel measurement.

2. **Regression on All Tier A End-to-End Cases (Uniform Degree Topology):**
   - **`MEAN` End-to-End:** $0.3433\text{ ms}$ (Degree-Aware) vs $0.2810\text{ ms}$ (Streams) $\rightarrow$ **$0.82\times$ speedup ($22.2\%$ slowdown)**.
   - **`SUM` End-to-End:** $0.3961\text{ ms}$ (Degree-Aware) vs $0.2827\text{ ms}$ (Streams) $\rightarrow$ **$0.71\times$ speedup ($40.1\%$ slowdown)**.
   - **`MAX` End-to-End:** $0.3832\text{ ms}$ (Degree-Aware) vs $0.2785\text{ ms}$ (Streams) $\rightarrow$ **$0.73\times$ speedup ($37.4\%$ slowdown)**.
   - **Root Cause (Host Bucketing Non-Amortization):**
     - On the Tier A graph ($N=10,000$, $M=94,945$), $94.95\%$ of nodes have degree 10, $5.01\%$ have degree 0, and the maximum degree is only 10. There are zero hub nodes ($d > 32$).
     - The host degree scanning and bucketing partition routine consumes $t_{\text{bucketing}} = 0.0533 - 0.1087\text{ ms}$ on the CPU.
     - In addition, dispatching standard nodes via index indirection (`d_std_nodes[global_warp_id]`) adds small GPU addressing overhead, increasing kernel latency slightly ($0.020 - 0.022\text{ ms}$ vs $0.016 - 0.017\text{ ms}$ in Streams).
     - Because there are zero high-degree hubs, there is zero warp execution divergence to recover. Consequently, the CPU bucketing time and launch indirection constitute pure, non-recoverable overhead, causing all Tier A end-to-end latencies to regress by $22\% - 40\%$ compared to the monolithic Streams runner.

---

## 10. Architectural Decision & Retention Policy (Stage 6)

- **Retention Decision:** **RETAINED AS SPECIALIZED ADAPTIVE DISPATCH ENGINE**.
- **Role in FGI Architecture:**
  1. **Canonical Engine:** Experiment 6.2 (Pinned Memory + Streams) is retained as the default aggregation engine for regular, low-degree graphs ($d_{\max} \le 32$).
  2. **Heavy-Tailed / Skewed Engine:** Experiment 6.3 (Degree-Aware Hybrid Dispatch) is retained as the high-throughput engine for power-law financial graphs exhibiting degree skew ($\max d > 32$), achieving up to $107.7\times$ speedup over Baseline and $3.39\times$ over Monolithic Streams on key operators (`MEAN`, `MAX`).
- **Phase 6 Optimization Roadmap:** Experiments 6.1 (Warp Coalescing), 6.2 (Pinned Streams), and 6.3 (Degree-Aware Dispatch) are now **100% complete, empirically benchmarked on Tesla T4, and closed**.
