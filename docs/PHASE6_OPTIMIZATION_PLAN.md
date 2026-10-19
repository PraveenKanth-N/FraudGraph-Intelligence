# Phase 6 Optimization Plan: Memory-Aware CUDA Graph Aggregation
## FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Workspace:** `D:\FraudGraph Intelligence`  
**Git Baseline Commit:** `dcebf05` (Phase 5 GPU Validation Passed)  
**Target Hardware:** NVIDIA Tesla T4 (Turing `sm_75`, 40 SMs, 14.9 GB VRAM, 320 GB/s Peak Bandwidth)  
**Compiler Baseline:** CUDA compilation tools release 13.0, V13.0.88 (`-O3 -std=c++17`)  
**Status:** PROPOSED PLANNING SPECIFICATION (Pre-Implementation)

---

## 1. Executive Summary & Purpose

In Phase 5, FraudGraph Intelligence established its first verified GPU baseline:
- An auditable, unoptimized **one-thread-per-node** CUDA C++ aggregation kernel ([`fgi_baseline_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu)).
- Execution on physical NVIDIA Tesla T4 hardware achieved **$1.39\text{ billion edges/s}$** at **$0.0681\text{ ms}$** pure kernel latency on the canonical Tier A financial graph ($N=10,000, M=94,945, D=32$).
- Numerical correctness achieved exact equivalence ($\text{atol} = 0.0 \le 1.0 \times 10^{-5}$) against the Phase 4 CPU reference across Sum, Mean, and Max operators.

The objective of **Phase 6: CUDA Optimization Laboratory** is to design, implement, benchmark, and profile specialized memory-aware optimizations to address the fundamental GPU hardware bottlenecks identified during Phase 5.

> [!IMPORTANT]
> **Research & Scientific Integrity Rules:**
> 1. All candidate optimizations are treated as **hypotheses** until empirically measured on real GPU hardware. No speedups are promised in advance.
> 2. The Phase 5 baseline kernel (`aggregate_baseline_kernel`) remains **strictly preserved and frozen** as the permanent regression baseline.
> 3. Numerical correctness ($\text{atol} \le 1.0 \times 10^{-5}$, zero NaNs/Infs, identical empty-neighborhood zero-sentinels) is a mandatory hard gate before any latency measurement is recorded.
> 4. Kernel-only execution times ($t_{\text{kernel}}$) and end-to-end pipeline latencies ($t_{\text{e2e}}$) must remain decoupled via hardware CUDA events.
> 5. If GPU hardware is unavailable, experiments must be labeled `NOT_RUN`; data must never be fabricated or simulated.

---

## 2. Phase 5 Baseline Telemetry Reference

To ensure strict experimental continuity, all Phase 6 candidate optimizations will be evaluated against the empirical Phase 5 baseline measurements recorded on NVIDIA Tesla T4:

| Metric | Baseline Sum | Baseline Mean | Baseline Max | Supporting Record |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Kernel Latency ($t_{\text{kernel}}$)** | $0.0681 \pm 0.0007\text{ ms}$ | $0.0682 \pm 0.0007\text{ ms}$ | $0.0673 \pm 0.0022\text{ ms}$ | [`exp_gpu_baseline_tier_a_*.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/) |
| **Host-to-Device Transfer ($t_{\text{H2D}}$)** | $0.5387\text{ ms}$ | $0.5579\text{ ms}$ | $0.5839\text{ ms}$ | Decoupled PCIe Gen3 transfer |
| **Device-to-Host Transfer ($t_{\text{D2H}}$)** | $0.3247\text{ ms}$ | $0.3292\text{ ms}$ | $0.3540\text{ ms}$ | Output tensor copy |
| **End-to-End Latency ($t_{\text{e2e}}$)** | $0.9314\text{ ms}$ | $0.9553\text{ ms}$ | $1.0052\text{ ms}$ | PCIe transfer dominates ($92.9\%$) |
| **Measured Throughput** | $1.394\times 10^9\text{ edges/s}$ | $1.393\times 10^9\text{ edges/s}$ | $1.411\times 10^9\text{ edges/s}$ | Analytical edge rate |
| **Effective Bandwidth** | $203.44\text{ GB/s}$ ($63.6\%$) | $203.24\text{ GB/s}$ ($63.5\%$) | $205.84\text{ GB/s}$ ($64.3\%$) | Analytical data movement model |
| **Numerical Error vs CPU Ref** | $0.0$ (`PASSED`) | $0.0$ (`PASSED`) | $0.0$ (`PASSED`) | Tolerance $\le 1.0\times 10^{-5}$ |

---

## 3. Candidate Optimization Investigation

```
+--------------------------------------------------------------------------------------------------+
|                            PHASE 6 CANDIDATE OPTIMIZATION SUITE                                  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  [Global Memory] ---- Coalescing ---> Contiguous 128-byte DRAM Transactions (Candidate 1)        |
|                                                                                                  |
|  [SIMT Warps]    ---- Cooperative --> Warp-per-Node / Sub-Warp Mapping (Candidate 2)             |
|                                                                                                  |
|  [Registers]     ---- Intrinsic ----> Single-Cycle Warp Shuffles __shfl_down_sync (Candidate 3)  |
|                                                                                                  |
|  [Graph Degree]  ---- Partition ----> Degree-Aware Node Bucketing (Candidate 4)                  |
|                                                                                                  |
|  [On-Chip SRAM]  ---- Caching ------> Shared Memory Tiles for Dense Hubs (Candidate 5)           |
|                                                                                                  |
|  [Launch Config] ---- Tuning -------> Block Size & Tile Dimension Sweep (Candidate 6)            |
|                                                                                                  |
|  [Host System]   ---- Pinned DMA ---> Pinned Memory & CUDA Streams (Candidate 7)                 |
+--------------------------------------------------------------------------------------------------+
```

---

### Candidate 1: Memory Access Coalescing via Feature-Dimension Thread Mapping

- **Suspected Bottleneck & Supporting Evidence:**
  In the Phase 5 baseline kernel, thread $i$ handles output node $i$. The node feature tensor $X$ is stored in row-major layout with stride $D$ (where $X[j, k]$ is at offset $j \times D + k$).
  When 32 consecutive threads in a warp (threads $i, i+1, \dots, i+31$) simultaneously process their respective first neighbors, thread $i$ fetches neighbor $j_i$ while thread $i+1$ fetches neighbor $j_{i+1}$. Because financial graphs exhibit arbitrary edge wiring, $j_i$ and $j_{i+1}$ are non-contiguous in memory.
  Consequently, adjacent threads in a warp request data from completely disparate 128-byte cache lines. Each 128-byte DRAM request transfers 32 floats, but only 1 float is consumed by the requesting thread—resulting in significant memory bus underutilization.
- **Proposed Kernel-Level Change:**
  Reorder the parallel dimension: assign consecutive threads within a thread cooperative (e.g. a 32-thread warp) to consecutive columns $k \in [0, D-1]$ of the *same* neighbor node $j$.
  When the warp loads $X[j, 0 \dots 31]$, the 32 threads request addresses $j \cdot D + 0, j \cdot D + 1, \dots, j \cdot D + 31$. Because these 32 single-precision floats (128 bytes) are contiguous and 128-byte aligned, the hardware memory controller coalesces the entire warp read into a **single 128-byte DRAM transaction**.
- **Expected Benefits & Possible Regressions:**
  - *Expected Benefit:* Near-optimal memory bus efficiency, higher L1/L2 cache hit rate, and elimination of DRAM transaction amplification.
  - *Possible Regression:* For graphs with small feature dimensions ($D < 32$), full 32-thread warps would experience underutilization unless sub-warp tiles ($T=4, 8, 16$) are used.
- **Correctness Tests Required:**
  Verification across feature dimensions $D \in \{1, 4, 16, 32, 128\}$ using synthetic fixtures. Maximum absolute error must satisfy $\text{atol} \le 1.0 \times 10^{-5}$.
- **Benchmark Methodology & Success Criteria:**
  Benchmark on `synthetic_tier_a_10k` ($D=32$); compare pure kernel execution time $t_{\text{kernel}}$ against Phase 5 baseline ($0.0681\text{ ms}$). Success criterion: $t_{\text{kernel, opt}} < t_{\text{kernel, base}}$ with exact numerical equivalence.
- **Estimated Complexity:** Medium.

---

### Candidate 2: Thread-to-Node Mapping (Warp-per-Node Architecture)

- **Suspected Bottleneck & Supporting Evidence:**
  Power-law financial graphs exhibit severe degree skew. In `synthetic_tier_a_10k`, node degrees range from $d_i = 0$ (isolated) up to $d_i = 433$ (hub account).
  In the baseline 1-thread-per-node model, if a warp contains 31 nodes with degree 1 and 1 node with degree 433, the 31 threads complete in 1 iteration and remain stalled in an idle state for 432 loop iterations while the single hub thread finishes. This causes severe SIMT warp divergence.
- **Proposed Kernel-Level Change:**
  Implement a **Warp-per-Node (Vector) Aggregator Kernel** (inspired by Bell & Garland 2009 SpMV vector kernels).
  Each 32-thread warp is assigned to process a single output node $i$:
  $$\text{warp\_id} = \frac{\text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}}{32}$$
  All 32 threads in the warp cooperatively iterate over node $i$'s neighbor list $[\text{start}, \text{end})$ in parallel chunks. Because all 32 threads share the identical loop termination condition ($\lceil d_i / 32 \rceil$), **intra-warp degree divergence is reduced to zero**.
- **Expected Benefits & Possible Regressions:**
  - *Expected Benefit:* Eliminates warp divergence caused by degree skew; distributes hub node processing across 32 threads.
  - *Possible Regression:* For low-degree graphs where average degree is low ($\bar{d} < 4$), dedicating an entire 32-thread warp per node may underutilize execution lanes. Sub-warp tile partitioning ($T=8$ or $T=16$) must be investigated.
- **Correctness Tests Required:**
  Star-graph fixture (`synthetic_star_1k`) with 1 hub node connected to 1,000 leaf nodes; isolated node handling; empty graph.
- **Benchmark Methodology & Success Criteria:**
  Compare kernel execution time on power-law graphs ($\gamma = 2.5$) vs uniform random graphs. Success criterion: lower execution time and reduced latency variance ($\sigma_{\text{kernel}}$) on skewed degree distributions.
- **Estimated Complexity:** Medium-High.

---

### Candidate 3: In-Register Reduction via Warp Shuffle Intrinsics (`__shfl_down_sync`)

- **Suspected Bottleneck & Supporting Evidence:**
  When multiple threads within a cooperative group reduce values across a neighborhood, inter-thread communication traditionally requires writing to shared memory (`__shared__`), synchronizing with `__syncthreads()`, and reading back. Shared memory round-trips incur latency (20–30 clock cycles) and require block-wide execution barriers.
- **Proposed Kernel-Level Change:**
  Utilize NVIDIA Turing/Ampere warp shuffle intrinsics (`__shfl_down_sync(0xffffffff, val, offset)`).
  Threads in a warp perform tree reduction across register banks in $\log_2(32) = 5$ instructions without writing to memory or invoking `__syncthreads()`:
  ```cpp
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
  }
  ```
  For Max aggregation, `fmaxf()` replaces addition.
- **Expected Benefits & Possible Regressions:**
  - *Expected Benefit:* Register-speed reduction (1 clock cycle per step); zero shared memory footprint; zero bank conflicts.
  - *Possible Regression:* Operates strictly within a 32-thread warp. Reductions spanning multiple warps still require a secondary block-level staging step.
- **Correctness Tests Required:**
  Verify tree reductions across Sum, Mean, and Max for both power-of-two and odd neighborhood sizes.
- **Benchmark Methodology & Success Criteria:**
  Compare kernel latency with and without shuffle intrinsics. Success criterion: measurable reduction in reduction overhead during cooperative aggregation.
- **Estimated Complexity:** Low-Medium.

---

### Candidate 4: Degree-Aware Node Bucketing & Hybrid Dispatch

- **Suspected Bottleneck & Supporting Evidence:**
  Financial transaction graphs are structurally heterogeneous:
  - $\sim 5\text{–}10\%$ of accounts are isolated ($d_i = 0$).
  - $\sim 70\text{–}80\%$ of accounts are sparse/consumer nodes ($1 \le d_i \le 8$).
  - $\sim 5\text{–}10\%$ of accounts are dense merchants/hubs ($d_i > 32$).
  A single execution mapping (either 1-thread-per-node or 1-warp-per-node) cannot be optimal for all three classes simultaneously.
- **Proposed Kernel-Level Change:**
  Implement a **Hybrid Adaptive Aggregator**:
  During graph loading/staging, partition node IDs into two or three execution streams:
  1. *Zero-Degree Stream:* Direct bulk zero assignment (`cudaMemsetAsync` or trivial memset kernel).
  2. *Low-Degree Stream ($d_i \le 16$):* 1-thread-per-node or sub-warp tile ($T=4$).
  3. *Hub Stream ($d_i > 16$):* Full warp-per-node or block-per-node with shared-memory tiling.
- **Expected Benefits & Possible Regressions:**
  - *Expected Benefit:* Optimal occupancy and zero divergence across every topological regime.
  - *Possible Regression:* Partitioning and sorting overhead on host or GPU; multiple kernel launch overheads.
- **Correctness Tests Required:**
  Ensure concatenation of partitioned outputs exactly preserves original 0-indexed node order.
- **Benchmark Methodology & Success Criteria:**
  Measure combined time (partitioning + multi-kernel launch) vs monolithic kernel. Success criterion: net latency reduction on heavy-tailed graphs.
- **Estimated Complexity:** High.

---

### Candidate 5: Shared-Memory Caching (`__shared__`) for High-Frequency Hub Neighbors

- **Suspected Bottleneck & Supporting Evidence:**
  In financial graphs, central payment gateways and fraud money mules are referenced repeatedly as neighbors by thousands of transactions. In baseline execution, every edge pointing to a hub re-reads the hub's feature vector from global DRAM/L2 cache.
- **Proposed Kernel-Level Change:**
  Allocate on-chip shared memory tiles (`extern __shared__ float s_hub_cache[]`) inside the block. The thread block cooperatively stages the embeddings of the top-$K$ highest-degree neighbor nodes into shared memory once, then serves all incident edge lookups from on-chip SRAM.
- **Expected Benefits & Possible Regressions:**
  - *Expected Benefit:* Shared memory latency ($\approx 20\text{ cycles}$) is an order of magnitude faster than DRAM ($\approx 200\text{–}400\text{ cycles}$).
  - *Possible Regression:* Shared memory capacity is hardware-limited (64 KB per SM on Tesla T4). Allocating large shared memory buffers reduces active block occupancy per SM.
- **Correctness Tests Required:**
  Boundary checks on hub indices; synchronization barriers (`__syncthreads()`) around staging; verify numerical match with baseline.
- **Benchmark Methodology & Success Criteria:**
  Evaluate on hub-dominated test topologies (`synthetic_star_1k`). Success criterion: significant reduction in global memory read transactions.
- **Estimated Complexity:** High.

---

### Candidate 6: Launch Geometry & Thread-Block Dimension Tuning

- **Suspected Bottleneck & Supporting Evidence:**
  The Phase 5 baseline hard-coded `block_size = 256`. Depending on register pressure and active warps, 256 threads per block may or may not achieve optimal occupancy on the Tesla T4 SM (which supports up to 1,024 active threads per SM and 16 active blocks per SM).
- **Proposed Kernel-Level Change:**
  Implement C++ template launch configurations (`template<int BLOCK_SIZE, int TILE_SIZE>`) and execute a systematic grid sweep across:
  - Block sizes: $\text{BLOCK\_SIZE} \in \{64, 128, 256, 512\}$.
  - Cooperative tile sizes: $\text{TILE\_SIZE} \in \{4, 8, 16, 32\}$.
- **Expected Benefits & Possible Regressions:**
  - *Expected Benefit:* Pinpoints the empirical hardware occupancy sweet spot for the Tesla T4 without code refactoring.
  - *Possible Regression:* None (pure tuning exploration).
- **Correctness Tests Required:**
  Automated parameter sweep asserting $\text{atol} \le 1.0 \times 10^{-5}$ across all configurations.
- **Benchmark Methodology & Success Criteria:**
  Execution matrix reporting kernel latency across all valid block/tile pairs.
- **Estimated Complexity:** Low-Medium.

---

### Candidate 7: Transfer Overhead Mitigation (Pinned Memory & Asynchronous Streams)

- **Suspected Bottleneck & Supporting Evidence:**
  In Phase 5, host-to-device ($0.55\text{ ms}$) and device-to-host ($0.33\text{ ms}$) transfers consumed **92.9%** of the total end-to-end latency ($0.955\text{ ms}$). Host allocations currently use standard pageable host memory (`std::vector`), which requires the CUDA driver to perform a two-stage copy via an internal staging buffer.
- **Proposed Kernel-Level Change:**
  Allocate host buffers using page-locked (pinned) memory via `cudaHostAlloc` / `cudaMallocHost`. Execute transfers using `cudaMemcpyAsync` on non-default CUDA streams (`cudaStream_t`), enabling direct DMA memory transfers at maximum PCIe bus speed and allowing overlap of compute and data movement.
- **Expected Benefits & Possible Regressions:**
  - *Expected Benefit:* Maximizes PCIe bandwidth utilization (up to ~12–15 GB/s on PCIe Gen3 x16); reduces $t_{\text{H2D}}$ and $t_{\text{D2H}}$ latency.
  - *Possible Regression:* Pinned memory cannot be paged out by the operating system; allocation overhead is slightly higher than standard `malloc`.
- **Correctness Tests Required:**
  Ensure proper stream synchronization (`cudaStreamSynchronize`) before host reads output tensors.
- **Benchmark Methodology & Success Criteria:**
  Measure end-to-end latency $t_{\text{e2e}}$ on pinned vs pageable memory. Success criterion: measurable reduction in $t_{\text{e2e}}$.
- **Estimated Complexity:** Medium.

---

## 4. Evaluation Matrix & Research Hypotheses

| Candidate ID | Optimization Hypothesis | Primary Target Bottleneck | Expected Risk / Tradeoff | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **OPT-01** | Coalesced feature-dimension loads will reduce DRAM transaction count by up to $32\times$ for $D=32$. | Uncoalesced global DRAM reads | Ineffective if $D < 4$ | Medium |
| **OPT-02** | Warp-per-node mapping will eliminate intra-warp thread divergence on power-law graphs. | SIMT warp divergence from degree skew | Underutilization if $d_i \ll 32$ | Medium-High |
| **OPT-03** | In-register reduction via `__shfl_down_sync` will reduce intra-block reduction latency to $\le 5$ cycles. | Shared-memory latency & barrier stalls | Restricted to 32-thread warp scope | Low-Medium |
| **OPT-04** | Degree-aware bucketing will maximize hardware occupancy across isolated, sparse, and hub nodes. | Multi-regime topological divergence | Partitioning and multi-kernel launch overhead | High |
| **OPT-05** | Shared-memory caching of hub embeddings will reduce redundant global DRAM fetches. | Repeated reads of high-degree hub nodes | Shared memory capacity limits SM occupancy | High |
| **OPT-06** | Launch geometry grid sweep will identify the empirical occupancy sweet spot for Tesla T4. | Suboptimal block size / warp occupancy | Compilation time for template instantiations | Low-Medium |
| **OPT-07** | Pinned host memory and asynchronous streams will reduce PCIe transfer latency by $15\text{–}30\%$. | Host pageable staging buffer overhead | Physical host RAM consumption | Medium |

---

## 5. Proposed First Optimization Experiment (Experiment 6.1)

### 5.1 Architecture: Warp-Coalesced Aggregator Kernel
- **Target File:** `backend/gpu_engine/cuda/src/fgi_coalesced_kernels.cu`
- **Candidate Combination:** OPT-01 (Coalesced Loads) + OPT-02 (Warp-per-Node) + OPT-03 (Warp Shuffle Reduction).
- **Core Design:**
  1. Assign 1 warp (32 threads) per output node $i$.
  2. For feature dimension $D = 32$, thread $t \in [0, 31]$ is assigned to feature column $k = t$.
  3. When the warp processes neighbor $j$, thread $t$ reads $X[j, t]$. Because threads $0 \dots 31$ read contiguous addresses $j \cdot 32 + 0 \dots j \cdot 32 + 31$, **the fetch is 100% coalesced into a single 128-byte DRAM transaction**.
  4. Each thread $t$ maintains an accumulator `acc_k` across all $d_i$ neighbors in register space.
  5. If $d_i = 0$, all 32 threads write $0.0\text{f}$ directly to output.
  6. For Max aggregation, thread $t$ tracks `max_k` initialized from neighbor 0.
  7. No shared memory and no inter-thread barriers are required for $D=32$.

### 5.2 Why Experiment 6.1 First?
1. **Directly Targets the Root Bottleneck:** Phase 5 bottleneck analysis proved that uncoalesced global reads are the primary limiting factor for memory bandwidth efficiency.
2. **Mathematical Elegance for $D=32$:** Our standard benchmark graph uses $D=32$ floats (128 bytes), matching the exact warp size (32 threads) and hardware cache line size (128 bytes).
3. **Zero Divergence:** All 32 threads in the warp process the exact same neighbor sequence, completely eliminating warp divergence during edge iteration.
4. **Preserves Correctness Guarantees:** Isolated nodes and negative features are handled cleanly in registers without race conditions.

---

## 6. Implementation Protocol & Phase 6 Gate Criteria

```
+--------------------------------------------------------------------------------------------------+
|                                  PHASE 6 EXECUTION GATEWAY                                       |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
| [1. Plan Approved] -> [2. Implement fgi_coalesced_kernels.cu] -> [3. Compile & Run Tests]        |
|                              (Preserve baseline unchanged)           (atol <= 1e-5 on toy graph) |
|                                                                                                  |
| [4. Google Colab Run] -> [5. Ingest JSON Telemetry] -> [6. Verify Speedup & Document Results]    |
|   (NVIDIA Tesla T4)           (Strict schema check)        (Update PHASE_STATUS.md & CHANGELOG)  |
+--------------------------------------------------------------------------------------------------+
```

### Gate Completion Criteria for Phase 6:
1. `backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu` remains completely untouched.
2. New kernel `fgi_coalesced_kernels.cu` compiles cleanly with `-O3 -std=c++17`.
3. Passes all 18 numerical correctness requirements ($\text{atol} \le 1.0 \times 10^{-5}$) on toy and Tier A graphs.
4. Google Colab execution on NVIDIA Tesla T4 produces verified `MEASURED` JSON telemetry.
5. Speedup over Phase 5 baseline is rigorously reported with decoupled kernel and end-to-end metrics.
6. Full test regression suite passes with 0 failures.

---

*Planning document authored and verified. Awaiting user authorization before modifying CUDA sources.*
