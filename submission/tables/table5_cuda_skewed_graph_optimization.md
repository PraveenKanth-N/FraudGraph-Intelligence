# Table 5: CUDA Performance on Power-Law Skewed Graph & Degree-Aware Trade-offs

**Source Artifacts:** `experiments/results/exp_gpu_*_skewed_*_20261024_142037.json`  
**Target Accelerator:** NVIDIA Tesla T4 GPU (SM 7.5, 40 SMs, 14.9 GB VRAM, Turing Architecture)  
**Workload Configuration:** Power-Law Skewed In-Degree Graph ($N = 10,000$ Nodes, $M = 94,970$ Directed Edges, $D = 32$, Maximum Hub Degree $d_{\max} = 6,340$ Super-Hub, Fixture: `synthetic_skewed_10k.npz`)  
**Timing Instrumentation:** CUDA Events (`cudaEventRecord`, `cudaEventElapsedTime`), 20 recorded repetitions with 5 warmup passes  

| Implementation Strategy | Operator | Kernel Latency (ms) | End-to-End Latency (ms) | Speedup vs Baseline Kernel | Speedup vs Coalesced Kernel | Effective Bandwidth (GB/s) | Processed Edges / Second | Micro-Architectural Behavior & Engineering Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CUDA Baseline (Thread/Node)** | SUM | 6.518180 ms | 7.489540 ms | 1.00x | — | 0.457 GB/s | $1.457 \times 10^7$ | Severe SIMT warp divergence: threads waiting on hub neighbor loop. |
| **CUDA Baseline (Thread/Node)** | MEAN | 6.985980 ms | 7.898550 ms | 1.00x | — | 0.427 GB/s | $1.359 \times 10^7$ | Extreme serialization across imbalanced neighbor counts. |
| **CUDA Baseline (Thread/Node)** | MAX | 7.214300 ms | 8.140290 ms | 1.00x | — | 0.413 GB/s | $1.316 \times 10^7$ | Uncoalesced scalar reads saturate memory controller queues. |
| **CUDA Coalesced (1 Warp/Node)** | SUM | 0.212122 ms | 1.096890 ms | **30.73x** | 1.00x | 14.05 GB/s | $4.477 \times 10^8$ | Warp cooperative vector loads satisfy memory in aligned transactions. |
| **CUDA Coalesced (1 Warp/Node)** | MEAN | 0.219774 ms | 1.091170 ms | **31.79x** | 1.00x | 13.56 GB/s | $4.321 \times 10^8$ | Elimination of non-coalesced DRAM accesses yields $>31\times$ gain. |
| **CUDA Coalesced (1 Warp/Node)** | MAX | 0.233390 ms | 1.133040 ms | **30.91x** | 1.00x | 12.77 GB/s | $4.069 \times 10^8$ | High efficiency, but warps on hub nodes still stall peer warps in block. |
| **CUDA Degree-Aware Hybrid** | **MEAN** | **0.064858 ms** | **0.396447 ms** | **107.71x** | **3.39x Faster** | **45.95 GB/s** | **$1.464 \times 10^9$** | **Optimal: Partitioning into low ($<32$), mid ($32-128$), and high ($>128$) nodes maximizes warp occupancy.** |
| **CUDA Degree-Aware Hybrid** | **MAX** | **0.110626 ms** | **0.443224 ms** | **65.21x** | **2.11x Faster** | **26.94 GB/s** | **$8.585 \times 10^8$** | **Block-level tree reduction for high-degree hub nodes eliminates long sequential inner loops.** |
| **CUDA Degree-Aware Hybrid** | **SUM** | **0.464106 ms** | **0.802457 ms** | **14.04x** | **0.46x (2.19x REGRESSION)** | **6.42 GB/s** | **$2.046 \times 10^8$** | **CRITICAL REGRESSION: Single-warp (Warp 0) serial accumulation enforced to guarantee float32 non-associative tolerance ($\text{atol} \le 10^{-5}$), leaving 7 of 8 warps idle in hub blocks.** |

> [!WARNING]
> **Engineering Analysis of the Degree-Aware SUM Regression:**
> In degree-aware dispatch (`fgi_degree_aware_kernels.cu`), nodes with in-degree $>32$ are processed by a dedicated cooperative block kernel (`hub_degree_block_kernel`, allocating 256 threads = 8 warps per hub node).
> - For **MEAN** and **MAX**, all 8 warps cooperatively interleave edge traversal in parallel (`p += 8`), stage partial results into shared memory (`s_warp_acc[8][32]`), and perform a clean 8-way reduction on Warp 0. This yields **$3.39\times$** and **$2.11\times$ speedups** over the monolithic coalesced kernel.
> - For **SUM**, however, cooperative reordering across 6,340 edges accumulates single-precision floating-point rounding drift due to the non-associativity of IEEE 754 float32 addition: $(a + b) + c \neq a + (b + c)$. In initial Colab runs, this drift ($3.97 \times 10^{-4}$) failed the project's strict numerical verification gate ($\text{atol} \le 10^{-5}$). To ensure strict mathematical parity with the CPU reference, the implementation enforces single-warp serial accumulation on Warp 0 (`if (warp_id == 0)`).
> - **Observed Impact:** Enforcing `if (warp_id == 0)` causes 7 out of 8 warps (224 threads out of 256) to sit completely idle while Warp 0 loops through all 6,340 edges. This severely reduces Streaming Multiprocessor (SM) warp occupancy and limits the hardware warp scheduler's ability to hide global memory read latency, presenting a highly plausible code-observed mechanism for the measured $2.19\times$ kernel slowdown ($0.212 \to 0.464$ ms).
> - **Scientific Classification:** The single-warp serial loop and 7-warp idle state are directly verified in the source code; attributing 100% of the $2.19\times$ slowdown to this occupancy loss is an engineering hypothesis supported by micro-architectural principles. Zero atomic operations were used in this kernel path.
