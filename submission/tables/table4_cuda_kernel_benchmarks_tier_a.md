# Table 4: CUDA Kernel Benchmark Performance on Tier A Uniform Graph

**Source Artifacts:** `experiments/results/exp_gpu_*_tier_a_*_20261024_142037.json`  
**Target Accelerator:** NVIDIA Tesla T4 GPU (SM 7.5, 40 SMs, 14.9 GB VRAM, Turing Architecture)  
**Workload Configuration:** Uniform In-Degree Graph ($N = 10,000$ Nodes, $M = 94,945$ Directed Edges, $D = 32$ Feature Dimension)  
**Timing Instrumentation:** CUDA Events (`cudaEventRecord`, `cudaEventElapsedTime`), 20 recorded repetitions with 5 warmup passes  

| Implementation Strategy | Operator | Kernel Latency (Mean $\pm$ Std) | H2D PCIe Transfer (ms) | D2H PCIe Transfer (ms) | End-to-End Latency (ms) | Kernel Speedup vs Baseline | E2E Speedup vs Baseline | Effective Bandwidth (GB/s) | Processed Edges / Second |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CUDA Baseline (Thread/Node)** | SUM | $0.068595 \pm 0.0020$ ms | 0.600544 ms | 0.322272 ms | 0.991411 ms | 1.00x | 1.00x | 43.44 GB/s | $1.384 \times 10^9$ |
| **CUDA Baseline (Thread/Node)** | MEAN | $0.067947 \pm 0.0019$ ms | 0.580416 ms | 0.336032 ms | 0.984395 ms | 1.00x | 1.00x | 43.85 GB/s | $1.397 \times 10^9$ |
| **CUDA Baseline (Thread/Node)** | MAX | $0.067662 \pm 0.0018$ ms | 0.561408 ms | 0.334368 ms | 0.963438 ms | 1.00x | 1.00x | 44.04 GB/s | $1.403 \times 10^9$ |
| **CUDA Coalesced (1 Warp/Node)** | SUM | **$0.017192 \pm 0.0006$ ms** | 0.548672 ms | 0.316704 ms | 0.882568 ms | **3.99x** | 1.12x | **173.32 GB/s** | **$5.523 \times 10^9$** |
| **CUDA Coalesced (1 Warp/Node)** | MEAN | **$0.016995 \pm 0.0005$ ms** | 0.573792 ms | 0.391744 ms | 0.982531 ms | **4.00x** | 1.00x | **175.33 GB/s** | **$5.587 \times 10^9$** |
| **CUDA Coalesced (1 Warp/Node)** | MAX | **$0.016878 \pm 0.0006$ ms** | 0.607936 ms | 0.387296 ms | 1.012110 ms | **4.01x** | 0.95x | **176.54 GB/s** | **$5.625 \times 10^9$** |
| **CUDA Pinned Streams** | SUM | $0.016698 \pm 0.0005$ ms | 0.159072 ms | 0.106880 ms | **0.282650 ms** | **4.11x** | **3.51x** | **178.46 GB/s** | **$5.686 \times 10^9$** |
| **CUDA Pinned Streams** | MEAN | $0.017197 \pm 0.0007$ ms | 0.157248 ms | 0.106560 ms | **0.281005 ms** | **3.95x** | **3.50x** | **173.28 GB/s** | **$5.521 \times 10^9$** |
| **CUDA Pinned Streams** | MAX | **$0.016360 \pm 0.0005$ ms** | 0.156768 ms | 0.105344 ms | **0.278472 ms** | **4.14x** | **3.46x** | **182.14 GB/s** | **$5.803 \times 10^9$** |
| **CUDA Degree-Aware Hybrid** | SUM | $0.020469 \pm 0.0008$ ms | 0.161024 ms | 0.105856 ms | 0.396069 ms | 3.35x | 2.50x | 145.58 GB/s | $4.639 \times 10^9$ |
| **CUDA Degree-Aware Hybrid** | MEAN | $0.020126 \pm 0.0007$ ms | 0.163776 ms | 0.106144 ms | 0.343334 ms | 3.38x | 2.87x | 148.05 GB/s | $4.717 \times 10^9$ |
| **CUDA Degree-Aware Hybrid** | MAX | $0.021536 \pm 0.0008$ ms | 0.164448 ms | 0.107744 ms | 0.382614 ms | 3.14x | 2.52x | 138.36 GB/s | $4.409 \times 10^9$ |

> [!IMPORTANT]
> **Key Architectural Takeaways:**
> 1. Mapping 1 warp (32 threads) per node to cooperatively gather feature vectors converts strided DRAM transactions into coalesced 128-byte bus requests, increasing effective memory bandwidth from 43.4 GB/s to **176.5 GB/s** (**4.0x kernel acceleration**).
> 2. Kernel execution latency (~0.017 ms) is negligible compared to PCIe bus transfers (~0.91 ms total for pageable memory).
> 3. Asynchronous multi-stream pipelining with host pinned memory overlaps transfer chunks with kernel execution, reducing end-to-end latency from 0.99 ms to **0.28 ms** (**3.5x end-to-end acceleration**).
