# Table 3: CPU Neighbor Aggregation Benchmark Performance

**Primary Source Artifact:** `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv`  
**Host Processor:** AMD64 Architecture, Python 3.14.3, NumPy 2.5.4, SciPy 1.15.1  
**Measurement Method:** Python `perf_counter` host timer, 20 recorded repetitions with warmup  

| Workload Configuration | Operator | Engine Implementation | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | Coeff. of Variation (CV%) | Speedup vs Ref | Compulsory Bandwidth (GB/s) | Algorithmic Bandwidth (GB/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier A 10k ($N=10k, M=100k, D=64$)** | MEAN | Sequential Reference Loop | 114.8598 ms | 115.1114 ms | 129.4814 ms | 9.80% | 1.00x (Baseline) | 0.0522 GB/s | 0.2528 GB/s |
| **Tier A 10k ($N=10k, M=100k, D=64$)** | MEAN | **Vectorized SpMM (SciPy)** | **2.5756 ms** | **2.3322 ms** | **4.0347 ms** | 34.29% | **44.60x** | **2.3296 GB/s** | **11.2750 GB/s** |
| **Tier A 10k ($N=10k, M=100k, D=64$)** | MAX | Sequential Reference Loop | 86.7878 ms | 83.9624 ms | 101.8732 ms | 11.78% | 1.00x (Baseline) | 0.0691 GB/s | 0.3346 GB/s |
| **Tier A 10k ($N=10k, M=100k, D=64$)** | MAX | Vectorized NumPy/SciPy | 87.4159 ms | 86.5556 ms | 105.7071 ms | 15.90% | **0.99x (No gain)** | 0.0686 GB/s | 0.3322 GB/s |
| **Toy Graph ($N=5, M=6, D=16$)** | MEAN | Sequential Reference Loop | 0.0381 ms | 0.0357 ms | 0.0487 ms | 13.91% | 1.00x (Baseline) | 0.0193 GB/s | 0.0210 GB/s |
| **Toy Graph ($N=5, M=6, D=16$)** | MEAN | Vectorized SpMM (SciPy) | 0.0822 ms | 0.0759 ms | 0.1268 ms | 26.68% | **0.46x (Slowdown)** | 0.0090 GB/s | 0.0097 GB/s |
| **Toy Graph ($N=5, M=6, D=16$)** | SUM | Sequential Reference Loop | 0.0354 ms | 0.0342 ms | 0.0406 ms | 6.48% | 1.00x (Baseline) | 0.0208 GB/s | 0.0226 GB/s |
| **Toy Graph ($N=5, M=6, D=16$)** | SUM | Vectorized SpMM (SciPy) | 0.0726 ms | 0.0683 ms | 0.1034 ms | 18.60% | **0.49x (Slowdown)** | 0.0101 GB/s | 0.0110 GB/s |
| **Historical Phase 3 Baseline** | SUM | Sequential Reference Loop | 166.8770 ms | 166.2100 ms | 175.4000 ms | 3.12% | 1.00x | — | — |
| **Historical Phase 3 Baseline** | MEAN | Sequential Reference Loop | 133.1770 ms | 132.8500 ms | 141.2000 ms | 2.85% | 1.00x | — | — |
| **Historical Phase 3 Baseline** | MAX | Sequential Reference Loop | 167.1730 ms | 166.9000 ms | 176.1000 ms | 2.94% | 1.00x | — | — |
| **Historical Phase 3 Vectorized** | SUM | Vectorized SpMM (SciPy) | **3.6240 ms** | 3.5100 ms | 4.1200 ms | 8.45% | **46.05x** | — | — |
| **Historical Phase 3 Vectorized** | MEAN | Vectorized SpMM (SciPy) | **6.3250 ms** | 6.1800 ms | 7.0500 ms | 7.90% | **21.06x** | — | — |

> [!NOTE]
> **Key Scientific Observations:**
> 1. Vectorized sparse matrix multiplication (`scipy.sparse.csr_matrix.dot`) eliminates Python bytecode dispatch overhead, yielding a **44.60x speedup** on MEAN aggregation for large graphs ($N=10k$).
> 2. On tiny graphs (Toy $N=5$), vectorization exhibits a **0.46x - 0.49x slowdown** due to fixed C-extension and matrix instantiation overhead.
> 3. Standard sparse linear algebra libraries do **not** support generalized MAX semirings; hence CPU vectorized MAX resorts to segmented operations, yielding no speedup (**0.99x**), highlighting the direct necessity of custom CUDA kernels.
