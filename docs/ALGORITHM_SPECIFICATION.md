# Algorithm Specification — FraudGraph Intelligence (FGI)

## 1. Mathematical Formalism

Let $G = (V, E)$ be a transaction graph represented in Compressed Sparse Row (CSR) format with arrays `row_ptr` $\in \mathbb{Z}^{N+1}$ and `col_idx` $\in \mathbb{Z}^M$. Let $X \in \mathbb{R}^{N \times D}$ denote the input feature matrix.

The objective is to compute the aggregated neighborhood feature matrix $Y \in \mathbb{R}^{N \times D}$, where for each node $i \in \{0, \dots, N-1\}$ and feature dimension $k \in \{0, \dots, D-1\}$:

$$Y[i, k] = \text{AGG}_{j \in \mathcal{N}(i)} \Big( X[j, k] \Big)$$

For the default **Mean Aggregator**:
$$Y[i, k] = \begin{cases} 
\frac{1}{d_i} \sum_{p = \text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} X[\text{col\_idx}[p], k] & \text{if } d_i > 0 \\
0 & \text{if } d_i = 0 
\end{cases}$$
where $d_i = \text{row\_ptr}[i+1] - \text{row\_ptr}[i]$ is the degree of node $i$.

---

## 2. Kernel Design Implementations

### 2.1 Kernel 1: Baseline Thread-per-Node (`fgi_aggregator_baseline.cu`)
- **Mapping:** 1 CUDA thread is assigned to each node $i$.
- **Thread Indexing:** `int node_idx = blockIdx.x * blockDim.x + threadIdx.x;`
- **Execution:**
  ```cpp
  if (node_idx < num_nodes) {
      int start = row_ptr[node_idx];
      int end = row_ptr[node_idx + 1];
      int degree = end - start;
      for (int k = 0; k < feat_dim; ++k) {
          float sum = 0.0f;
          for (int e = start; e < end; ++e) {
              int neighbor = col_idx[e];
              sum += features[neighbor * feat_dim + k];
          }
          out_features[node_idx * feat_dim + k] = (degree > 0) ? (sum / degree) : 0.0f;
      }
  }
  ```
- **Bottlenecks:**
  - Severe branch divergence across threads in a warp when degrees differ.
  - Non-coalesced global memory reads when accessing `features[neighbor * feat_dim + k]`.

### 2.2 Kernel 2: Memory-Coalesced Warp-per-Node (`fgi_aggregator_coalesced.cu`)
- **Mapping:** 1 full warp (32 threads) is assigned to each node $i$.
- **Execution:**
  - Warp lane ID: `int lane_id = threadIdx.x % 32;`
  - Warp ID: `int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;`
  - Threads in the warp collaborate over the feature dimension $k$:
    - Thread `lane_id` processes dimensions $k = \text{lane\_id}, \text{lane\_id} + 32, \dots < D$.
    - When accessing `features[neighbor * feat_dim + k]`, 32 consecutive threads read 32 consecutive 4-byte floating-point values, perfectly coalescing into single 128-byte transactions.

### 2.3 Kernel 3: Shared-Memory Tiling (`fgi_aggregator_shared.cu`)
- **Mapping:** Threadblock cooperatively loads chunks of neighbor indices and feature dimensions into fast `__shared__` memory buffers.
- **Optimization:** Reduces repeated global memory latency for high-degree hub nodes.
