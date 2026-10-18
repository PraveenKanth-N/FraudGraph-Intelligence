# Algorithm Specification & Mathematical Foundation — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Document Status:** Phase 2 Authoritative Specification  

---

## 1. Formal Mathematical Notation

Let $G = (V, E)$ be a transaction graph where:
- $V = \{0, 1, \dots, N-1\}$ is the set of $N$ nodes.
- $E \subseteq V \times V$ is the set of $M$ directed or undirected edges.
- $X \in \mathbb{R}^{N \times D}$ is the input node feature matrix stored in row-major order, where $X[i, :]$ represents the $D$-dimensional feature vector of node $i$.

### 1.1 Compressed Sparse Row (CSR) Representation
The graph topology is defined by two integer arrays:
1. `row_ptr` $\in \mathbb{Z}^{N+1}$: Monotonically non-decreasing array where `row_ptr[0] = 0` and `row_ptr[N] = M`.
   - The degree (neighbor count) of node $i$ is:
     $$d_i = \text{row\_ptr}[i+1] - \text{row\_ptr}[i]$$
2. `col_idx` $\in \mathbb{Z}^{M}$: Contiguous array containing the neighbor indices.
   - For node $i$, the incoming/adjacent neighbors are given by:
     $$\mathcal{N}(i) = \Big\{ \text{col\_idx}[p] \;\Big|\; p \in [\text{row\_ptr}[i],\, \text{row\_ptr}[i+1] - 1] \Big\}$$

---

## 2. Permutation-Invariant Aggregation Operators

For each node $i \in V$ and feature dimension $k \in \{0, \dots, D-1\}$, the aggregated output tensor element $Y[i, k]$ is defined across its neighborhood $\mathcal{N}(i)$:

### 2.1 Operator 1: Mean Aggregator (Primary Benchmark Operator)
The Mean operator averages the feature values of all adjacent neighbors:
$$Y_{\text{mean}}[i, k] = \begin{cases} 
\frac{1}{d_i} \sum_{p = \text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} X[\text{col\_idx}[p], k] & \text{if } d_i > 0 \\
0.0 & \text{if } d_i = 0 \text{ (Isolated Node)}
\end{cases}$$
*Zero-Degree Handling:* Explicitly guarded against division-by-zero. Isolated nodes with $d_i = 0$ output a zero vector ($\vec{0} \in \mathbb{R}^D$).

### 2.2 Operator 2: Sum Aggregator
The Sum operator accumulates total feature mass across adjacent neighbors:
$$Y_{\text{sum}}[i, k] = \begin{cases} 
\sum_{p = \text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} X[\text{col\_idx}[p], k] & \text{if } d_i > 0 \\
0.0 & \text{if } d_i = 0 \text{ (Isolated Node)}
\end{cases}$$
*Zero-Degree Handling:* Empty sum evaluates naturally to $0.0$.

### 2.3 Operator 3: Max Aggregator (Specialized Considerations)
The Max operator identifies extreme feature values across adjacent neighbors:
$$Y_{\text{max}}[i, k] = \begin{cases} 
\max_{p \in [\text{row\_ptr}[i], \text{row\_ptr}[i+1]-1]} X[\text{col\_idx}[p], k] & \text{if } d_i > 0 \\
0.0 & \text{if } d_i = 0 \text{ (Isolated Node Default)}
\end{cases}$$
*Defensibility Note on Max Aggregation:* Standard mathematical reduction defines the empty maximum as $-\infty$. However, in downstream neural networks and gradient calculations, $-\infty$ produces severe numerical instability and NaN propagation. Therefore, FGI specifies a strict zero-sentinel policy: **if $d_i = 0$, $Y_{\text{max}}[i, k] = 0.0$**. Mean aggregation remains the primary benchmark operator.

---

## 3. Algorithm Architectures & Kernel Designs

### 3.1 Architecture A: CPU Reference Engine (`neighbor_aggregator.py`)
- **Execution Model:** Vectorized NumPy / SciPy sparse linear algebra using BLAS/SIMD primitives.
- **Computation:** Computes $Y_{\text{sum}} = A \cdot X$ using sparse matrix-dense matrix multiplication, followed by row-wise normalization by the in-degree vector $\vec{d} = \max(\vec{d}, 1)$.
- **Role:** Deterministic gold standard for numerical tolerance and baseline latency.

### 3.2 Architecture B: Baseline CUDA Kernel (`fgi_aggregator_baseline.cu`)
- **Mapping:** 1 CUDA thread assigned per node $i$.
- **Grid Configuration:**
  - Block size: 256 threads (`dim3 block(256);`).
  - Grid size: $\lceil N / 256 \rceil$ blocks (`dim3 grid((N + 255) / 256);`).
- **Algorithm (Thread $t$):**
  1. Calculate `int i = blockIdx.x * blockDim.x + threadIdx.x;`
  2. If $i \ge N$, exit.
  3. Load `start = row_ptr[i]` and `end = row_ptr[i+1]`.
  4. Compute `int degree = end - start;`
  5. For each dimension $k \in [0, D-1]$:
     - Initialize `float acc = 0.0f;`
     - For `p = start` to `end - 1`:
       - `int neighbor = col_idx[p];`
       - `acc += features[neighbor * D + k];`
     - Write $Y[i * D + k] = (\text{degree} > 0) \,?\, (\text{acc} / \text{degree}) : 0.0\text{f};$
- **Micro-Architectural Flaws:**
  - Adjacent threads in the warp process adjacent nodes $i$ and $i+1$, whose neighbors $\mathcal{N}(i)$ and $\mathcal{N}(i+1)$ are disjoint. Memory accesses into `features` are scattered and uncoalesced.
  - When node $i$ has degree 2 and node $i+1$ has degree 1,500, thread $i$ finishes in 2 iterations but stalls inside the warp until thread $i+1$ completes all 1,500 iterations.

### 3.3 Architecture C: Memory-Coalesced Warp-per-Node Kernel (`fgi_aggregator_coalesced.cu`)
- **Mapping:** 1 Warp (32 consecutive threads) assigned to each node $i$.
- **Grid Configuration:**
  - Block size: 256 threads (8 warps per block).
  - Total warps: $N$. Grid size: $\lceil N / 8 \rceil$ blocks.
- **Algorithm (Warp $w$ on Node $i$, Lane $\ell \in [0, 31]$):**
  1. Determine node index $i = (\text{blockIdx.x} \times 8) + (\text{threadIdx.x} / 32)$.
  2. If $i \ge N$, exit.
  3. Lane 0 loads `start = row_ptr[i]` and `end = row_ptr[i+1]`, broadcasting to warp via `__shfl_sync(0xFFFFFFFF, ..., 0)`.
  4. All 32 threads in the warp cooperatively loop over feature dimensions:
     - Thread $\ell$ processes dimensions $k = \ell, \ell + 32, \ell + 64, \dots < D$.
     - For each dimension $k$, accumulate:
       $$\text{acc} = \sum_{p=\text{start}}^{\text{end}-1} X[\text{col\_idx}[p] \times D + k]$$
     - Because threads $\ell = 0 \dots 31$ access addresses $(\text{neighbor} \times D) + 0, \dots, (\text{neighbor} \times D) + 31$, all 32 memory requests fall into a single aligned 128-byte cache line.
  5. Store output coalescently into $Y[i \times D + k]$.
- **Hardware Benefits:**
  - **100% Coalesced Global Memory Access:** Eliminates memory bus thrashing.
  - **Zero Intra-Warp Degree Divergence:** All 32 threads in the warp iterate over the exact same neighbor list bounds $[\text{start}, \text{end})$.

### 3.4 Architecture D: Shared-Memory Tiled Kernel (`fgi_aggregator_shared.cu`)
- **Mapping:** Threadblock shared-memory buffer caching tiles of `col_idx` and feature slices.
- **Optimization:** Alleviates repeated global memory lookups for dense hub subgraphs.

---

## 4. Numerical Equivalence & Tolerance Thresholds

To verify correctness across single-precision floating-point arithmetic (IEEE 754 float32), outputs are evaluated against the double-precision rounded CPU reference matrix:

$$\text{Max Absolute Error:} \quad \epsilon_{\max} = \max_{i \in [0, N-1],\, k \in [0, D-1]} |Y_{\text{GPU}}[i, k] - Y_{\text{CPU}}[i, k]|$$
$$\text{Relative Error:} \quad \epsilon_{\text{rel}} = \frac{|Y_{\text{GPU}}[i, k] - Y_{\text{CPU}}[i, k]|}{|Y_{\text{CPU}}[i, k]| + 10^{-7}}$$

### Verification Gates:
1. **Mean & Sum Aggregation Gate:** $\epsilon_{\max} \le 1.0 \times 10^{-5}$ across $\ge 99.99\%$ of elements, with no individual element exceeding $5.0 \times 10^{-5}$ (accounting for non-associative floating-point addition order).
2. **Max Aggregation Gate:** $\epsilon_{\max} == 0.0$ (exact bitwise equivalence since max is purely a selection operator).
3. **Exceptional Cases:**
   - Any NaN or $\pm\infty$ in output triggers an immediate **TEST_FAILED** condition.
   - Any degree-0 isolated node must produce exactly $0.0\text{f}$ across all $D$ dimensions.
