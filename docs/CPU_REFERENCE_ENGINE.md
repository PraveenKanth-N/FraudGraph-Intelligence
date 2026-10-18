# CPU Reference Aggregation Engine — Architecture & Specification

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Course Code:** CSI4007  
**Project:** FraudGraph Intelligence (FGI)  
**Module:** `backend/cpu_engine/aggregators.py`  
**Phase:** Phase 4 Deliverable  

---

## 1. Overview & Research Role

The **CPU Reference Aggregation Engine** serves as the authoritative mathematical and numerical baseline for FraudGraph Intelligence. Its primary responsibilities are:
1. Providing an unimpeachable ground truth for validating CUDA C++ parallel kernel outputs in Phase 5 (Baseline Kernel) and Phase 6 (Memory-Aware Optimizations).
2. Establishing baseline CPU throughput and latency metrics under reproducible conditions to compute honest GPU speedup factors ($S = t_{\text{CPU}} / t_{\text{GPU}}$).
3. Strictly enforcing all project invariants: single-precision floating-point arithmetic (`np.float32`), explicit zero-degree sentinel vectors ($\vec{0} \in \mathbb{R}^D$), input immutability, and directed/undirected graph semantics.

---

## 2. Mathematical Operators & Formulas

Given a transaction graph in canonical Compressed Sparse Row (CSR) format with $N$ nodes, $M$ directed adjacency entries, and feature tensor $X \in \mathbb{R}^{N \times D}$:

For each node $i \in \{0, \dots, N-1\}$ and feature column $k \in \{0, \dots, D-1\}$, its neighbor index set is defined by the contiguous slice of `col_idx`:
$$\mathcal{N}(i) = \Big\{ \text{col\_idx}[p] \;\Big|\; p \in [\text{row\_ptr}[i],\, \text{row\_ptr}[i+1] - 1] \Big\}$$
Its out-degree (stored neighbor count) is:
$$d_i = |\mathcal{N}(i)| = \text{row\_ptr}[i+1] - \text{row\_ptr}[i]$$

### 2.1 Sum Aggregation (`sum`)
Accumulates the raw feature vectors across all adjacent neighbors:
$$Y_{\text{sum}}[i, k] = \begin{cases} 
\sum_{j \in \mathcal{N}(i)} X[j, k], & d_i > 0 \\ 
0.0, & d_i = 0 \text{ (Isolated Node)} 
\end{cases}$$

### 2.2 Mean Aggregation (`mean`) — Primary Benchmark Operator
Averages adjacent neighbor features, strictly guarded against division-by-zero:
$$Y_{\text{mean}}[i, k] = \begin{cases} 
\frac{1}{d_i} \sum_{j \in \mathcal{N}(i)} X[j, k], & d_i > 0 \\ 
0.0, & d_i = 0 \text{ (Isolated Node)} 
\end{cases}$$

### 2.3 Max Aggregation (`max`)
Extracts the component-wise supremum across adjacent neighbors:
$$Y_{\text{max}}[i, k] = \begin{cases} 
\max_{j \in \mathcal{N}(i)} X[j, k], & d_i > 0 \\ 
0.0, & d_i = 0 \text{ (Strict Zero-Sentinel Policy)} 
\end{cases}$$

> [!IMPORTANT]
> **Zero-Sentinel Policy for Max Aggregation:** Standard algebraic semiring reduction defines the supremum over the empty set as $\max(\emptyset) \triangleq -\infty$. In financial feature engineering, allowing $-\infty$ induces floating-point exceptions and propagates NaNs into neural layers. FGI enforces an explicit zero-vector return ($\vec{0}$) whenever $d_i = 0$.

---

## 3. Implementation Architectures

The engine provides two decoupled execution pathways in [`backend/cpu_engine/aggregators.py`](file:///D:/FraudGraph%20Intelligence/backend/cpu_engine/aggregators.py):

### 3.1 Method A: Reference Node-Iterative Engine (`method="reference"`)
- **Design Philosophy:** Transparent, step-by-step neighbor slicing matching the mathematical definition directly.
- **Algorithm:**
  1. Preallocate $Y \in \mathbb{R}^{N \times D}$ initialized with zeros in `np.float32`.
  2. For each node $i \in \{0, \dots, N-1\}$:
     - Read start offset $s = \text{row\_ptr}[i]$ and end offset $e = \text{row\_ptr}[i+1]$.
     - If $e > s$, slice neighbor indices `idx = col_idx[s:e]`.
     - Vectorize feature accumulation over the $D$ columns using NumPy C-level reduction:
       - Sum: `np.sum(features[idx], axis=0, dtype=np.float32)`
       - Mean: `np.sum(features[idx], axis=0) / deg`
       - Max: `np.max(features[idx], axis=0)`
- **Strengths:** 100% auditable, zero complex dependencies, guarantees zero side effects, immune to sparse library matrix-multiplication rounding variations.

### 3.2 Method B: Vectorized Sparse Matrix Engine (`method="vectorized"`)
- **Design Philosophy:** High-performance sparse-dense matrix multiplication (SpMM) leveraging SciPy CSR routines and BLAS.
- **Algorithm:**
  - **Sum:** Constructs adjacency matrix $A = \text{CSR}(1.0, \text{col\_idx}, \text{row\_ptr})$ and evaluates SpMM product:
    $$Y_{\text{sum}} = A \cdot X$$
  - **Mean:** Precomputes inverse non-zero degrees $\vec{w}$ where $w_i = 1 / d_i$ if $d_i > 0$ else $0$. Replicates weights along row edges to construct row-normalized matrix $\tilde{A}$, then evaluates:
    $$Y_{\text{mean}} = \tilde{A} \cdot X$$
  - **Max:** Dispatches to optimized segment reduction (SpMM does not natively support max-plus semirings).
- **Strengths:** Explores multithreaded BLAS parallelization, executing 10,000 nodes in under 5 milliseconds on modern multi-core CPUs.

---

## 4. Verification Evidence & Numerical Tolerance

### 4.1 Numerical Tolerance Policy
- **Absolute Error Ceiling:** $\text{atol} \le 1.0 \times 10^{-5}$.
- Floating-point addition across varying summation orders (BLAS SIMD vector trees vs. sequential NumPy loops) is associative only up to machine epsilon. Empirical max absolute difference between Reference and Vectorized SpMM on synthetic graphs is $0.0$ to $4.7 \times 10^{-7}$, well within the $10^{-5}$ threshold.

### 4.2 Automated Test Coverage
All 18 core requirements are verified by automated tests in [`backend/tests/test_aggregators.py`](file:///D:/FraudGraph%20Intelligence/backend/tests/test_aggregators.py):
1. Hand-calculated sum aggregation on 5-node graph.
2. Hand-calculated mean aggregation on 5-node graph.
3. Hand-calculated max aggregation on 5-node graph.
4. Node with no neighbors returns explicit zero vector.
5. Entirely empty graph ($N=0$) returns $(0, D)$ float32 array.
6. Single-node graph (with and without self-loop).
7. Directed graphs with asymmetric adjacency.
8. Undirected graphs with symmetric adjacency.
9. Graphs with extreme degree skew (star graph with 500 peripheral nodes).
10. Duplicate edges and self-loop policy enforcement.
11. Multi-scale feature dimensions: $D \in \{1, 4, 16, 32, 128\}$.
12. Negative and zero-valued features (verifying max does not erroneously return zero for negative inputs).
13. Deterministic output reproducibility across repeated invocations.
14. Immutability of input graph arrays (`row_ptr`, `col_idx`, `features`).
15. Strict defensive input validation (`TypeError` and `ValueError`).
16. Guaranteed shape $(N, D)$ and dtype `float32`.
17. Rejection of non-finite inputs (`NaN` or `Inf`).
18. Cross-verification between reference and vectorized implementations.

---

## 5. Measured CPU Benchmark Performance

Empirically measured on host workstation (Windows 11, Intel Core i5-1135G7 @ 2.40GHz, 8 logical cores, Python 3.14.3):

| Fixture | Nodes ($N$) | Edges ($M$) | Dim ($D$) | Operator | Method | Mean Latency (ms) | Median Latency (ms) | Throughput (Edges/sec) | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`toy_graph_5nodes.npz`** | 5 | 8 | 4 | Mean | Reference | 0.033 ms | 0.033 ms | 245,927 | `PASSED` |
| `toy_graph_5nodes.npz` | 5 | 8 | 4 | Mean | Vectorized | 0.036 ms | 0.036 ms | 219,118 | `PASSED` |
| `toy_graph_5nodes.npz` | 5 | 8 | 4 | Sum | Reference | 0.023 ms | 0.023 ms | 352,734 | `PASSED` |
| `toy_graph_5nodes.npz` | 5 | 8 | 4 | Sum | Vectorized | 0.028 ms | 0.027 ms | 288,392 | `PASSED` |
| `toy_graph_5nodes.npz` | 5 | 8 | 4 | Max | Reference | 0.023 ms | 0.023 ms | 351,958 | `PASSED` |
| **`synthetic_tier_a_10k.npz`** | 10,000 | 94,945 | 32 | Mean | Reference | 133.177 ms | 80.228 ms | 712,923 | `PASSED` |
| `synthetic_tier_a_10k.npz` | 10,000 | 94,945 | 32 | Mean | Vectorized (SpMM) | **6.325 ms** | **6.097 ms** | **15,011,945** | `PASSED` |
| `synthetic_tier_a_10k.npz` | 10,000 | 94,945 | 32 | Sum | Reference | 166.877 ms | 166.250 ms | 568,953 | `PASSED` |
| `synthetic_tier_a_10k.npz` | 10,000 | 94,945 | 32 | Sum | Vectorized (SpMM) | **3.624 ms** | **3.581 ms** | **26,196,494** | `PASSED` |
| `synthetic_tier_a_10k.npz` | 10,000 | 94,945 | 32 | Max | Reference | 167.173 ms | 167.582 ms | 567,945 | `PASSED` |

Full benchmark observation records are stored in machine-readable JSON under [`experiments/results/`](file:///D:/FraudGraph%20Intelligence/experiments/results/).
