# Research Gap & Literature Context — FraudGraph Intelligence (FGI)

## 1. Survey of Existing Methodologies

| Approach | Typical Representative Work | Strengths | Critical Limitations in Financial Fraud Scenarios |
| :--- | :--- | :--- | :--- |
| **Tabular Tree Ensembles** | XGBoost, LightGBM, CatBoost | High speed, robust handling of non-linear tabular features, native categorical splits. | **Relational Blindness:** Completely ignores structural multi-hop connectivity between fraudulent transaction rings and card laundering hubs. |
| **Standard Deep Graph Frameworks** | PyTorch Geometric (PyG), DGL | High-level abstraction, extensive GNN layer library, automatic differentiation. | **Heavy Framework Overhead:** High Python interpreter latency; generic message-passing kernels unoptimized for sparse, highly skewed tabular financial graphs. |
| **Generic SpMM / SpGEMM Libraries** | cuSPARSE, Intel MKL | Highly optimized general sparse matrix multiplication. | **Rigid Operators & Memory Footprint:** Often assumes static adjacency without custom edge filters, non-linear pooling, or hybrid memory staging tailored to tabular feature tensors. |

---

## 2. Identified Research Gaps

1. **Warp Serialization under Financial Power-Law Graphs:**
   Standard GNN libraries allocate threads naively (thread-per-node or thread-per-edge). In financial graphs, fraud hubs create nodes with degrees $> 10^3$, while average nodes have degree $\sim 5$. When grouped in a 32-thread warp, 31 threads stall while the thread processing the super-node loops through thousands of neighbors, degrading warp efficiency below 15%.

2. **Uncoalesced Global Memory Gathering:**
   Gathering neighbor features $X[col\_idx[k], :]$ requires irregular indexing into global memory. Without explicit feature-dimension thread mapping, memory controllers cannot combine adjacent requests into 32-byte or 128-byte transactions, saturating GPU memory buses with redundant cache line fetches.

3. **Separation of Computation vs. Transfer Overhead:**
   Many published academic papers report GPU speedups while conflating kernel execution time with host-to-device (H2D) PCIe transfer times. In real-world fraud scoring, if data transfer dominates kernel time, naive GPU offloading yields net latency regression.

---

## 3. FraudGraph Intelligence Contributions

- **Mathematical Specialization:** Tailoring CSR parallel aggregation specifically to fraud detection topological patterns.
- **Micro-Architectural CUDA Design:** Explicitly comparing thread-per-node vs. warp-per-node thread mappings and evaluating shared-memory caching.
- **Transparent Benchmarking:** Strictly decoupling kernel-only execution time (via CUDA events) from PCIe memory transfers and end-to-end pipeline latency.
