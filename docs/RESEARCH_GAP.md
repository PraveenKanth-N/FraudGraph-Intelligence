# Evidence-Backed Research Gap Analysis — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Authoritative References:** Bell & Garland (SC '09), Huang et al. (SC '20), Hu et al. (SC '20), Dou et al. (CIKM '20), Weber et al. (KDD '19).

---

## 1. Survey of Existing Paradigms & Their Bottlenecks

### 1.1 Tabular Tree Ensembles (XGBoost, LightGBM, CatBoost)
- **Current Practice:** Standard industrial fraud detection relies heavily on gradient-boosted decision trees trained on raw transaction features (e.g. transaction amount, timestamp deltas, categorical payment card codes).
- **Critical Limitation (Relational Blindness):** As demonstrated by Wang et al. (ICDM '19) and Dou et al. (CIKM '20), tabular classifiers evaluate each transaction in isolation. They are fundamentally incapable of recognizing coordinated fraud structures (e.g., synthetic fraud rings, card-testing attacks, or rapid multi-account fund layering) that manifest strictly as relational graph topologies.

### 1.2 Generic Deep Graph Frameworks (PyTorch Geometric, DGL)
- **Current Practice:** Frameworks like PyTorch Geometric (PyG) and Deep Graph Library (DGL) implement generalized message-passing interfaces (`MessagePassing.propagate()`) to support diverse GNN architectures (GCN, GAT, GraphSAGE).
- **Critical Limitation (Framework & Memory Overhead):**
  - **Memory Allocation Churn:** PyG and DGL rely on dynamic Python memory allocations and high-level PyTorch tensor dispatches, introducing substantial runtime interpreter overhead during simple aggregation passes.
  - **Uncoalesced Memory Gathering:** As established by Huang et al. (SC '20, *GE-SpMM*), generic gather-scatter implementations in PyTorch Geometric trigger irregular global memory accesses across high-dimensional feature matrices, failing to saturate GPU DRAM bandwidth.
  - **Lack of Financial Domain Specialization:** Generic frameworks assume static, homogeneous citation or social networks. They do not natively handle the extreme degree skew and bipartite entity-transaction projections typical of financial transaction graphs.

### 1.3 General-Purpose Sparse Linear Algebra Libraries (cuSPARSE, Intel MKL)
- **Current Practice:** High-performance libraries offer vendor-tuned Sparse Matrix-Vector (SpMV) and Sparse Matrix-Dense Matrix Multiplication (SpMM) routines (`cusparseSpMM`).
- **Critical Limitation (Algorithmic Inflexibility):**
  - General SpMM libraries are strictly optimized for matrix multiplication ($Y = \alpha A X + \beta C$). They do not natively support customized non-linear reduction operators, localized degree normalization (Mean aggregation with zero-degree protection), or fused edge-weighting functions without allocating additional intermediate memory buffers.
  - As proven by Hu et al. (SC '20, *FeatGraph*), vendor SpMM kernels often optimize exclusively for the sparse matrix dimension, failing to co-optimize the dense feature dimension schedule ($D$).

---

## 2. Explicit Gap Formulations for FraudGraph Intelligence

| Research Gap | Prior Academic & Industry Evidence | FGI Concrete Engineering Address |
| :--- | :--- | :--- |
| **Gap 1: Warp Divergence under Power-Law Degree Skew** | In financial transaction graphs, a small fraction of merchant hubs or card payment gateways have degrees exceeding $10^3$, while typical transactions have degree $< 5$. Bell & Garland (SC '09) established that naive thread-per-row mapping causes extreme warp serialization on such matrices. | Implement a **Warp-per-Node** execution model (`fgi_aggregator_coalesced.cu`), dedicating an entire 32-thread warp to each node's neighbor list, balancing workloads and eliminating intra-warp branch stalls. |
| **Gap 2: Uncoalesced Memory Gathering in GNN Aggregation** | Huang et al. (SC '20) showed that loading neighbor features $X[\text{col\_idx}[e], :]$ in a scalar loop generates non-contiguous memory requests that waste up to 75% of peak global memory bandwidth. | Map consecutive warp threads to contiguous indices along the feature dimension $D$, guaranteeing that 32 threads issue aligned 128-byte DRAM transactions. |
| **Gap 3: Conflation of Kernel Time vs. PCIe Data Transfer** | Many published GPU benchmark papers report inflated speedup numbers by ignoring Host-to-Device (H2D) and Device-to-Host (D2H) PCIe transfer overhead. In latency-sensitive fraud pipelines, transfer time can dominate compute time. | Mandate strict decoupled telemetry recording via CUDA events (`cudaEventRecord`), logging kernel execution time and PCIe transfers separately, and reporting both kernel-only and end-to-end speedups. |
| **Gap 4: Target Leakage in Graph-Based Fraud Benchmarks** | Dou et al. (CIKM '20) and Weber et al. (KDD '19) emphasize that fraud labels (`isFraud`) must be strictly quarantined from graph structure and neighbor features to prevent historical leakage. | Formulate strict mathematical graph construction rules where edges and features depend exclusively on pre-transaction metadata, validating leakage prevention prior to evaluation. |
