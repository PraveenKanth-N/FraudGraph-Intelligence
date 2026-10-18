# Literature Review — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Document Status:** Phase 2 Authoritative Research Synthesis  
**Last Updated:** 2026-10-18T21:30:00+05:30  

---

## 1. Thematic Synthesis & Scope

This literature review establishes the scientific, algorithmic, and micro-architectural foundation for FraudGraph Intelligence (FGI). The analysis is organized into four core thematic areas:
1. **GPU Architecture, SIMT Execution, and Memory Subsystems:** Warp scheduling, memory coalescing, bank conflicts, and transfer bottlenecks.
2. **Sparse Matrix & Graph Primitive Acceleration (SpMV / SpMM):** CSR indexing, load balancing across irregular degrees, and GPU kernel design.
3. **Graph Neural Networks & Feature Aggregation Systems:** Specialized GNN backends and feature-dimension parallel schedules.
4. **Graph-Based Financial Fraud Detection:** Relational anomaly detection, fraudster camouflage, and benchmark datasets.

---

## 2. Verified Academic Publications & Official Documentation

### Theme 1: GPU Execution Models & Sparse Computations

#### Reference 1.1: Bell & Garland (2009) — Foundational Sparse GPU Acceleration
- **Full Citation:** Nathan Bell and Michael Garland. "Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors." *Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis (SC '09)*, Portland, OR, USA, November 2009.
- **Publisher / Identifier:** ACM / IEEE. DOI: `10.1145/1654059.1654078`. [Official PDF / Tech Report](https://www.nvidia.com/docs/IO/66880/nvr-2008-004.pdf).
- **Inspection Level:** Full technical report and paper inspected.
- **Specific Relevance to FGI:**
  - Evaluated sparse matrix storage formats on CUDA: CSR, COO, ELLPACK, DIA, and HYB.
  - Demonstrated that for Compressed Sparse Row (CSR), a naive scalar kernel (1 thread per row) suffers from uncoalesced memory access and severe warp divergence on irregular matrices.
  - Introduced the **CSR-Vector kernel** (assigning one 32-thread warp to each row), where warp threads cooperate to read column indices and non-zero values coalescently.
- **Direct Application:** FGI adopts the CSR-Vector paradigm to design the `fgi_aggregator_coalesced.cu` kernel, mapping 1 warp per node to gather neighbor features.
- **Limitations for FGI:** Bell & Garland focused on SpMV (vector multiplication where each non-zero element is a single scalar). In GNN neighbor aggregation, each node carries a dense feature vector of length $D \ge 32$, converting SpMV into Sparse Matrix-Dense Matrix Multiplication (SpMM).

#### Reference 1.2: NVIDIA Corporation — CUDA Programming Guide & Best Practices
- **Full Citation:** NVIDIA Corporation. *CUDA C++ Programming Guide* (Release 12.x / 13.x) and *CUDA C++ Best Practices Guide*. NVIDIA Developer Documentation, Santa Clara, CA, USA, 2024–2026.
- **Stable URL:** [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **Inspection Level:** Full official documentation inspected.
- **Specific Relevance to FGI:**
  - Formulates the exact hardware rules for **global memory coalescing**: global memory instructions load or store data in 32-byte, 64-byte, or 128-byte aligned transactions. If threads in a warp access contiguous addresses of 4-byte words, the hardware satisfies all 32 requests in a single 128-byte transaction.
  - Explains the SIMT (Single Instruction, Multiple Threads) branch divergence penalty: when threads within a warp diverge on dynamic conditions (e.g. `neighbor_count`), paths are serialized.
  - Documents microsecond timing methodology using `cudaEventRecord`, `cudaEventSynchronize`, and `cudaEventElapsedTime` to isolate kernel compute from PCIe transfers.
- **Limitations for FGI:** Hardware guides provide abstract micro-architectural rules rather than graph-specific algorithms; application-level mapping must be engineered specifically for power-law degree distributions.

#### Reference 1.3: Wang et al. (2016) — Gunrock Graph Processing
- **Full Citation:** Yangzihao Wang, Andrew Davidson, Yuechao Pan, Yuduo Wu, Andy Riffel, and John D. Owens. "Gunrock: A High-Performance Graph Processing Library on the GPU." *Proceedings of the 21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '16)*, Barcelona, Spain, March 2016.
- **Publisher / Identifier:** ACM. DOI: `10.1145/2851141.2851145`.
- **Inspection Level:** Full text inspected.
- **Specific Relevance to FGI:**
  - Introduced high-performance frontier-based graph processing on CUDA.
  - Demonstrated fine-grained load-balancing strategies to handle power-law graph topologies on GPUs, avoiding catastrophic warp idling when high-degree hub nodes are processed.
- **Limitations for FGI:** Gunrock focuses on topological graph traversal (BFS, SSSP, PageRank) where values per node are small scalars, rather than multi-dimensional feature tensor aggregations.

---

### Theme 2: Accelerated GNN Aggregation & SpMM Systems

#### Reference 2.1: Huang et al. (2020) — GE-SpMM
- **Full Citation:** Guyue Huang, Guohao Dai, Yu Wang, and Huazhong Yang. "GE-SpMM: General-Purpose SpMM for Graph Neural Network Acceleration." *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '20)*, Article 85, November 2020.
- **Publisher / Identifier:** IEEE / ACM. DOI: `10.1109/SC41405.2020.00089`. arXiv: `2007.03179`.
- **Inspection Level:** Full paper inspected.
- **Specific Relevance to FGI:**
  - Proves that SpMM in GNNs differs fundamentally from classical scientific SpMV because each edge operation involves a dense feature row $X[j, :]$ of dimension $D$.
  - Analyzed the memory traffic of naive GNN aggregation: uncoalesced random memory reads across global memory saturate the GPU DRAM bus.
  - Introduced coalesced row caching and coarse-grained warp merging natively on the CSR format, achieving $1.41\times$ speedup over NVIDIA cuSPARSE.
- **Direct Application:** FGI adapts the insight that feature dimension loops should be parallelized across warp threads to guarantee coalesced 128-byte memory transactions.
- **Limitations for FGI:** Evaluated on standard academic citation networks (Cora, Citeseer, Reddit) with homogeneous degree distributions, rather than skewed financial transaction graphs with transaction-card bipartite structures.

#### Reference 2.2: Hu et al. (2020) — FeatGraph
- **Full Citation:** Yuwei Hu, Zihao Ye, Minjie Wang, Jiali Yu, Da Zheng, Mu Li, Zheng Zhang, Zhiru Zhang, and Yida Wang. "FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems." *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '20)*, Article 71, November 2020.
- **Publisher / Identifier:** IEEE / ACM. DOI: `10.1109/SC41405.2020.00075`. arXiv: `2008.11359`.
- **Inspection Level:** Full paper inspected.
- **Specific Relevance to FGI:**
  - Demonstrated that GNN execution is governed by a **two-dimensional optimization space**: the sparse topological dimension (nodes/edges) and the dense feature dimension ($D$).
  - Showed that neither graph-only nor tensor-only optimization is sufficient; tiling both the neighbor list and the feature dimension is essential for maximizing GPU L1/L2 cache hit rates.
- **Direct Application:** Informs the FGI Shared-Memory Tiled kernel design (`fgi_aggregator_shared.cu`).

---

### Theme 3: Graph-Based Financial Fraud Detection

#### Reference 3.1: Dou et al. (2020) — Camouflage-Resistant GNNs (CARE-GNN)
- **Full Citation:** Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao Peng, and Philip S. Yu. "Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters." *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)*, Virtual Event, Ireland, October 2020, pp. 315–324.
- **Publisher / Identifier:** ACM. DOI: `10.1145/3340531.3411903`. arXiv: `2008.08692`.
- **Inspection Level:** Full paper inspected.
- **Specific Relevance to FGI:**
  - Analyzed **fraudster camouflage**: fraudsters intentionally connect to high-reputation benign entities (relation camouflage) and craft benign-looking metadata (feature camouflage) to dilute their suspiciousness in graph neighborhoods.
  - Showed that standard mean/sum neighbor aggregation can be polluted by adversarial edges, motivating edge-weighting and degree-aware normalization.
- **Direct Application:** Guides FGI's neighbor aggregation design, reinforcing that simple raw degree counting is insufficient and that normalized Mean aggregation ($\frac{1}{d_i} \sum X_j$) is necessary to prevent high-degree hubs from distorting node representations.

#### Reference 3.2: Wang et al. (2019) — SemiGNN for Financial Fraud
- **Full Citation:** Daixin Wang, Jianbin Lin, Peng Cui, Quanhui Jia, Zhen Wang, Yanming Fang, Quan Yu, Jun Zhou, Shuang Yang, and Yuan Qi. "A Semi-supervised Graph Attentive Network for Financial Fraud Detection." *Proceedings of the 2019 IEEE International Conference on Data Mining (ICDM '19)*, Beijing, China, November 2019, pp. 598–607.
- **Publisher / Identifier:** IEEE. DOI: `10.1109/ICDM.2019.00070`.
- **Inspection Level:** Full paper inspected.
- **Specific Relevance to FGI:**
  - Formulated financial fraud detection over multi-relational user-device-card networks at Alipay scale.
  - Proved that aggregating relational neighborhoods significantly improves classification recall over purely tabular gradient-boosted trees under extreme class imbalance ($< 1\%$ fraud rate).

#### Reference 3.3: Weber et al. (2019) — Elliptic Benchmark Dataset
- **Full Citation:** Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel K. I. Weidele, Claudio Bellei, Tom Robinson, and Charles E. Leiserson. "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics." *KDD '19 Workshop on Anomaly Detection in Finance*, Anchorage, AK, USA, August 2019.
- **Publisher / Identifier:** arXiv: `1908.02591`. [Kaggle Dataset Hub](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
- **Inspection Level:** Full paper inspected.
- **Specific Relevance to FGI:**
  - Published the Elliptic benchmark: 203,769 Bitcoin transaction nodes, 234,355 directed payment edges, and 166 features across 49 distinct time steps.
  - Established standard benchmark metrics: because fraud is heavily imbalanced, **PR-AUC (Precision-Recall Area Under Curve)** and **Minority-Class F1-Score** are the only defensible evaluation metrics; overall accuracy is clinically misleading.
- **Direct Application:** FGI adopts PR-AUC and minority F1-score as mandatory evaluation criteria.

---

## 3. Unverified References (Verification Needed)

The following works were surveyed in preliminary literature searches but have not been fully verified via independent publisher record checks or DOI resolution. In compliance with Phase 2 integrity rules, they are **excluded** from authoritative claims:
1. *Zhang et al. (2022) — "Ultra-Fast Graph Aggregation on Heterogeneous Architectures"*: Unconfirmed conference venue. Excluded until verified.
2. *Kumar & Patel (2021) — "Accelerating Financial Fraud Detection via Tensor Cores"*: Unverified preprint. Excluded.

---

## 4. Key Takeaways for FraudGraph Intelligence

1. **CSR Format is the Industry Standard:** Bell & Garland (2009) and Huang et al. (2020) prove that 0-indexed CSR arrays (`row_ptr`, `col_idx`) provide the optimal balance of compact memory footprint and direct GPU memory addressing.
2. **Coalesced Memory Access is Paramount:** In GNN feature aggregation, adjacent threads in a warp must load adjacent elements along the feature dimension $D$ of neighbor vectors, turning scattered reads into coalesced 128-byte DRAM transactions.
3. **Decoupled Benchmarking is Essential:** Evaluating GPU acceleration without isolating PCIe transfer time from pure kernel time produces invalid research claims. FGI will measure and report both.
