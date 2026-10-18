# Project Charter — FraudGraph Intelligence (FGI)

## 1. Project Overview & Institutional Identity

- **Course:** VIT Vellore — CSI4007 GPU Programming
- **Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation
- **Project ID:** FGI
- **Repository Root:** `D:\FraudGraph Intelligence`
- **Responsible Engineering Agent:** Antigravity (Google DeepMind)
- **Academic Mentor / Review:** Faculty Review Committee, School of Computer Science & Engineering (SCOPE), VIT Vellore

---

## 2. Problem Statement & Motivation

Financial transaction fraud poses billions of dollars in annual losses worldwide. Fraudsters rarely operate in isolation; they leverage coordinated fraud rings, shared synthetic identities, shared device fingerprints, and distributed micro-transactions to evade rule-based filters.

Graph Neural Networks (GNNs) and structural graph embeddings offer unmatched detection capabilities by propagating relational features across nodes in a transaction network. However, transaction networks feature:
1. **Severe Power-Law Degree Skew:** A minority of "super-nodes" (e.g., major merchant IDs or gateway hubs) have thousands of neighbors, while ordinary transactions have few.
2. **Memory Bandwidth Bottlenecks:** Message passing requires fetching high-dimensional feature vectors from non-contiguous memory locations (irregular neighbor gathering).
3. **GPU Thread Divergence:** Naive parallel mappings (e.g., assigning one GPU thread per node) cause severe intra-warp branch divergence and catastrophic load imbalance when neighbor list lengths differ by orders of magnitude.

This capstone project engineers a **Memory-Aware CUDA Acceleration Engine** for Parallel Neighbor Aggregation, benchmarked against a verified CPU reference implementation, and integrated into a full-stack forensic research platform.

---

## 3. Scope of Work

### In-Scope
1. **Graph Representation:** Construction and verification of Compressed Sparse Row (CSR) adjacency representations (`row_ptr`, `col_idx`) from real-world financial transaction data (IEEE-CIS Fraud Detection benchmark).
2. **CPU Reference Engine:** Deterministic, vector-parallel neighbor aggregator in Python (NumPy/SciPy) establishing numerical tolerance baselines.
3. **CUDA Compute Engines:**
   - *Baseline Kernel:* Thread-per-node parallel aggregation.
   - *Optimized Kernel:* Warp-per-node mapping with memory coalescing.
   - *Shared-Memory Kernel:* Tiled aggregation exploiting threadblock shared memory.
4. **Fraud Evaluation Pipeline:** Evaluating downstream fraud detection performance (PR-AUC, F1-score, Precision, Recall) using enriched neighbor embeddings.
5. **Experimentation & Telemetry:** Rigorous benchmarking recording kernel time (via CUDA events), data transfer overhead, speedup factors, and variance.
6. **Web Interface:** React 18 + TypeScript + Vite frontend utilizing the *Obsidian Cybernetics* design system and FastAPI backend.

### Out-of-Scope
- Real-time banking network ingestion or live card processing.
- Multi-node distributed GPU clustering (MPI / NCCL).
- Arbitrary custom GNN training from scratch (the primary focus is parallel neighbor aggregation kernel performance).

---

## 4. Key Performance Indicators (KPIs)

1. **Kernel Correctness:** Absolute numerical tolerance between CPU reference and CUDA outputs: $\text{atol} \le 10^{-5}$.
2. **Speedup:** Measured speedup of optimized CUDA aggregation over CPU reference baseline across varied graph scales (10K to 500K nodes).
3. **Memory Coalescing Efficiency:** Significant reduction in uncoalesced global memory transactions verified via memory-access pattern analysis.
4. **Classification Performance:** Measured improvement in fraud detection F1-score and PR-AUC when augmenting raw transaction features with aggregated graph neighborhood representations.
