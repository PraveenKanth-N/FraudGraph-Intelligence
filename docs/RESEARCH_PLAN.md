# Research Plan — FraudGraph Intelligence (FGI)

## 1. Research Questions & Hypotheses

### Primary Research Questions
- **RQ1:** How does irregular degree distribution in financial transaction graphs degrade GPU hardware utilization during neighbor feature aggregation?
- **RQ2:** To what extent can warp-synchronous programming and memory coalescing mitigate uncoalesced global memory reads in CSR-formatted graph feature propagation?
- **RQ3:** Does feature enrichment via localized GPU parallel aggregation provide statistically significant gains in classification performance (PR-AUC, F1-score) over non-relational baselines under severe class imbalance?

### Research Hypotheses
- **$H_1$ (Warp Coalescing):** A warp-per-node thread mapping where 32 consecutive threads coalesce feature dimension reads across neighbor lists will achieve at least a $2.5\times$ speedup over a naive thread-per-node mapping on power-law subgraphs.
- **$H_2$ (Transfer Ratio):** As graph density and feature dimensionality increase, the kernel execution time will transition from transfer-bound (PCIe dominant) to compute/memory-bandwidth bound, widening the GPU advantage over optimized CPU SIMD execution.
- **$H_3$ (Detection Quality):** Appending 1-hop and 2-hop aggregated neighbor feature vectors to raw tabular transaction features will increase downstream PR-AUC on the IEEE-CIS benchmark by $\ge 0.04$.

---

## 2. Experimental Methodology

### Graph Formulation
Let $G = (V, E, X)$ be a directed/undirected transaction graph, where:
- $V = \{v_1, v_2, \dots, v_N\}$ is the set of $N$ nodes (transactions or entities).
- $E \subseteq V \times V$ represents $M$ relational edges (shared card number, IP subnet, device fingerprint, or merchant ID).
- $X \in \mathbb{R}^{N \times D}$ is the node feature matrix with dimensionality $D$.

### Mathematical Formulation of Neighbor Aggregation
For each node $i \in V$, the aggregated neighborhood representation $h_i \in \mathbb{R}^D$ is computed as:
$$h_i = \bigoplus_{j \in \mathcal{N}(i)} \psi(x_j, e_{ij})$$
Where:
- $\mathcal{N}(i) = \{j \in V : (j, i) \in E\}$ denotes the incoming neighbor set of node $i$.
- $\bigoplus$ is a permutation-invariant aggregation operator: **Mean** ($\frac{1}{|\mathcal{N}(i)|} \sum$), **Sum** ($\sum$), or **Max** ($\max$).
- $\psi$ is an optional edge-weighted transformation: $\psi(x_j) = w_{ij} x_j$.

### Scaled Benchmark Tiers
To rigorously assess scalability, experiments will evaluate three standard graph tiers:
1. **Tier A (Small / Debug):** $N = 10,000$ nodes, $M = 50,000$ edges, $D = 32$.
2. **Tier B (Medium / Workload):** $N = 50,000$ nodes, $M = 350,000$ edges, $D = 64$.
3. **Tier C (Large / Stress):** $N = 200,000$ nodes, $M = 1,800,000$ edges, $D = 128$.
