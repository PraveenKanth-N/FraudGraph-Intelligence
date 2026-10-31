# Table 1: Synthetic Transaction Graph & Evaluation Split Characteristics

**Dataset Identifier:** `synthetic_fraud_10k`  
**Storage Format:** Compressed Sparse Row (CSR) NPZ Fixture  
**Canonical SHA-256:** `aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430`  
**Temporal Span:** 30.0 Days ($2,592,000$ seconds) | **Lookback Window:** 7.0 Days ($604,800$ seconds)  

| Parameter / Structural Property | Metric Value | Verification Source | Definition / Engineering Role |
| :--- | :--- | :--- | :--- |
| **Total Transaction Nodes ($N$)** | 10,000 | `synthetic_fraud_10k.npz` | Total transaction vertices in chronological sequence. |
| **Input Feature Dimension ($D$)** | 32 continuous features | `features.shape[1]` | Standardized transactional and entity features ($X \in \mathbb{R}^{N \times D}$). |
| **Directed Causal Edges ($M$)** | 231,984 edges | `col_idx.shape[0]` | Relational edges linking transactions sharing `card_id` or `device_id`. |
| **CSR Row Offsets ($N + 1$)** | 10,001 entries | `row_ptr.shape[0]` | 32-bit integer array of row pointer offsets. |
| **Isolated Transactions ($d_i = 0$)** | 1,155 (11.55%) | `synthetic_fraud_10k_manifest.json` | First-time buyers or isolated cards with no prior transactions in window. |
| **Graph Density ($\rho$)** | 0.00232 | `manifest.stats.density` | Computed as $M / (N(N-1))$; highly sparse relational structure. |
| **In-Degree Min / Mean / Median** | 0 / 23.20 / 6.0 | `manifest.stats.degree_*` | Highly skewed topology; typical node has 6 historical neighbors. |
| **In-Degree 90th / 99th Percentile** | 71.0 / 143.0 | `manifest.stats.degree_*` | Hub entities exhibiting high transaction velocity. |
| **In-Degree Maximum ($d_{\max}$)** | 252 edges | `manifest.stats.degree_max` | Extreme hub causing catastrophic SIMT warp serialization in naive CUDA. |
| **Total Fraud Cases ($Y=1$)** | 350 transactions | `labels.sum()` | Verified ground truth positive labels. |
| **Total Licit Cases ($Y=0$)** | 9,650 transactions | `10000 - labels.sum()` | Background legitimate transactions. |
| **Overall Fraud Prevalence** | 3.50% | `fraud_stats.fraud_percentage` | Extreme class imbalance reflecting real-world banking regimes. |
| **Collusive Ring Fraud Cases** | 245 cases (70.0% of fraud) | `manifest.label_breakdown` | Circular/clique transactional rings sharing cards and devices. |
| **Opportunistic Anomaly Cases** | 105 cases (30.0% of fraud) | `manifest.label_breakdown` | High-velocity point anomalies without dense ring topologies. |
| **Chronological Train Split** | 7,000 nodes (70.0%) | $t \in [0.0, 21.0]$ days | 239 frauds (3.41% prevalence) |
| **Chronological Validation Split** | 1,500 nodes (15.0%) | $t \in (21.0, 25.5]$ days | 58 frauds (3.87% prevalence) |
| **Chronological Test Split** | 1,500 nodes (15.0%) | $t \in (25.5, 30.0]$ days | 53 frauds (3.53% prevalence) |
| **Self-Loops Allowed** | False ($u \ne v$) | `manifest.metadata.allow_self_loops` | Transactions do not connect to themselves. |
| **Causal Temporal Invariant** | Strictly forward ($t_u < t_v$) | `timestamps` monotonic ordering | Prevents temporal data leakage from future transactions. |
