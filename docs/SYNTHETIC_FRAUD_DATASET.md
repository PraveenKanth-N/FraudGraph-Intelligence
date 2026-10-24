# Synthetic Labeled Fraud Transaction Graph Specification (`synthetic_fraud_10k`)

**Project:** FraudGraph Intelligence (FGI)
**Module:** FGI-MOD-07 (Fraud Scoring & Classifier Baseline)
**Course:** VIT Vellore — CSI4007 GPU Programming
**Authoritative Source:** [`backend/cpu_engine/synthetic_fraud_generator.py`](file:///D:/FraudGraph%20Intelligence/backend/cpu_engine/synthetic_fraud_generator.py)
**Binary Fixture:** [`data/fixtures/synthetic_fraud_10k.npz`](file:///D:/FraudGraph%20Intelligence/data/fixtures/synthetic_fraud_10k.npz)
**Status:** Phase 7.2 Complete & Empirically Verified

---

## 1. Research Integrity & Dataset Provenance Notice

> [!IMPORTANT]
> **SYNTHETIC BENCHMARK DISCLOSURE:**
> This dataset is a **100% synthetic, deterministic benchmark fixture** designed exclusively for controlled empirical evaluation of graph-based neighbor aggregation algorithms in financial fraud classification.
>
> - The **3.50% fraud prevalence** (exactly 350 positive fraud labels out of 10,000 transactions) is a controlled design choice inspired by real-world class-imbalance rates observed in public benchmarks (e.g., IEEE-CIS Fraud Detection).
> - This fixture **does NOT claim to represent the true distribution, adversarial sophistication, or empirical reality of live banking fraud**.
> - Classification performance evaluated on this fixture represents micro-benchmarking of relational feature enrichment, not production fraud detection effectiveness.

---

## 2. Synthetic Data-Generating Process (DGP)

The dataset is generated via a mathematically formalized 5-stage pipeline:

```
[1. Poisson Arrival Process] -> Timestamps t_i in [0, 2.592e6 s] (Monotonically non-decreasing)
            |
[2. Zipfian Entity Allocation] -> card_id in [0, 2499], device_id in [0, 2999] (Power-law frequency)
            |
[3. Ground-Truth Allocation] -> Exactly 350 Frauds (245 ring frauds on compromised entities, 105 opportunistic)
            |
[4. Feature Matrix Synthesis] -> 32 continuous dimensions (Licit ~ N(0, 1); Fraud has subtle multivariate shifts)
            |
[5. Temporal Edge Construction] -> Shared entity within 7-day lookback; directed from query node i to historical neighbor j (t_j < t_i)
```

### 2.1 Stage 1: Chronological Timestamps
- Simulated across a **30-day temporal observation window** ($T_{\max} = 30 \times 86,400 = 2,592,000.0\text{ seconds}$).
- Generated from a Poisson arrival process using exponential inter-arrival intervals:
  $$\Delta t_i \sim \text{Exp}\left(\frac{T_{\max}}{N}\right), \quad t_i = \sum_{k=0}^i \Delta t_k$$
- Timestamps are strictly non-decreasing ($t_0 \le t_1 \le \dots \le t_{N-1}$), matching sequential transaction arrival logs in core banking engines.

### 2.2 Stage 2: Entity Identifier Distribution
- Simulates $2,500$ distinct payment cards (`card_id`) and $3,000$ distinct merchant/terminal endpoints (`device_id`).
- Follows an empirical power-law / Zipfian frequency distribution:
  $$P(\text{entity} = k) \propto k^{-0.8}$$
  where a small subset of cards and terminals participate in frequent transactions, while the majority exhibit low activity.

### 2.3 Stage 3: Fraud Allocation & Relational Topologies
Out of $10,000$ transactions, exactly **$350$ nodes ($3.50\%$)** are assigned positive labels ($y_i = 1$):
1. **Organized Fraud Rings ($70\%$ of frauds = $245$ transactions):**
   - $20$ entity pairs (`card_id`, `device_id`) are designated as compromised clusters.
   - Fraudulent transactions are allocated to these compromised clusters, simulating coordinated card-testing rings and multi-transaction account drain attacks that share infrastructure.
2. **Opportunistic Standalone Frauds ($30\%$ of frauds = $105$ transactions):**
   - Uniformly distributed across remaining transactions to represent isolated, sporadic fraud events.

### 2.4 Stage 4: Continuous Feature Generation & Non-Trivial Separation
To avoid making the classification task trivially solvable by a single decision stump:
- **Base Distribution:** Licit transactions are drawn from standard Gaussian $\mathcal{N}(0, I_{32})$.
- **Subtle Multivariate Shift:** Fraudulent transactions exhibit subtle, overlapping shifts across distinct functional feature groups:
  - *Features 0–3 (Amount / Velocity proxies):* Mean shift $+0.55$, standard deviation $0.60$.
  - *Features 4–7 (Consistency / Risk proxies):* Mean shift $-0.45$, standard deviation $0.50$.
  - *Features 8–31 (Noise / Static descriptors):* Identical $\mathcal{N}(0, 1)$ distribution (zero separation).
- **Target Leakage Protection:**
  - Ground truth labels are stored in an independent array and are **never** included in the feature tensor $X$.
  - Maximum absolute Pearson correlation between any individual feature column and $y$ is strictly bounded: $\max_k |r(X_k, y)| \approx 0.120 < 0.35$.
  - 100% of feature values are finite (zero NaNs, zero Infs).

### 2.5 Stage 5: Temporal Graph Construction & Causality Protection
- **Edge Semantics:** In the CSR row-pointer representation, an edge $(i, j)$ indicates that **query transaction $i$ points to historical context neighbor $j$** ($t_j < t_i$, $j < i$). Neighbor aggregation at node $i$ gathers context strictly from prior transactions.
- **Edge Formation Rule:** Two transactions $i$ and $j$ form a directed CSR edge $(i, j)$ if:
  1. They share a relational entity (`card_id` or `device_id`).
  2. Transaction $j$ occurred strictly prior to transaction $i$: $t_j < t_i$ ($j < i$).
  3. The time difference is within the sliding lookback window: $(t_i - t_j) \le \Delta t_{\text{window}} = 7 \text{ days}$ ($604,800.0\text{ seconds}$).
- **Zero Future Leakage:** For 100% of the $231,984$ CSR edges in the canonical fixture, $t_j < t_i$ holds strictly ($0$ future edges, $0$ same-timestamp edges, $0$ self-loops). No transaction can ever gather feature mass from future events.

---

## 3. Archive & Array Schema (`synthetic_fraud_10k.npz`)

The serialized archive [`data/fixtures/synthetic_fraud_10k.npz`](file:///D:/FraudGraph%20Intelligence/data/fixtures/synthetic_fraud_10k.npz) adheres to the FGI data contract:

| Array Key | Data Type | Shape | Meaning & Semantics |
| :--- | :--- | :--- | :--- |
| **`row_ptr`** | `np.int32` | $(10,001,)$ | CSR row offsets. `row_ptr[0] == 0`, `row_ptr[N] == M`. |
| **`col_idx`** | `np.int32` | $(231,984,)$ | Historical neighbor indices. $0 \le \text{col\_idx}[p] < N$. |
| **`features`**| `np.float32` | $(10,000, 32)$ | Dense feature matrix $X$. Guaranteed 100% finite. |
| **`labels`** | `np.int32` | $(10,000,)$ | Ground truth fraud indicators: $0 = \text{licit}$, $1 = \text{fraud}$. |
| **`timestamps`**| `np.float64` | $(10,000,)$ | Chronological transaction timestamps (seconds from $0.0$). |
| **`manifest`** | String | Scalar JSON | Embedded manifest documenting provenance, schemas, and metrics. |

### Archive Backward Compatibility:
- Can be loaded as a standard CSR graph via:
  ```python
  from backend.cpu_engine.csr_graph import CSRGraph
  graph = CSRGraph.load("data/fixtures/synthetic_fraud_10k.npz")
  ```
- Or with complete label and temporal metadata via:
  ```python
  from backend.cpu_engine.synthetic_fraud_generator import load_labeled_graph
  graph, labels, timestamps = load_labeled_graph("data/fixtures/synthetic_fraud_10k.npz")
  ```

---

## 4. Graph Topological Profile

| Topological Metric | Measured Value | Analysis & Significance |
| :--- | :--- | :--- |
| **Node Count ($N$)** | $10,000$ | Standard FGI Tier A transaction scale. |
| **Directed Edges ($M$)** | $231,984$ | Total historical lookback connections. |
| **Graph Density** | $0.00232$ | Realistic sparse transaction network. |
| **Isolated Nodes ($d_i = 0$)** | $1,155$ ($11.55\%$) | Transactions with unique cards/devices or first-time users. |
| **Minimum Degree** | $0$ | Correctly handled by FGI $\vec{0}$ zero sentinels. |
| **Maximum Degree** | $252$ | Frequent card/terminal entities. |
| **Mean Degree ($\bar{d}$)** | $23.20$ | Average historical context per transaction. |
| **Median Degree (p50)** | $6.0$ | Typical cardholder repetition. |
| **90th Percentile Degree (p90)** | $71.0$ | Heavy activity terminals. |
| **99th Percentile Degree (p99)** | $143.0$ | Concentrated corporate or testing endpoints. |
| **Memory Footprint** | $2.1438\text{ MB}$ | Fits comfortably in CPU cache and GPU VRAM. |

---

## 5. Evaluation Protocol & Chronological Split

To evaluate tabular classifiers against graph-enriched models without transductive snooping:

### 5.1 Chronological Split Boundaries
Because transaction nodes are strictly sorted chronologically by timestamp, continuous index slicing enforces temporal isolation:
- **Train Set ($70\%$):** Nodes $0 \dots 6,999$ ($239$ frauds, $3.41\%$ fraud rate).
- **Validation Set ($15\%$):** Nodes $7,000 \dots 8,499$ ($58$ frauds, $3.87\%$ fraud rate).
- **Test Set ($15\%$):** Nodes $8,500 \dots 9,999$ ($53$ frauds, $3.53\%$ fraud rate).
- Total Frauds: $239 + 58 + 53 = 350$ ($3.50\%$).

Implemented in [`backend/cpu_engine/synthetic_fraud_generator.py`](file:///D:/FraudGraph%20Intelligence/backend/cpu_engine/synthetic_fraud_generator.py#L367-L395):
```python
train_idx, val_idx, test_idx = temporal_train_test_split(num_nodes=10000, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
```

### 5.2 Research Limitations on Graph Leakage
1. **Structural Inductive Separation:**
   - In dynamic inductive deployments, test transactions aggregate historical context from prior training transactions ($t_{\text{train}} \le t_{\text{test}}$). This reflects real-world inference where past transaction histories are stored in production feature stores.
   - However, training transactions must **never** aggregate from validation or test transactions ($t_{\text{test}} > t_{\text{train}}$). Because $100\%$ of edges satisfy $j < i$, training nodes have exactly zero edges into validation or test partitions.
2. **Homophily vs. Spurious Correlation:**
   - Because fraudulent transactions in fraud rings share compromised cards and devices, neighborhood aggregation naturally propagates risk signals.
   - In Phase 7.4, evaluation will explicitly contrast raw tabular features ($X$) against graph-enriched features ($[X \,\|\, Y_{\text{mean}}]$) to measure the exact marginal contribution ($\Delta \text{PR-AUC}$) of neighborhood context.

### 5.3 Preliminary Tabular Classifier Baseline (Synthetic Data Verification)
To verify that the synthetic fixture provides a non-trivial benchmark (exceeding random guessing without trivial separability) prior to Phase 7.3 graph enrichment, standard scikit-learn models were evaluated on the canonical split with `StandardScaler` fitted strictly on `X_train` (zero transductive leakage):
- **Tabular Logistic Regression ($C=1.0$):** Test ROC-AUC = $0.7855$, Test PR-AUC = $0.3088$ (minority prevalence $\pi = 0.0353$).
- **Tabular Random Forest ($100$ trees):** Test ROC-AUC = $0.7786$, Test PR-AUC = $0.2445$.

> [!CAUTION]
> **Research Integrity Notice:** These scores are preliminary verification metrics measured strictly on the synthetic benchmark fixture (`synthetic_fraud_10k.npz`). They do not represent real-world fraud detection accuracy, nor should earlier unverified prototype estimates be cited. They establish the non-trivial tabular baseline against which Phase 7.3/7.4 graph feature aggregation gains ($\Delta \text{PR-AUC}$) will be evaluated.
