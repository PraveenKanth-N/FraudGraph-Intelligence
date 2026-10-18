# Dataset Evaluation & Selection Strategy — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Document Status:** Phase 2 Authoritative Specification  

---

## 1. Candidate Dataset Comparative Evaluation

To select the most defensible dataset foundation for FraudGraph Intelligence, three candidate datasets were evaluated against our research requirements:
1. **IEEE-CIS Fraud Detection Benchmark** (Tabular E-commerce Transactions).
2. **Elliptic Data Set** (Native Bitcoin Financial Forensics Graph).
3. **Controlled Synthetic Graph Benchmarking Fixture** (R-MAT / Power-Law Generator).

| Criterion | Candidate 1: IEEE-CIS Fraud Detection | Candidate 2: Elliptic Data Set | Candidate 3: Synthetic Power-Law Fixture |
| :--- | :--- | :--- | :--- |
| **Official Source** | Vesta Corp / Kaggle Competition | MIT-IBM Watson AI Lab / Kaggle | Self-Contained NetworkX / NumPy Generator |
| **License & Access** | Open Competition Research License (Free, requires Kaggle authentication or direct archive). | Open Academic License (Free public download, CC BY-NC 4.0). | MIT License / Public Domain (Native generation). |
| **Record & Feature Scale** | 590,540 train transactions; 394 transaction & identity features. | 203,769 transaction nodes; 234,355 directed payment edges; 166 features. | Arbitrary configurable ($10^3$ to $10^6$ nodes; $D \in [16, 256]$). |
| **Fraud Class Imbalance** | ~3.5% positive fraud rate (`isFraud = 1`). Severe real-world imbalance. | 4,545 illicit (2.2%), 42,019 licit (20.6%), 157,205 unlabeled (77.1%). | Configurable synthetic anomaly rate (e.g., 2% to 5%). |
| **Native Graph Structure** | **No native graph.** Tabular transactions with relational identifier keys (`card1`, `P_emaildomain`, `DeviceInfo`). | **Native directed graph.** Edges represent verified on-chain Bitcoin transaction output-to-input payment flows. | **Controlled synthetic graph.** Configurable degree distribution (power-law vs. uniform). |
| **Target Leakage Risk** | **Moderate to High.** Edge construction must be strictly isolated from future timestamps and `isFraud` labels. | **Low.** Edges represent actual historical blockchain transactions organized across 49 distinct time intervals. | **Zero.** Synthetic ground truth generated under controlled parameters. |
| **Primary Utility for FGI** | Ideal for **downstream fraud evaluation** on real tabular financial data. | Ideal for **native graph forensics** and validating real-world message passing. | Ideal for **controlled CUDA kernel micro-benchmarking** and numerical tolerance testing. |

---

## 2. Deep Dive: IEEE-CIS Fraud Detection Dataset

### 2.1 Available Fields & Statistical Properties
- **Transactions Table:** `TransactionID` (unique key), `isFraud` (binary label), `TransactionDT` (timedelta in seconds), `TransactionAmt` (amount USD), `ProductCD`, `card1`–`card6` (card details), `addr1`/`addr2` (geographic codes), `dist1`/`dist2` (billing/shipping distance), `P_emaildomain` / `R_emaildomain`, `C1`–`C14` (counting metrics), `D1`–`D15` (time deltas), `V1`–`V339` (rich Vesta engineered features).
- **Identity Table:** `DeviceInfo`, `DeviceType`, `id_01`–`id_38` (network / browser fingerprints).
- **Class Imbalance:** 20,663 fraud transactions out of 590,540 (~3.50%).

### 2.2 Defensible Graph Construction Rules
Because IEEE-CIS is tabular, a transaction graph must be derived from justified relational keys without inventing unverified connections:
1. **Rule 1 — Payment Instrument Co-Occurrence (Card Edges):**
   - Two transactions $u$ and $v$ share an undirected edge if and only if:
     $$\text{card1}(u) = \text{card1}(v) \quad \text{AND} \quad |\text{TransactionDT}(u) - \text{TransactionDT}(v)| \le \Delta t_{\max}$$
   - *Justification:* In financial payments, multiple transactions utilizing the exact same payment card entity within a short temporal window (e.g., $\Delta t_{\max} = 7 \text{ days}$) represent the transaction history of that payment vehicle. This captures card-testing, rapid-fire dissipation, and multi-merchant card attacks.
2. **Rule 2 — Device / Identity Co-Occurrence (Device Edges):**
   - Two transactions $u$ and $v$ share an edge if they share an identical `DeviceInfo` and `P_emaildomain`.
   - *Justification:* Fraud rings frequently execute attacks across multiple compromised cards using a single physical terminal or virtual machine.

### 2.3 Target Leakage Prevention Protocol
1. **Quarantine `isFraud`:** The target column `isFraud` is completely omitted from the node feature tensor $X$ and is never used in edge construction.
2. **Temporal Splitting:** Data is partitioned chronologically based on `TransactionDT`:
   - Training graph: First $75\%$ of transactions temporally ($t \le t_{\text{split}}$).
   - Validation/Test graph: Subsequent $25\%$ of transactions ($t > t_{\text{split}}$).
   - In dynamic inductive evaluation, test nodes may aggregate from prior training nodes, but edges must never flow backwards from future to past.

---

## 3. Deep Dive: Elliptic Data Set

### 3.1 Properties & Advantages
- Contributed by MIT-IBM Watson AI Lab (Weber et al., 2019).
- **True Graph Topology:** 203,769 transactions and 234,355 directed edges.
- **Node Features:** 166 features per node (94 local features like fees and output counts, plus 72 pre-aggregated neighborhood features).
- **Temporal Steps:** Discretized into 49 distinct time intervals (each representing a 2-week window).

### 3.2 Limitations for FGI
- The dataset is pre-anonymized; feature names are obfuscated into numerical indices without explicit financial semantics.
- Graph edges represent directed payment flows in a cryptocurrency DAG (Directed Acyclic Graph), which differs from bipartite e-commerce merchant-card transaction topologies.

---

## 4. Final Recommendation & Hybrid Strategy

To ensure absolute experimental validity and maintain high engineering defensibility, FraudGraph Intelligence adopts a **Dual-Dataset Strategy**:

```
+----------------------------------------------------------------------------------------------------+
|                                    DUAL-DATASET STRATEGY                                           |
+--------------------------------------------------+-------------------------------------------------+
|                    TIER 1                        |                     TIER 2                      |
|         CONTROLLED SYNTHETIC FIXTURES            |         EMPIRICAL BENCHMARK SUBSET              |
|        (CUDA Correctness & Scalability)          |         (Downstream Fraud Evaluation)           |
+--------------------------------------------------+-------------------------------------------------+
| - Deterministic R-MAT / Barabási-Albert graphs   | - Managed 50,000-node temporal subset of        |
| - Exact node/edge counts ($10^4$ to $2\cdot 10^5$)|   IEEE-CIS Fraud Detection.                     |
| - Controlled degree skew ($\gamma \in [2.0, 3.0]$| - Grounded in card1/device relational rules.    |
| - Controlled feature dimensions ($D \in 16..128$) | - Imbalanced fraud classification benchmark     |
| - Zero download requirement for CI/CD unit tests |   (F1-score, PR-AUC, Confusion Matrix).         |
| - Mathematical verification against CPU reference| - Demonstrates real-world utility of aggregated |
|   with exact floating-point tolerances.          |   neighbor embeddings.                          |
+--------------------------------------------------+-------------------------------------------------+
```

### Action for Phase 2:
In compliance with Phase 2 rules, **no large datasets will be downloaded or processed in this phase**. The dataset schema, leakage controls, and graph construction rules are fully specified and locked for implementation in Phase 3.
