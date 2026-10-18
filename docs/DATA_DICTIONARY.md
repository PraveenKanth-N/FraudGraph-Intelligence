# Data Dictionary & Schema Specification — FraudGraph Intelligence (FGI)

**Module:** FGI-MOD-02 (Dataset Engineering) & FGI-MOD-03 (Graph Representation)  
**Authoritative Data Contract:** [`docs/GRAPH_DATA_CONTRACT.md`](file:///D:/FraudGraph%20Intelligence/docs/GRAPH_DATA_CONTRACT.md)  
**Status:** Phase 3 Verified Specification  

---

## 1. IEEE-CIS Financial Fraud Dataset Overview

The dataset represents real-world e-commerce transaction data provided by Vesta Corporation. To ensure high research fidelity, reproducible execution, and prevent target leakage, features are categorized into distinct functional groups.

---

## 2. Raw Transaction Features & Semantic Roles

| Column Name | Type | Description | Target Leakage Risk | Role in FraudGraph Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| `TransactionID` | Integer | Unique identifier for each transaction | None (Index) | Mapped bijectively to contiguous node index $[0, N-1]$. |
| `isFraud` | Binary (0/1) | Ground-truth fraud label | **CRITICAL LEAKAGE** | Target variable only. **Strictly quarantined** from graph features and edge construction. |
| `TransactionDT` | Float | Timedelta (seconds) from reference datetime | Low | Temporal sequencing & chronological train/test splitting. |
| `TransactionAmt` | Float | Transaction amount in USD | None | Node numerical feature (normalized / standardized). |
| `ProductCD` | Categorical | Product code (W, C, R, H, S) | None | One-hot encoded node feature. |
| `card1` | Integer | Primary payment card issuer identifier | None | **Primary Relational Key** (Temporal edges between transactions sharing `card1`). |
| `card2` - `card6`| Numeric/Cat | Bank identification, card category, card type | None | Node categorical/numerical features. |
| `addr1`, `addr2` | Numeric | Purchaser billing region and country | None | Node spatial features. |
| `dist1`, `dist2` | Float | Distance metrics (billing/shipping/IP) | None | Node features (median-imputed). |
| `P_emaildomain` | String | Purchaser email domain | None | **Secondary Relational Key** (Temporal edges across rare domains). |
| `R_emaildomain` | String | Recipient email domain | None | Node feature. |
| `C1` - `C14` | Float | Counting attributes (e.g. phone numbers associated with card) | None | Node numerical features. |
| `D1` - `D15` | Float | Timedeltas between previous transactions | None | Node numerical features. |
| `V1` - `V339` | Float | Vesta engineered rich transaction features | None | Dimensionality-reduced node feature matrix. |

---

## 3. Implemented Graph Construction & Leakage Controls

Implemented in [`backend/cpu_engine/transaction_graph_builder.py`](file:///D:/FraudGraph%20Intelligence/backend/cpu_engine/transaction_graph_builder.py):

1. **Temporal Causality ($t_u \le t_v$):**
   - Transactions are chronologically sorted by `TransactionDT` prior to index assignment.
   - For any pair of transactions $u$ and $v$ sharing a relational key (`card1` or `P_emaildomain`):
     $$\text{Edge } (u, v) \text{ created} \iff \text{TransactionDT}(u) \le \text{TransactionDT}(v) \quad \text{AND} \quad (\text{TransactionDT}(v) - \text{TransactionDT}(u)) \le \Delta t_{\text{window}}$$
   - Prevents future transactions from ever influencing historical neighborhood feature representations.
2. **Strict Null Sentinel Rejection:**
   - Transactions where a relational key is missing (`None`, `np.nan`, or empty string `""`) are **strictly excluded** from grouping. Missing identifiers NEVER connect to other missing identifiers, eliminating spurious dense cliques.
3. **Target Label Quarantine:**
   - `isFraud` is completely excised from the feature tensor $X$ and is returned as a decoupled labels array for downstream supervised evaluation.

---

## 4. Compressed Sparse Row (CSR) Representation

The in-memory and serialized graph format conforms to [`docs/GRAPH_DATA_CONTRACT.md`](file:///D:/FraudGraph%20Intelligence/docs/GRAPH_DATA_CONTRACT.md):

- `row_ptr`: 1D array of shape $(N + 1,)$ with dtype `np.int32`. `row_ptr[i]` marks the start index in `col_idx` for node $i$.
- `col_idx`: 1D array of shape $(M,)$ with dtype `np.int32` containing target neighbor indices.
- `features`: 2D array of shape $(N, D)$ with dtype `np.float32`. Guaranteed 100% finite (zero NaNs or Infs).
- `metadata`: Comprehensive JSON manifest tracking node counts, edge counts, density, degree percentiles (min, max, mean, median, p90, p99), memory footprints, and generation policies.
