# Data Dictionary & Schema Specification — FraudGraph Intelligence (FGI)

## 1. IEEE-CIS Financial Fraud Dataset Overview

The dataset represents real-world e-commerce transaction data provided by Vesta Corporation. To ensure high research fidelity, reproducible execution, and prevent target leakage, features are categorized into distinct functional groups.

---

## 2. Raw Transaction Features & Semantic Roles

| Column Name | Type | Description | Target Leakage Risk | Role in FraudGraph |
| :--- | :--- | :--- | :--- | :--- |
| `TransactionID` | Integer | Unique identifier for each transaction | None (Index) | Node ID |
| `isFraud` | Binary (0/1) | Ground-truth fraud label | **CRITICAL** | Target variable only. **Strictly excluded** from graph features and edge construction. |
| `TransactionDT` | Float | Timedelta from reference datetime | Low | Temporal sequencing & train/test splitting |
| `TransactionAmt` | Float | Transaction amount in USD | None | Node numerical feature (normalized) |
| `ProductCD` | Categorical | Product code (W, C, R, H, S) | None | One-hot encoded node feature |
| `card1` | Integer | Payment card payment identifier | None | **Primary Relational Key** (Edges constructed between transactions sharing `card1`) |
| `card2` - `card6`| Numeric/Cat | Bank identification, card category, card type | None | Node categorical/numerical features |
| `addr1`, `addr2` | Numeric | Purchaser billing region and country | None | Node features |
| `dist1`, `dist2` | Float | Distance metrics (billing/shipping/IP) | None | Node features (imputed median) |
| `P_emaildomain` | String | Purchaser email domain | None | **Secondary Relational Key** (Entities sharing rare domains) |
| `R_emaildomain` | String | Recipient email domain | None | Node feature |
| `C1` - `C14` | Float | Counting attributes (e.g. phone numbers associated with card) | None | Node numerical features |
| `D1` - `D15` | Float | Timedeltas between previous transactions | None | Node numerical features |
| `V1` - `V339` | Float | Vesta engineered rich transaction features | None | Dimensionality-reduced node feature matrix |

---

## 3. Graph Relational Construction Rules

In strict adherence to Section 11 of the Master Directive:
1. **No Target Leakage:** `isFraud` is NEVER used in node feature creation or edge assignment.
2. **Justified Edge Topology:** Edges represent verified structural relationships:
   - **Card Co-Occurrence:** Two transactions $u$ and $v$ are connected if $card1(u) = card1(v)$ within a sliding temporal window of $\Delta t \le 7$ days.
   - **Identity Co-Occurrence:** Two transactions $u$ and $v$ are connected if they share an identical device fingerprint or user IP subnet.
3. **Graph Directionality:** Adjacency is stored symmetrically (undirected) or directed along the timeline of transactions ($t_u < t_v$) depending on the experiment configuration.

---

## 4. Compressed Sparse Row (CSR) Layout

The graph topology is stored in three compact contiguous arrays:

```
Nodes (N): 5  |  Edges (M): 6
Adjacency:
  0 -> [1, 2]
  1 -> [0, 2, 3]
  2 -> [0, 1]
  3 -> [1]
  4 -> [] (Isolated Node)

row_ptr (Length N + 1): [0,  2,  5,  7,  8,  8]
col_idx (Length M):     [1, 2,  0, 2, 3,  0, 1,  1]
```

- `row_ptr`: Array of length $N + 1$ (int32 or int64) indicating the starting index in `col_idx` for each node's neighbor list.
- `col_idx`: Contiguous array of length $M$ (int32 or int64) containing neighbor node IDs.
- `node_features`: 2D contiguous array of shape $(N, D)$ in row-major order (float32).
