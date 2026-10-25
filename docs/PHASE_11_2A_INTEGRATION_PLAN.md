# Phase 11.2A — Integration QA and API Expansion Plan

**Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Institution:** Vellore Institute of Technology, CSI4007 GPU Programming Capstone  
**Workspace Root:** `D:\FraudGraph Intelligence`  
**Expected Baseline Git HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Planning Mode:** READ-ONLY ARCHITECTURAL & IMPLEMENTATION PLAN (No code modified in this pass)  
**Deliverable Document:** `docs/PHASE_11_2A_INTEGRATION_PLAN.md`  

---

## 1. Executive Summary & Objective

In the Phase 11.1 Quality Assurance Audit (`docs/PHASE_11_1_QA_REPRODUCIBILITY_AUDIT.md`), Findings **F-02** and **F-03** documented that two major platform views (`GraphExplorerPage.tsx` and `TransactionsPage.tsx`) operate under placeholder boundaries:
- `GraphExplorerPage.tsx` displays static topology statistics with an explicit status badge (`ENDPOINT NOT IMPLEMENTED`) and disabled controls (`ZOOM`, `FIT VIEW`) because the backend lacked a dynamic subgraph extraction endpoint.
- `TransactionsPage.tsx` renders 6 grounded sample rows with an explicit badge (`REST PAGINATION DEFERRED`) because the backend lacked a paginated transaction ledger endpoint.

The objective of **Phase 11.2** is to implement both missing backend REST endpoints and connect them cleanly to the React frontend while strictly adhering to research integrity rules:
1. `GET /api/v1/graph/subgraph?node_id=...&hops=...`
2. `GET /api/v1/transactions?offset=...&limit=...&filter=...`

This document establishes the authoritative data contracts, algorithmic traversal mechanics, pagination schemas, memory/payload guardrails, and risk assessments before any implementation begins.

---

## 2. Authoritative Data Source & Provenance Analysis

### 2.1 Inspection of `data/fixtures/synthetic_fraud_10k.npz`
Direct array inspection of the canonical primary fixture confirms the following physical schema:

| Array Key | Shape | Dtype | Semantic Meaning & Research Role |
|---|:---:|:---:|---|
| `row_ptr` | `(10001,)` | `int32` | CSR row offsets. Monotonically increasing from 0 to 231,984. Index $i$ defines slice `col_idx[row_ptr[i] : row_ptr[i+1]]`. |
| `col_idx` | `(231984,)` | `int32` | Neighbor transaction indices. All entries strictly satisfy $0 \le \text{col\_idx}[p] < 10000$. |
| `features` | `(10000, 32)` | `float32` | 32 continuous, 100% finite transaction features ($F_0 \dots F_{31}$). Ground truth labels are strictly quarantined. |
| `labels` | `(10000,)` | `int32` | Binary fraud annotations ($y \in \{0, 1\}$). Exactly 350 fraud nodes (3.5% prevalence) and 9,650 licit nodes. |
| `timestamps` | `(10000,)` | `float64` | Chronological offsets in seconds $[0.0, 2592000.0]$ (30-day timeline). Strictly sorted in ascending order ($t_i \le t_{i+1}$). |
| `manifest` | `()` | `<U60617` | Embedded JSON string containing construction metadata, degree distributions, and cryptographic checksum. |

### 2.2 Capability Assessment: Zero Dataset Regeneration
- **Verdict:** **SUFFICIENT WITHOUT MODIFICATION.**  
- The canonical fixture `synthetic_fraud_10k.npz` (1.43 MB) contains 100% of the nodes, edges, timestamps, features, and fraud labels needed to power both graph extraction and transaction pagination.
- **Strict Invariant:** The canonical `.npz` file will **not** be overwritten, re-generated, or modified. All new endpoints will access it strictly read-only.

### 2.3 Relational Entity Keys (`card_id`, `device_id`) Provenance Truth
- During initial dataset generation (`backend/cpu_engine/synthetic_fraud_generator.py`), synthetic payment card IDs and device IDs were generated in memory to compute shared-entity temporal edges.
- As documented in `save_labeled_graph()`, only the resulting CSR topology (`row_ptr`, `col_idx`), `timestamps`, `features`, and `labels` were persisted to disk. Individual synthetic card/device string identifiers were discarded upon compression.
- **Research Rule Compliance:** The API will **never silently invent or fabricate** fake card or device strings. In the transaction API schema:
  - `source_card`: `None`
  - `target_device`: `None`
  - `schema_notes`: `"Relational entity strings were discarded during CSR compression; graph topology preserves entity relations via directed edges."`
  - In `TransactionsPage.tsx`, the UI will display authentic `tx_id`, `timestamp_sec`, `time_display`, `fraud_label`, `in_degree`, and feature attributes with a clear note.

---

## 3. Detailed API Specifications & Contracts

### 3.1 Subgraph Extraction Endpoint (`GET /api/v1/graph/subgraph`)

#### Purpose
Extracts an authentic $k$-hop directed sub-graph centered around a target transaction node using the verified CSR topology.

#### Query Parameters
| Parameter | Type | Required | Default | Constraints | Description |
|---|:---:|:---:|:---:|:---:|---|
| `node_id` | `int` | **Yes** | — | $0 \le \text{node\_id} < 10000$ | Root transaction node index for sub-graph extraction. |
| `hops` | `int` | No | `1` | $1 \le \text{hops} \le 3$ | Traversal depth. Clamped to maximum 3 hops to prevent exponential fanout. |
| `max_nodes` | `int` | No | `150` | $10 \le \text{max\_nodes} \le 300$ | Memory guardrail limiting total extracted vertices. |
| `direction` | `string` | No | `"incoming"` | `"incoming"` \| `"outgoing"` \| `"both"` | Traversal direction. `"incoming"` follows historical neighbor aggregation. |

#### Algorithm & Traversal Mechanics
1. **Validation:** Check `0 <= node_id < num_nodes` and `1 <= hops <= 3`. Return HTTP 400 on violation.
2. **Breadth-First Search (BFS):**
   - Initialize `visited = {node_id}`, `node_depth = {node_id: 0}`.
   - Queue: `frontier = [node_id]`.
   - For `current_hop` from 1 to `hops`:
     - Collect neighbors for all $u \in \text{frontier}$ via CSR slice `col_idx[row_ptr[u] : row_ptr[u+1]]`.
     - Add unseen neighbors to `visited` up to `max_nodes`.
     - If `len(visited) >= max_nodes`, set `is_truncated = True` and break.
3. **Edge Assembly:**
   - For every node $u \in \text{visited}$, inspect its CSR neighbors. Any neighbor $v \in \text{visited}$ forms an extracted edge $(u, v)$.
   - Strict Non-Leakage Rule: Neighbors are selected exclusively by topological connectivity in `col_idx`, **never** filtered or selected by fraud labels.
4. **Node Attributes:**
   - `id`: integer node index.
   - `depth`: distance from root ($0, 1, 2, 3$).
   - `timestamp_sec`: float timestamp $t_i$.
   - `fraud_label`: binary $y_i \in \{0, 1\}$.
   - `in_degree`: total historical neighbors in full graph.
   - `is_root`: `True` if `node_id == root`.

#### Response Schema (`SubgraphResponse`)
```json
{
  "status": "success",
  "query": {
    "root_node_id": 5407,
    "requested_hops": 2,
    "actual_hops": 2,
    "direction": "incoming",
    "max_nodes": 150
  },
  "summary": {
    "total_nodes": 150,
    "total_edges": 312,
    "fraud_node_count": 8,
    "fraud_percentage": 5.33,
    "is_truncated": true,
    "truncation_reason": "MAX_NODES_LIMIT_REACHED"
  },
  "nodes": [
    {
      "id": 5407,
      "depth": 0,
      "is_root": true,
      "timestamp_sec": 1420550.2,
      "fraud_label": 0,
      "in_degree": 252
    }
  ],
  "edges": [
    {
      "source": 5407,
      "target": 3812,
      "temporal_delta_sec": 3410.5
    }
  ],
  "provenance": {
    "fixture_name": "synthetic_fraud_10k.npz",
    "sha256": "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430",
    "causality_policy": "causal_history (t_target <= t_source)"
  }
}
```

---

### 3.2 Transaction Ledger Pagination Endpoint (`GET /api/v1/transactions`)

#### Purpose
Provides paginated, filtered forensic ledger access to the 10,000 synthetic transaction records.

#### Query Parameters
| Parameter | Type | Required | Default | Constraints | Description |
|---|:---:|:---:|:---:|:---:|---|
| `offset` | `int` | No | `0` | $\ge 0$ | Starting record index in sorted/filtered sequence. |
| `limit` | `int` | No | `50` | $1 \le \text{limit} \le 100$ | Number of records to return per page. |
| `filter` | `string` | No | `"all"` | `"all"` \| `"fraud"` \| `"licit"` | Filter by fraud classification label. |
| `sort` | `string` | No | `"asc"` | `"asc"` \| `"desc"` | Chronological timestamp sorting order. |

#### Algorithm & Filtering Mechanics
1. **Filter Index Selection:**
   - `"all"`: indices $0 \dots 9999$ (length: 10,000).
   - `"fraud"`: precomputed indices `np.where(labels == 1)[0]` (length: 350).
   - `"licit"`: precomputed indices `np.where(labels == 0)[0]` (length: 9,650).
2. **Sorting:**
   - Since canonical timestamps are strictly ascending, `"asc"` preserves natural order; `"desc"` reverses indices.
3. **Slicing:**
   - `paged_indices = indices[offset : offset + limit]`.
4. **Record Serialization:**
   - `tx_id`: integer index.
   - `timestamp_sec`: float timestamp.
   - `time_display`: deterministic string formatted as `"Day DD, HH:MM:SS"`.
   - `fraud_label`: binary $0$ or $1$.
   - `risk_tier`: `"CONFIRMED_FRAUD"` if label == 1 else `"LICIT_CLEARED"`.
   - `in_degree`: neighbor count `row_ptr[i+1] - row_ptr[i]`.
   - `feature_sample`: slice of first 4 float features `[F0, F1, F2, F3]` rounded to 4 decimals.
   - `source_card`: `None` (honestly marked).
   - `target_device`: `None` (honestly marked).

#### Response Schema (`TransactionListResponse`)
```json
{
  "status": "success",
  "pagination": {
    "total_count": 350,
    "offset": 0,
    "limit": 50,
    "returned_count": 50,
    "has_next": true,
    "has_prev": false
  },
  "items": [
    {
      "tx_id": 1,
      "timestamp_sec": 382.1,
      "time_display": "Day 01, 00:06:22",
      "fraud_label": 1,
      "risk_tier": "CONFIRMED_FRAUD",
      "feature_dim": 32,
      "in_degree": 0,
      "feature_sample": [0.6521, -0.4219, 0.1192, 0.8841],
      "source_card": null,
      "target_device": null,
      "notes": "Relational entity strings discarded in CSR; connectivity stored via graph edges."
    }
  ],
  "provenance": {
    "dataset_name": "synthetic_fraud_10k",
    "sha256": "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430",
    "total_dataset_records": 10000,
    "fraud_rate_pct": 3.5
  }
}
```

---

## 4. Memory & Performance Guardrails

### 4.1 In-Memory Caching Strategy
- To avoid disk I/O latency on every HTTP request, a thread-safe singleton cache (`_DatasetCache`) will store the loaded `synthetic_fraud_10k.npz` arrays in memory upon first access.
- **RAM Footprint:**
  - `row_ptr`: $10,001 \times 4\text{ bytes} \approx 40\text{ KB}$
  - `col_idx`: $231,984 \times 4\text{ bytes} \approx 928\text{ KB}$
  - `features`: $10,000 \times 32 \times 4\text{ bytes} \approx 1,280\text{ KB}$
  - `labels`: $10,000 \times 4\text{ bytes} \approx 40\text{ KB}$
  - `timestamps`: $10,000 \times 8\text{ bytes} \approx 80\text{ KB}$
  - **Total Memory:** $< 2.5\text{ MB RAM}$. Completely negligible on modern host workstations.
- **Mtime Invalidation:** The cache checks file modification time (`os.path.getmtime`); if the underlying file changes, it reloads automatically.

### 4.2 Strict Response Size Bounds
- **Subgraph:** Hard limit of `max_nodes = 300` and `max_edges = 1000`. Maximum JSON payload is $< 80\text{ KB}$. Prevents browser UI rendering freezes in Cytoscape/SVG.
- **Transactions:** Hard limit of `limit = 100`. Maximum JSON payload is $< 45\text{ KB}$.
- **Compute Latency:** Benchmark prototype runs demonstrated traversal and slice extraction in $< 0.1\text{ ms}$. Network round-trip latency will dominate.

---

## 5. Frontend Integration Architecture

### 5.1 Graph Explorer Page (`GraphExplorerPage.tsx`)
- **Query Controls:**
  - Quick-pick selector for representative nodes:
    - Node `#5407` (Topological Hub, degree 252).
    - Node `#1` (Isolated Fraud Node, degree 0).
    - Node `#4834` (Dense Fraud Cluster Node, degree 245).
    - Custom Node Index input ($0 \dots 9999$).
  - Hop selector: `1 Hop`, `2 Hops`, `3 Hops`.
- **Status Badges:** Replace `ENDPOINT NOT IMPLEMENTED` with `LIVE SUBGRAPH REST` and `CSR TOPOLOGY VALIDATED`.
- **Canvas Rendering:**
  - Lightweight interactive SVG / Canvas node-link diagram rendering nodes positioned via a deterministic concentric radial layout (depth 0 center, depth 1 ring 1, depth 2 ring 2).
  - Node styling: Query node highlighted in Cyan, fraud nodes highlighted in Amber/Red, licit nodes in Muted Blue.
  - Hover tooltip: Node index, timestamp, in-degree, fraud status.
  - Zoom & Fit View buttons enabled.

### 5.2 Transactions Forensic Ledger (`TransactionsPage.tsx`)
- **Dynamic Connection:**
  - Connect to `usePollingResource` or direct client fetch calling `GET /api/v1/transactions`.
  - Replace static `sampleData` with dynamic `items`.
- **Active Pagination Bar:**
  - Display: `Showing 1 - 50 of 350 transactions` (when filtered) or `of 10,000` (when all).
  - Controls: `Previous Page`, `Next Page`, Page Selector.
- **Filter Buttons:**
  - `ALL (10,000)`, `FRAUD (350)`, `LICIT (9,650)` triggers backend query with `filter=...`.
- **Status Badges:** Replace `REST PAGINATION DEFERRED` with `REST PAGINATION ACTIVE (PAGE 1)`.

---

## 6. Testing Strategy & Quality Assurance Plan

### 6.1 Backend Test Coverage (`backend/tests/test_graph_api.py` & `test_transactions_api.py`)
1. **Graph API Test Matrix:**
   - Valid 1-hop traversal on normal node, hub node, and isolated node.
   - 2-hop and 3-hop depth validation and node counting.
   - Out-of-bounds `node_id` validation (`-1`, `10000`, `"abc"` -> HTTP 400).
   - Invalid `hops` validation (`0`, `4` -> HTTP 400).
   - Max nodes truncation enforcement (`max_nodes=10` -> `is_truncated: True`).
   - Non-leakage verification: edge list is 100% identical regardless of whether labels are known.
2. **Transactions API Test Matrix:**
   - Default pagination (`offset=0, limit=50`).
   - Custom limit clamping (`limit=200` clamped to `100`).
   - Filter accuracy: `filter=fraud` returns exactly 350 total count, only label 1 records.
   - Filter accuracy: `filter=licit` returns exactly 9,650 total count, only label 0 records.
   - Negative offset / invalid limits return HTTP 400.
   - Verified that `source_card` and `target_device` are null with honest schema notes.

### 6.2 Frontend Client & Integration Tests
- Add typed callers `getSubgraph(params)` and `getTransactions(params)` in `frontend/src/services/apiClient.ts`.
- Add integration tests in `frontend/tests/api_integration.test.mjs` asserting contract conformance against the new endpoints.

---

## 7. Risk Assessment & Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|:---:|:---:|---|
| **Hub Fanout Explosion** | High | Medium | Dense hubs (degree > 200) can explode exponentially at 2-3 hops. **Mitigation:** Enforce strict BFS queue truncation at `max_nodes = 300` and `max_edges = 1000`. |
| **Silent Metric Fabrication** | Low | High | Adding card/device fields might tempt placeholder generation. **Mitigation:** Enforce strict research rule: fields not in `.npz` return `None` with schema disclosure. |
| **Label Leakage into Edges** | Zero | Critical | Graph edges must never be informed by fraud labels. **Mitigation:** Traversal reads exclusively from `row_ptr` and `col_idx`; labels array is only used to populate node presentation color. |
| **NPZ Disk Contention** | Medium | Low | Multiple clients reading 1.43 MB file simultaneously. **Mitigation:** Singleton in-memory cache initialized once at startup. |
| **CORS / Proxy Mismatches** | Low | Low | New endpoints not proxied by Vite dev server. **Mitigation:** Endpoints will be registered under existing `/api/v1` prefix (`/api/v1/graph/subgraph` and `/api/v1/transactions`), which Vite dev proxy already routes to FastAPI. |

---

## 8. Implementation Checklist & Phase Boundaries

- [ ] **Step 1: Backend Schemas & Router Creation**
  - Create `backend/app/api/v1/endpoints/graph.py`
  - Create `backend/app/api/v1/endpoints/transactions.py`
  - Mount routers in `backend/app/main.py` under `/api/v1`
- [ ] **Step 2: Backend Test Suite Implementation**
  - Implement `backend/tests/test_graph_api.py`
  - Implement `backend/tests/test_transactions_api.py`
  - Execute full backend regression suite (`pytest -v`)
- [ ] **Step 3: Frontend TypeScript Models & Client Integration**
  - Update `frontend/src/types/api.ts` with Subgraph and Transaction pagination interfaces
  - Update `frontend/src/services/apiClient.ts` and `frontend/src/services/api.ts`
- [ ] **Step 4: Frontend View Implementation**
  - Update `frontend/src/pages/GraphExplorerPage.tsx` with dynamic query panel and SVG canvas
  - Update `frontend/src/pages/TransactionsPage.tsx` with pagination controls and dynamic rows
- [ ] **Step 5: Verification & Build Verification**
  - Run `npx tsc --noEmit` and `npm run build`
  - Execute `git diff --check` and verify zero canonical artifact modifications
- [ ] **Step 6: Final Verification Report**
  - Generate `docs/PHASE_11_2_IMPLEMENTATION_REPORT.md`

---

## 9. Clarifications & Architectural Decisions for Phase 11.2B Implementation

Based on Phase 11.2B specifications and verified fixture properties:

1. **Subgraph Visualizer Component Library (Approved: Zero-Dependency Custom SVG Radial Layout):**
   - The graph visualizer in `GraphExplorerPage.tsx` will use a zero-dependency custom SVG radial layout.
   - Hop 0 (query node) is positioned at the center $(cx, cy)$.
   - Subsequent hops (hop 1, 2, 3) are distributed along concentric rings at distinct radii ($r_1, r_2, r_3$).
   - Directed edges are rendered as directed SVG lines with arrowhead markers to faithfully display CSR edge orientation.
   - Prevents external npm dependencies, preserves the cyber-telemetry design aesthetic, and guarantees instant rendering and accessibility.

2. **Transaction Card/Device ID Disclosure (Approved: Truthful Null Reporting with Schema Notes):**
   - Verified that `synthetic_fraud_10k.npz` stores only graph connectivity, continuous features, labels, and timestamps. String entity keys were intentionally discarded upon CSR matrix compilation.
   - The backend returns `source_card: null` and `target_device: null` with explicit `schema_notes`.
   - The UI displays `N/A (Omitted in CSR)` and provides an informational tooltip/callout explaining the graph compression model.

3. **Node #1 Isolation Verification:**
   - Fixture verification confirmed: `row_ptr[1] = 0` and `row_ptr[2] = 0`. Node #1 has degree 0 and exactly zero outgoing/incoming edges in CSR.
   - Node #1 is confirmed as a genuinely isolated node in the canonical fixture, making it a reliable fixture-backed test case for zero-neighbor edge cases.
   - Node #5407 is confirmed as a hub node with degree 252 (`row_ptr[5408] - row_ptr[5407] = 252`).

4. **Directed CSR Edge Semantics & Label Quarantine:**
   - In FGI CSR, `col_idx[row_ptr[u] : row_ptr[u+1]]` represents directed edges from node $u$ to target $v$.
   - No reverse edges or synthetic undirected edges will be added. Edges are strictly directed as represented in the canonical CSR.
   - Traversal logic relies strictly on `row_ptr` and `col_idx`. The `labels` array is never accessed during BFS neighbor exploration or edge collection, ensuring mathematical quarantine.

5. **Deterministic Deduplication & Truncation Transparency:**
   - Returned nodes and edges are deterministically deduplicated and ordered (nodes sorted by `(depth, id)`, edges sorted by `(source, target)`).
   - If traversal reaches `max_nodes` (default 150, max 300) or `max_edges` (default 500, max 1000), `is_truncated` is set to `true`, accompanied by `truncation_reason`, `node_cap_hit`, and `edge_cap_hit`.

6. **Deterministic Transaction Ordering & Label-Derived Totals:**
   - Natural transaction index order ($0 \dots 9999$) corresponds identically to ascending chronological order ($t_0 \le t_1 \dots \le t_{9999}$).
   - `total_count` and `filtered_count` are dynamically derived from `np.sum(labels == 1)` and `np.sum(labels == 0)` (or full length), never hardcoded.
   - Label filtering occurs prior to slicing `offset : offset + limit`.

---

## 10. Conclusion

This plan provides a secure, fully grounded architectural roadmap for Phase 11.2 without modifying any research artifacts. All repository safety rules and research provenance standards are preserved.
