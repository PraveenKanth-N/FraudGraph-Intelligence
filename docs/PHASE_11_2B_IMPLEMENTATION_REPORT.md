# Phase 11.2B — Implementation Report: Graph Subgraph and Transaction Pagination APIs

**Project:** FraudGraph Intelligence (FGI)  
**Course & Institution:** VIT Vellore — CSI4007 GPU Programming Capstone  
**Workspace:** `D:\FraudGraph Intelligence`  
**Git Baseline & Expected HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Execution Date:** October 25, 2026  
**Status:** COMPLETE — Verified & Ready for Review  

---

## 1. Executive Summary & Repository Preservation Invariants

In accordance with Phase 11.1 QA Audit findings (`F-02` and `F-03`) and Phase 11.2B requirements, dynamic graph subgraph extraction and paginated transaction ledger APIs have been implemented and integrated into the React/TypeScript frontend without compromising research provenance or repository integrity.

### 1.1 Critical Repository Preservation Verification
- **Git HEAD:** Confirmed unchanged at `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`.
- **Staging Area (`git diff --cached`):** Exactly 0 files staged (`git add` was never run).
- **Working Tree Preservation:** All 14 prior uncommitted modifications and 17 untracked deliverables were preserved intact.
- **Canonical Datasets & Experiments:** Both `data/` (`synthetic_fraud_10k.npz` SHA256: `aecafe3b...`) and `experiments/results/` (`summary_evaluation_metrics.csv`, `summary_table.csv`) were accessed strictly read-only and remain 100% unaltered.
- **No Phase 11.3 Actions:** Work stopped strictly upon completion of Phase 11.2B deliverables.

---

## 2. Updated Architectural Plan Summary (`docs/PHASE_11_2A_INTEGRATION_PLAN.md`)

Before implementation, `docs/PHASE_11_2A_INTEGRATION_PLAN.md` was updated in Section 9 with verified architectural specifications:
1. **Zero-Dependency Custom SVG Radial Layout:** Selected for `GraphExplorerPage.tsx` to preserve cyber-telemetry aesthetic, avoid package bloat, and guarantee responsiveness.
2. **Honest Relational Entity Disclosure:** Confirmed that `synthetic_fraud_10k.npz` stores only graph connectivity, continuous features, labels, and timestamps. String entity keys (`card_id`, `device_id`) were discarded upon CSR compression. The API returns `source_card: null` and `target_device: null` with explicit `schema_notes`.
3. **Node #1 Fixture Degree Verification:** Directly inspected canonical fixture `row_ptr`; confirmed `row_ptr[1] == 0` and `row_ptr[2] == 0` (degree = 0). Node #1 is verified as a genuine isolated vertex in the canonical graph.
4. **Directed CSR Orientation & Label Quarantine:** Adjacency slices `col_idx[row_ptr[u] : row_ptr[u+1]]` are traversed in directed orientation without synthetic reverse edges. Traversal reads exclusively from `row_ptr` and `col_idx`, mathematically quarantining fraud labels from graph topology.
5. **Deterministic Deduplication & Truncation Transparency:** Nodes and edges are sorted deterministically; explicit truncation metadata (`node_cap_hit`, `edge_cap_hit`, `truncation_reason`) informs the UI when caps are reached.
6. **Label-Derived Transaction Totals:** `total_count` and `filtered_count` are derived dynamically via `np.where(labels == 1)` and `np.where(labels == 0)`, never hardcoded.

---

## 3. Implemented Backend Endpoints & Data Contracts

### 3.1 Subgraph Extraction (`GET /api/v1/graph/subgraph`)
- **Module:** `backend/app/api/v1/endpoints/graph.py`
- **Mounted In:** `backend/app/main.py` at `/api/v1/graph`
- **Query Parameters:**
  - `node_id: int` (required, validated $0 \le \text{node\_id} < \text{num\_nodes}$; HTTP 400 on violation).
  - `hops: int = 1` (validated $1 \le \text{hops} \le 3$; HTTP 422 on violation).
  - `max_nodes: int = 150` (validated $1 \le \text{max\_nodes} \le 300$; HTTP 422 on violation).
  - `max_edges: int = 500` (validated $1 \le \text{max\_edges} \le 1000$; HTTP 422 on violation).
  - `npz_path: Optional[str] = None` (path-traversal protected via `_is_safe_path`).
- **Response Model (`SubgraphResponse`):**
  ```json
  {
    "status": "success",
    "query": {
      "root_node_id": 5407,
      "requested_hops": 1,
      "actual_hops": 1,
      "direction": "directed_csr",
      "max_nodes": 150,
      "max_edges": 500
    },
    "summary": {
      "total_nodes": 150,
      "total_edges": 149,
      "fraud_node_count": 8,
      "fraud_percentage": 5.33,
      "is_truncated": true,
      "node_cap_hit": true,
      "edge_cap_hit": false,
      "truncation_reason": "MAX_NODES_LIMIT_REACHED"
    },
    "nodes": [
      {
        "id": 5407,
        "depth": 0,
        "is_root": true,
        "timestamp_sec": 1420550.2,
        "fraud_label": 0,
        "is_fraud": false,
        "degree": 252
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
      "causality_policy": "causal_history (directed CSR)"
    }
  }
  ```

### 3.2 Transaction Ledger Pagination (`GET /api/v1/transactions`)
- **Module:** `backend/app/api/v1/endpoints/transactions.py`
- **Mounted In:** `backend/app/main.py` at `/api/v1/transactions`
- **Query Parameters:**
  - `offset: int = 0` (validated $\ge 0$; HTTP 422 on violation).
  - `limit: int = 50` (validated $1 \le \text{limit} \le 100$; HTTP 422 on violation).
  - `filter: str = "all"` (validated against `{"all", "fraud", "licit"}`; HTTP 400 on violation).
  - `npz_path: Optional[str] = None` (path-traversal protected via `_is_safe_path`).
- **Response Model (`TransactionListResponse`):**
  ```json
  {
    "status": "success",
    "total_dataset_records": 10000,
    "total_fraud_records": 350,
    "total_licit_records": 9650,
    "filtered_count": 350,
    "offset": 0,
    "limit": 50,
    "returned_count": 50,
    "has_next": true,
    "has_prev": false,
    "current_filter": "fraud",
    "ordering": "original_transaction_index_chronological",
    "items": [
      {
        "tx_id": 1,
        "timestamp_sec": 382.1,
        "fraud_label": 1,
        "is_fraud": true,
        "risk_tier": "CONFIRMED_FRAUD",
        "in_degree": 0,
        "feature_dim": 32,
        "feature_sample": [0.6521, -0.4219, 0.1192, 0.8841],
        "source_card": null,
        "target_device": null,
        "notes": "Relational entity strings were discarded during CSR compression; graph topology preserves entity relations via directed edges."
      }
    ],
    "schema_notes": "source_card and target_device are omitted in CSR graph representation; in_degree indicates connected transactions in historical graph.",
    "provenance_sha256": "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430"
  }
  ```

---

## 4. Frontend Integration Architecture

### 4.1 Type Definitions & Client Callers
- **`frontend/src/types/api.ts`:** Added typed interfaces for `SubgraphResponse`, `SubgraphNode`, `SubgraphEdge`, `SubgraphQueryMeta`, `SubgraphSummary`, `SubgraphProvenance`, `TransactionItem`, `TransactionListResponse`, and `TransactionFilter`.
- **`frontend/src/services/apiClient.ts`:** Implemented typed `getSubgraph(params, options)` and `getTransactions(params, options)` with automatic query-string serialization and structured error handling.
- **`frontend/src/services/api.ts`:** Re-exported `fetchSubgraph` and `fetchTransactions` for consumption by page components.

### 4.2 Graph Explorer Page (`frontend/src/pages/GraphExplorerPage.tsx`)
- **Interactive REST Querying:** Connected to `fetchSubgraph` with custom node ID input ($0 \dots 9999$), hop selectors (1, 2, 3), and max nodes caps (50, 150, 300).
- **Preset Quick-Picks:** Instant one-click selection for canonical representative nodes:
  - Hub Node `#5407` (Degree 252, dense neighbor expansion).
  - Isolated Node `#1` (Degree 0, isolated sentinel verification).
  - Fraud Node `#4834` (Degree 245, dense fraud cluster).
- **Custom SVG Radial Visualization:**
  - Concentric guide rings displaying hop depth (center = root depth 0, inner ring = depth 1, middle ring = depth 2, outer ring = depth 3).
  - Directed edges rendered with directional arrowhead markers.
  - Distinct node color coding: Cyan glow for query root, Red for confirmed frauds ($y=1$), Teal for licit transactions ($y=0$).
  - Interactive tooltip overlay on node hover showing Transaction ID, Depth, Degree, Timestamp, and Fraud status.
  - Canvas controls: `ZOOM +`, `ZOOM -`, and `FIT VIEW` (reset zoom/pan).
- **Status Badges & Banners:**
  - `LIVE SUBGRAPH REST` (primary badge).
  - `CSR TOPOLOGY VALIDATED` (success badge).
  - Truncation warning banner rendered when `summary.is_truncated === true` explaining server-side caps.
  - Error state with `RETRY` button.

### 4.3 Transactions Forensic Ledger (`frontend/src/pages/TransactionsPage.tsx`)
- **Server-Side Pagination:** Connected to `fetchTransactions` with active pagination bar (`PREVIOUS`, `NEXT`, records indicator `Showing 1 - 50 of 350 records`, page indicator `Page 1 of 7`).
- **Dynamic Filter Controls:** Interactive filter buttons displaying label-derived counts: `ALL (10,000)`, `FRAUD (350)`, `LICIT (9,650)`.
- **Page Size Selector:** Configurable page limit buttons (`25`, `50`, `100` rows per page).
- **Client Search Filter:** Filters current page items by transaction index or risk tier.
- **Honest Schema Disclosure:** Prominent callout explaining that payment card and device strings were omitted during CSR compression and that transaction connectivity is preserved via directed graph edges.
- **Status Badges:** `REST PAGINATION ACTIVE` (primary badge).

---

## 5. Verification Results & Test Suites

### 5.1 Backend Regression Suite (`pytest`)
All 178 backend tests passed:
```
backend/tests/test_aggregators.py ......................                 [ 12%]
backend/tests/test_benchmark_automation.py ...............               [ 20%]
backend/tests/test_csr_graph.py ....................                     [ 32%]
backend/tests/test_cuda_baseline.py ...........s                         [ 38%]
backend/tests/test_cuda_coalesced.py .........                           [ 43%]
backend/tests/test_cuda_degree_aware.py ..........                       [ 49%]
backend/tests/test_cuda_streams.py ........                              [ 53%]
backend/tests/test_dashboard.py ...................                      [ 64%]
backend/tests/test_feature_enrichment.py ...............s                [ 73%]
backend/tests/test_graph_api.py ........                                 [ 78%]
backend/tests/test_health.py ..                                          [ 79%]
backend/tests/test_model_evaluation.py ........                          [ 83%]
backend/tests/test_reporting.py ...........                              [ 89%]
backend/tests/test_synthetic_fraud_generator.py ...........              [ 96%]
backend/tests/test_transactions_api.py .......                           [100%]

================== 176 passed, 2 skipped in 99.28s (0:01:39) ==================
```

#### Graph API Isolated Tests (`backend/tests/test_graph_api.py`):
1. `test_isolated_node_degree_and_subgraph`: Verified node #1 degree is 0 in canonical fixture; returns exactly 1 node and 0 edges, `is_truncated=False`.
2. `test_hub_node_traversal_directed_semantics`: Verified hub #5407 (degree 252); every returned edge exists in canonical CSR slice with no synthetic reverse edges.
3. `test_subgraph_multi_hop_depth_and_actual_hops`: Verified 2-hop depth validation and `actual_hops` derivation.
4. `test_deterministic_deduplication_and_ordering`: Verified zero duplicate node IDs or edges; verified deterministic sorting.
5. `test_invalid_inputs_client_errors`: Verified 400 / 422 client errors for $node\_id < 0$, $node\_id \ge 10000$, invalid hops ($0, 4$), out-of-range caps ($0, 301$), and path escapes.
6. `test_node_and_edge_caps_with_truncation_metadata`: Verified `is_truncated=True`, `node_cap_hit=True`, and `edge_cap_hit=True` when caps are exceeded.
7. `test_label_quarantine`: Verified that mutating fraud labels in temporary fixtures produces 100% identical graph topology (nodes and edges).
8. `test_dataset_missing_and_corrupt`: Verified safe HTTP 404 for missing fixtures and HTTP 503 for malformed/empty fixtures.

#### Transactions API Isolated Tests (`backend/tests/test_transactions_api.py`):
1. `test_default_pagination_and_metadata`: Verified default offset 0, limit 50, totals (10,000 / 350 / 9,650), and schema notes.
2. `test_all_three_filters`: Verified `all`, `fraud`, and `licit` filter isolation and derived totals.
3. `test_filter_before_pagination_slicing`: Verified that label filtering precedes pagination slicing across consecutive pages.
4. `test_pagination_boundaries_and_empty_pages`: Verified boundary conditions (partial last page, exact boundary, offset past end).
5. `test_invalid_inputs_client_errors`: Verified 422 for negative offset, zero/excess limit, and 400 for unknown filter string.
6. `test_honest_entity_key_and_feature_presentation`: Verified `source_card` and `target_device` are null, notes are present, and feature samples match canonical arrays.
7. `test_dataset_missing_and_corrupt`: Verified safe HTTP 404 and 503 responses for missing or empty fixtures.

### 5.2 Frontend Test Suite (`node --test`)
All 24 frontend tests passed:
```
▶ Phase 9C.1.2 Remediation Tests (21 tests)
  ✔ F-04: Structured API Error Formatting
  ✔ F-07: Timeout vs. Abort Error Classification
  ✔ Direct apiClient Request & Typed Callers
  ✔ F-01: Polling Timer Race & Monotonic Sequencing
  ✔ F-02: Stop and Start Lifecycle Stability
  ✔ F-03: Loading and Refreshing Indicator Purity
  ✔ F-05: Page Visibility Defers Scheduling
  ✔ Phase 9C.2: Frontend Data Integration & State Consistency
▶ Phase 11.2B: Subgraph and Transactions Client Tests (3 tests)
  ✔ getSubgraph constructs the expected query string and parses response
  ✔ getTransactions constructs query string with offset, limit, filter and parses response
  ✔ handles empty query parameters on getTransactions cleanly
✔ Total: 24 passed, 0 failed (346ms)
```

### 5.3 Frontend TypeScript & Production Build Verification
- **TypeScript Static Analysis:** `npx tsc --noEmit` exited with code 0 (zero errors).
- **Vite Production Build:** `npm run build` completed cleanly:
  ```
  vite v6.4.4 building for production...
  ✓ 1621 modules transformed.
  dist/index.html                   0.87 kB │ gzip:  0.51 kB
  dist/assets/index-ChKqFaYm.css    1.89 kB │ gzip:  0.87 kB
  dist/assets/index-Xyoe1w84.js   294.96 kB │ gzip: 85.02 kB
  ✓ built in 2.73s
  ```

### 5.4 Test Methodology & Environment Scope
- **Automated In-Memory Tests (Executed):** All backend tests run against FastAPI `TestClient(app)` with direct memory array validation; frontend client tests run under Node.js native test runner (`node --test`).
- **Browser E2E Execution (Distinction):** Real browser interaction with live Vite dev server proxying to a standalone running Uvicorn daemon was audited in Phase 9D.1; in Phase 11.2B, full headless TypeScript validation and bundle compilation were verified without starting conflicting external daemons.

---

## 6. Changed Files & Repository Delta Summary

### New Files Created (4):
1. `backend/app/api/v1/endpoints/graph.py` — Subgraph extraction REST endpoint.
2. `backend/app/api/v1/endpoints/transactions.py` — Transaction ledger pagination REST endpoint.
3. `backend/tests/test_graph_api.py` — 8 isolated test cases for subgraph extraction.
4. `backend/tests/test_transactions_api.py` — 7 isolated test cases for transaction pagination.
5. `frontend/tests/subgraph_transactions_client.test.mjs` — 3 frontend client contract tests.
6. `docs/PHASE_11_2B_IMPLEMENTATION_REPORT.md` — This verification report.

### Modified Existing Files (6):
1. `docs/PHASE_11_2A_INTEGRATION_PLAN.md` — Updated with Section 9 implementation decisions.
2. `backend/app/main.py` — Mounted `graph.router` and `transactions.router`.
3. `frontend/src/types/api.ts` — Added Subgraph and Transaction response interfaces.
4. `frontend/src/services/apiClient.ts` — Added `getSubgraph` and `getTransactions`.
5. `frontend/src/services/api.ts` — Re-exported `fetchSubgraph` and `fetchTransactions`.
6. `frontend/src/pages/GraphExplorerPage.tsx` — Integrated live SVG concentric radial graph visualization.
7. `frontend/src/pages/TransactionsPage.tsx` — Integrated live server-side pagination and label filters.

### Unchanged Protected Files:
- All canonical dataset fixtures under `data/fixtures/*.npz`.
- All experiment evaluation and benchmark artifacts under `experiments/results/`.
- Git HEAD: strictly preserved at `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`.
- Git Staging: zero files staged.

---

## 7. Known Limitations & Recommendations for Subsequent Phases

1. **Transaction Card & Device Identifiers:** As documented, string entity identifiers were discarded during CSR matrix compression. The API and UI truthfully report `null` / `N/A (Omitted in CSR)`. If human-readable card/device names are ever desired for non-graph views in future phases, a separate metadata sidecar mapping file would need to be generated without modifying the canonical `.npz`.
2. **WebGL Large Graph Rendering:** The custom SVG radial layout provides rendering for up to 300 nodes. If future phases require rendering $> 1,000$ nodes simultaneously, an external WebGL library (e.g., Cytoscape.js with Canvas/WebGL renderer) may be evaluated.
3. **Phase Boundary:** Phase 11.2B implementation is complete. No Phase 11.3 work has been initiated. Awaiting review.
