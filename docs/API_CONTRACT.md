# API Contract & Endpoint Specification â€” FraudGraph Intelligence (FGI)

**Base URL:** `http://127.0.0.1:8000/api/v1`
**Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
**OpenAPI Spec JSON:** `http://127.0.0.1:8000/openapi.json`
**Framework:** FastAPI 0.115+ / Python 3.14.3 (AMD64)
**Contract Version:** Phase 11.4 Production Specification

---

## 1. System & Telemetry Endpoints

### `GET /api/v1/health`
Returns system diagnostics, runtime health, and host GPU detection telemetry.

- **Method:** `GET`
- **Route:** `/api/v1/health`
- **Query Parameters:** None
- **Response Model:** JSON Object (unbounded dictionary)

#### Truthful Hardware Fallback Specification
- When an NVIDIA GPU is physically detected via `nvidia-smi`:
  - `detected`: `true`
  - `local_gpu_detected`: Genuine device name string (e.g., `"NVIDIA GeForce RTX 3080"`)
  - `vram_free_mb`: Numerical megabytes of unallocated VRAM (e.g., `8192.0`)
  - `cuda_compiler_status`: `"AVAILABLE_LOCAL"` if local `nvcc` compiler is located on system `PATH`, otherwise `"UNAVAILABLE_LOCAL_COLAB_READY"`
- When no physical GPU is present, or `nvidia-smi` is missing, times out, or fails:
  - `detected`: `false`
  - `local_gpu_detected`: `null` (zero fabricated hardware identifiers)
  - `vram_free_mb`: `null`
  - `cuda_available_locally`: `false`
  - `cuda_compiler_status`: `"NO_GPU_DETECTED_COLAB_READY"`

#### Response Example (No Local GPU Detected) (200 OK)
```json
{
  "status": "healthy",
  "project": "FraudGraph Intelligence",
  "project_id": "FGI",
  "version": "1.0.0",
  "compute_engine": {
    "backend": "cpu_reference",
    "cuda_available_locally": false,
    "local_gpu_detected": null,
    "vram_free_mb": null,
    "cuda_compiler_status": "NO_GPU_DETECTED_COLAB_READY",
    "colab_workflow_enabled": true
  },
  "system": {
    "python_version": "3.14.3",
    "timestamp": "2026-10-25T11:45:00.123456"
  }
}
```

---

## 2. Dashboard Analytics Endpoints

### `GET /api/v1/dashboard/summary`
Returns consolidated platform metrics, research evaluation statistics, and benchmark acceleration summaries derived strictly from genuine repository artifacts.

- **Method:** `GET`
- **Route:** `/api/v1/dashboard/summary`
- **Query Parameters:**
  - `eval_csv_path` *(string, optional)*: Explicit filesystem path to model evaluation CSV. Must reside within workspace boundaries or system temp directory.
  - `bench_csv_path` *(string, optional)*: Explicit filesystem path to benchmark summary CSV. Must reside within workspace boundaries or system temp directory.
- **Overall Platform Status:**
  - `"operational"`: Both evaluation and benchmark artifacts exist and are parseable (`artifact_state == "AVAILABLE"`).
  - `"degraded"`: One or both artifacts are missing, corrupt, or invalid.
- **Artifact Tri-State Enum (`artifact_state`):**
  - `"AVAILABLE"`: Genuine CSV artifact successfully located, schema validated, and parsed.
  - `"MISSING"`: File not found at the canonical path or designated override.
  - `"INVALID"`: File exists but is empty, lacks required columns, contains non-numeric/infinite metric values, or lacks target model/workload rows.

#### Response Model (`DashboardSummaryResponse`)
```json
{
  "status": "operational",
  "project": "FraudGraph Intelligence",
  "dataset": {
    "name": "synthetic_fraud_10k",
    "file_name": "synthetic_fraud_10k.npz",
    "num_nodes": 10000,
    "num_edges": 231984,
    "feature_dim": 32,
    "fraud_count": 350,
    "fraud_rate_pct": 3.5,
    "sha256": "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430"
  },
  "evaluation": {
    "artifact_state": "AVAILABLE",
    "experiment_id": "exp_phase_7_5_causal_frozen",
    "phase": "Phase 7.5",
    "top_model": "Logistic Regression (max)",
    "top_test_pr_auc": 0.3315,
    "top_test_roc_auc": 0.7865,
    "top_test_f1": 0.3265,
    "primary_policy": "causal_history",
    "sensitivity_policy": "train_only_context",
    "num_models_evaluated": 18,
    "num_representations": 6,
    "adr_007_notice": "Primary policy causal_history enforces strict temporal precedence (t_edge < t_target).",
    "error_message": null
  },
  "benchmarks": {
    "artifact_state": "AVAILABLE",
    "session_id": "session_20261024_230008_smoke",
    "phase": "Phase 8",
    "total_runs": 36,
    "measured_cpu_runs": 12,
    "skipped_gpu_runs": 24,
    "top_speedup_vs_cpu_ref": 44.6,
    "top_speedup_workload": "synthetic_tier_a_10k_mean (cpu_vectorized)",
    "local_gpu_status": "SKIPPED_CUDA_UNAVAILABLE",
    "high_variability_label": "MEAN vectorized high variability documented (CV 34.29%)",
    "error_message": null
  },
  "hardware": {
    "local_gpu_detected": null,
    "local_compiler_status": "NO_GPU_DETECTED_COLAB_READY",
    "local_vram_free_mb": null,
    "canonical_colab_gpu": "NVIDIA Tesla T4 (Phase 6 Archive)",
    "colab_kernel_latency_ms": 0.0679
  },
  "timestamp": "2026-10-25T11:45:00.123456"
}
```

#### Error Responses
- **400 Bad Request:** Path traversal attempt outside authorized workspace (`"eval_csv_path escapes authorized workspace boundaries"`).

---

## 3. Dataset Registry Endpoints

### `GET /api/v1/datasets`
Lists authentic Compressed Sparse Row (CSR) graph dataset fixtures verified on the filesystem.

- **Method:** `GET`
- **Route:** `/api/v1/datasets`
- **Query Parameters:**
  - `base_dir` *(string, optional)*: Override base directory for fixtures (used in test mocks).
- **Response Model (`DatasetListResponse`):**
  - `status`: `"success"`
  - `total_count`: Total cataloged fixtures (4 fixtures).
  - `fixtures`: Array of `DatasetFixtureItem`.

#### Fixture Schema (`DatasetFixtureItem`)
- `id` *(string)*: Unique identifier (e.g., `"synthetic_fraud_10k"`).
- `name` *(string)*: Human-readable dataset title.
- `file_name` *(string)*: Physical filename in `data/fixtures/`.
- `num_nodes` *(int)*: Total vertices ($N$). Derived from `len(row_ptr) - 1`.
- `num_edges` *(int)*: Directed edges ($M$). Derived from `len(col_idx)`.
- `feature_dim` *(int)*: Feature dimensionality ($D$).
- `size_bytes` *(int)*: Exact physical file size on disk.
- `is_synthetic` *(bool)*: Boolean flag denoting synthetic generation.
- `description` *(string)*: Graph characteristics and topological parameters.
- `sha256` *(string)*: Hexadecimal SHA-256 cryptographic checksum.
- `is_available` *(bool)*: True if present and parseable as valid NPZ; False if missing or corrupt.

#### Response Example (200 OK)
```json
{
  "status": "success",
  "total_count": 4,
  "fixtures": [
    {
      "id": "synthetic_fraud_10k",
      "name": "Synthetic Temporal Fraud Graph (10k)",
      "file_name": "synthetic_fraud_10k.npz",
      "num_nodes": 10000,
      "num_edges": 231984,
      "feature_dim": 32,
      "size_bytes": 1478292,
      "is_synthetic": true,
      "description": "Temporal transaction graph with 3.5% fraud prevalence (350 frauds), card/device relational keys, and 30-day timeline.",
      "sha256": "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430",
      "is_available": true
    }
  ]
}
```

---

## 4. Graph Analytics Endpoints

### `GET /api/v1/graph/subgraph`
Extracts an authentic $k$-hop directed subgraph centered around a root transaction node using CSR adjacency arrays (`row_ptr`, `col_idx`).

- **Method:** `GET`
- **Route:** `/api/v1/graph/subgraph`
- **Query Parameters:**
  - `node_id` *(int, required)*: Root transaction node index ($0 \le \text{node\_id} < N$).
  - `hops` *(int, default: 1)*: Traversal depth ($1 \le \text{hops} \le 3$).
  - `max_nodes` *(int, default: 150)*: Maximum allowed vertices in returned subgraph ($1 \le \text{max\_nodes} \le 300$).
  - `max_edges` *(int, default: 500)*: Maximum allowed directed edges ($1 \le \text{max\_edges} \le 1000$).
  - `npz_path` *(string, optional)*: Path override to NPZ dataset file.
- **Topological Invariants:**
  - **Directed CSR Traversal:** Traverses only authentic forward edges `u -> v` present in `col_idx`. Does not synthesize reverse edges.
  - **Deterministic Ordering:** Nodes ordered by `(depth, id)` ascending; edges sorted by `(source, target)`.
  - **Label Quarantine:** Neighbor selection and BFS expansion operate purely on adjacency topology without reading fraud labels. Labels are joined strictly after structural traversal.
  - **Truncation Transparency:** When `max_nodes` or `max_edges` caps are hit, `is_truncated` is set to `true`, and `truncation_reason` discloses whether nodes or edges were truncated.

#### Response Model (`SubgraphResponse`)
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
    "total_nodes": 24,
    "total_edges": 31,
    "fraud_node_count": 2,
    "fraud_percentage": 8.33,
    "is_truncated": false,
    "node_cap_hit": false,
    "edge_cap_hit": false,
    "truncation_reason": null
  },
  "nodes": [
    {
      "id": 5407,
      "depth": 0,
      "is_root": true,
      "timestamp_sec": 1401205.12,
      "fraud_label": 1,
      "is_fraud": true,
      "degree": 23
    }
  ],
  "edges": [
    {
      "source": 5407,
      "target": 5410,
      "temporal_delta_sec": 142.5
    }
  ],
  "provenance": {
    "fixture_name": "synthetic_fraud_10k.npz",
    "sha256": "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430",
    "causality_policy": "causal_history (directed CSR)"
  }
}
```

#### Error Responses
- **400 Bad Request:**
  - `node_id` out of range: `"node_id 15000 is out of bounds. Valid range is 0 to 9999."`
  - Parameter constraints violated (`hops` not in 1â€“3, `max_nodes` not in 1â€“300, `max_edges` not in 1â€“1000).
  - Path traversal escape attempt in `npz_path`.
- **404 Not Found:** Specified `npz_path` fixture does not exist.
- **503 Service Unavailable:** NPZ archive is corrupt or missing required CSR arrays.

---

## 5. Transactions Forensic Ledger Endpoints

### `GET /api/v1/transactions`
Provides paginated, filtered transaction records derived authentically from dataset fixtures.

- **Method:** `GET`
- **Route:** `/api/v1/transactions` (also supports trailing slash `/api/v1/transactions/`)
- **Query Parameters:**
  - `offset` *(int, default: 0)*: Zero-based record offset ($\ge 0$).
  - `limit` *(int, default: 50)*: Number of items per page ($1 \le \text{limit} \le 100$).
  - `filter` *(string, default: "all")*: Partition filter. Allowed values: `"all"`, `"fraud"`, `"licit"`.
  - `npz_path` *(string, optional)*: Path override to NPZ dataset file.
- **Data Integrity & Ordering:**
  - **Ordering:** Chronological by original dataset index (`original_transaction_index_chronological`).
  - **Pre-Slice Filtering:** Filters by label before applying pagination offset/limit.
  - **Accurate Pagination Metadata:** `filtered_count`, `total_dataset_records`, `total_fraud_records`, `total_licit_records`, `has_next`, `has_prev`.
  - **Honest Schema Representation:** Raw relational string attributes (`source_card`, `target_device`) were discarded during graph array compression; they are returned truthfully as `null` with explanatory notes. Feature samples provide the first 4 feature dimensions.

#### Response Model (`TransactionListResponse`)
```json
{
  "status": "success",
  "total_dataset_records": 10000,
  "total_fraud_records": 350,
  "total_licit_records": 9650,
  "filtered_count": 350,
  "offset": 0,
  "limit": 25,
  "returned_count": 25,
  "has_next": true,
  "has_prev": false,
  "current_filter": "fraud",
  "ordering": "original_transaction_index_chronological",
  "items": [
    {
      "tx_id": 142,
      "timestamp_sec": 36812.5,
      "fraud_label": 1,
      "is_fraud": true,
      "risk_tier": "CONFIRMED_FRAUD",
      "in_degree": 4,
      "feature_dim": 32,
      "feature_sample": [-0.1425, 0.8812, 1.2045, -0.0512],
      "source_card": null,
      "target_device": null,
      "notes": "Relational entity strings were discarded during CSR compression; graph topology preserves entity relations via directed edges."
    }
  ],
  "schema_notes": "source_card and target_device are omitted in CSR graph representation; in_degree indicates connected transactions in historical graph.",
  "provenance_sha256": "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430"
}
```

#### Error Responses
- **400 Bad Request:**
  - Invalid filter value: `"Invalid filter 'suspicious'. Must be one of 'all', 'fraud', or 'licit'."`
  - Negative offset or limit out of bounds ($1 \le \text{limit} \le 100$).
  - Path traversal attempt in `npz_path`.
- **404 Not Found:** Dataset fixture file not found.
- **503 Service Unavailable:** Dataset fixture corrupt or missing required arrays.

---

## 6. Deferred / Unimplemented Endpoints (Historical Roadmap)

Early project roadmaps envisioned client-triggered compute mutation endpoints:
- `POST /api/v1/compute/aggregate` (Neighbor aggregation trigger)
- `GET /api/v1/experiments` (Interactive experiment database)

**Implementation Status:** **DEFERRED / SUPERSEDED**
In the finalized architecture, benchmark execution and model training are performed offline via reproducible Python CLI engines (`experiments/` and `backend/cpu_engine/`) under strict cryptographic provenance. The FastAPI layer operates as a read-only analytics, telemetry, and forensic exploration server.
