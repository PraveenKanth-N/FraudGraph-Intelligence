# API Contract & Endpoint Specification — FraudGraph Intelligence (FGI)

**Base URL:** `http://127.0.0.1:8000/api/v1`  
**Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`  
**OpenAPI Spec JSON:** `http://127.0.0.1:8000/openapi.json`

---

## 1. System & Telemetry Endpoints

### `GET /api/v1/health`
Returns system diagnostics, runtime health, and local vs. cloud GPU execution status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "project": "FraudGraph Intelligence",
  "project_id": "FGI",
  "version": "1.0.0",
  "compute_engine": {
    "backend": "cpu_reference",
    "cuda_available_locally": false,
    "local_gpu_detected": "NVIDIA GeForce MX450",
    "vram_free_mb": 1907.0,
    "cuda_compiler_status": "UNAVAILABLE_LOCAL_COLAB_READY",
    "colab_workflow_enabled": true
  },
  "system": {
    "python_version": "3.14.3",
    "timestamp": "2026-10-18T19:30:00Z"
  }
}
```

---

## 2. Dataset Endpoints (Planned for Phase 3)

### `GET /api/v1/datasets/summary`
Lists available datasets, row counts, feature dimensionality, and missingness metrics.

### `POST /api/v1/datasets/build-graph`
Constructs a CSR graph from an active dataset with user-selected relational rules.
**Request Body:**
```json
{
  "dataset_name": "ieee_cis_sampled",
  "edge_keys": ["card1", "P_emaildomain"],
  "temporal_window_days": 7,
  "max_nodes": 50000
}
```

---

## 3. Graph Explorer Endpoints (Planned for Phase 3 & 4)

### `GET /api/v1/graph/metadata`
Returns global graph metrics: total nodes, total edges, density, degree distribution (min, max, mean, median, skewness).

### `GET /api/v1/graph/neighborhood/{node_id}`
Returns immediate $k$-hop neighborhood around a given transaction node for Cytoscape.js rendering.

---

## 4. Compute & Benchmark Endpoints (Planned for Phase 5, 6, 8)

### `POST /api/v1/compute/aggregate`
Triggers parallel neighbor aggregation on the loaded graph using the specified engine.
**Request Body:**
```json
{
  "engine": "cpu_reference",
  "aggregation_op": "mean",
  "feature_dim": 64
}
```

### `GET /api/v1/experiments`
Lists all executed benchmark runs, configurations, kernel times, PCIe transfer times, and speedup ratios.
