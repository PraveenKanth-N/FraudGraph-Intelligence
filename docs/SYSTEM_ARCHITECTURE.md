# System Architecture — FraudGraph Intelligence (FGI)

## 1. High-Level System Architecture

FraudGraph Intelligence is architected as an asynchronous, decoupled research and forensic platform consisting of five primary subsystems:

```
+---------------------------------------------------------------------------------------------------------+
|                                    PRESENTATION LAYER (REACT 18 + TS)                                   |
|   - Obsidian Cybernetics Design System (Zero-Radius Dark Minimal, Telemetry Grids)                      |
|   - Interactive Graph Explorer (Cytoscape.js) & Performance Visualizers (Apache ECharts)                |
|   - Application Routes: / (Home), /methodology, /app/overview, /app/datasets, /app/graph,               |
|     /app/transactions, /app/gpu-lab, /app/experiments, /app/evaluation, /app/reports, /app/system       |
+---------------------------------------------------------------------------------------------------------+
                                                     |  REST API (/api/v1) & JSON Payloads
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                                      API & SERVICE LAYER (FASTAPI)                                      |
|   - Endpoints: Health, Dataset Ingestion, Graph Query, Compute Dispatcher, Experiments, Evaluation     |
|   - ThreadPool Task Execution for non-blocking graph building & benchmark orchestration                 |
+---------------------------------------------------------------------------------------------------------+
            |                                                                   |
            v                                                                   v
+---------------------------------------+           +---------------------------------------------------+
|      GRAPH & CPU ENGINE (NUMPY/SCIPY) |           |              GPU ACCELERATION ENGINE              |
+---------------------------------------+           +---------------------------------------------------+
| - IEEE-CIS Data Pipeline & Scrubbing  |           | - CUDA C++ Kernels (baseline, coalesced, shared)  |
| - Relational Adjacency Construction   |           | - Hardware-Specific Thread & Warp Mappings        |
| - CSR Matrix Representation           |           | - Colab GPU Runner (Verified Cloud Execution)     |
|   (row_ptr, col_idx, features)        |           | - Exact Timing via cudaEventRecord / cudaEvent-   |
| - Deterministic Baseline Aggregation  |           |   ElapsedTime                                     |
| - scikit-learn Fraud Evaluation Suite |           | - Verification against CPU reference (atol <= 1e-5)|
+---------------------------------------+           +---------------------------------------------------+
                                          \       /
                                           v     v
+---------------------------------------------------------------------------------------------------------+
|                                         PERSISTENCE & REPRODUCIBILITY                                   |
|   - /data/raw: Immutable source archives (IEEE-CIS subset)                                              |
|   - /data/processed: Cleaned CSR arrays & feature tensors                                               |
|   - /experiments/results: Machine-readable JSON benchmark logs, metrics, and classification reports    |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Subsystem Detailed Descriptions

### 2.1 Graph & CPU Engine
- **Module ID:** FGI-MOD-02, FGI-MOD-03, FGI-MOD-04
- **Language:** Python 3.14 (NumPy 2.5, SciPy 1.18, Pandas 3.0)
- **Role:** Ingests tabular transaction data, establishes justified relational edges (e.g. shared `card1` credit instrument, shared device fingerprint, or shared `P_emaildomain`), builds 0-indexed contiguous CSR arrays, and computes deterministic reference embeddings.

### 2.2 CUDA Acceleration Engine
- **Module ID:** FGI-MOD-05, FGI-MOD-06
- **Language:** CUDA C++ (NVCC, C++17)
- **Role:** High-throughput parallel execution of neighbor feature gathering and reduction. Supports three kernel variants:
  1. `fgi_aggregator_baseline.cu`: 1 thread per node (naive).
  2. `fgi_aggregator_coalesced.cu`: 32 threads (1 warp) per node with vectorized feature strides.
  3. `fgi_aggregator_shared.cu`: Threadblock-level shared-memory caching of high-degree node neighbor lists.

### 2.3 Colab Execution Adapter
- **Module ID:** FGI-MOD-05, FGI-MOD-08
- **Role:** Bridges local development on Windows (where `nvcc` is missing) to an NVIDIA GPU runtime (e.g., T4/A100) via self-contained execution bundles. Results are programmatically validated and synced back.

### 2.4 FastAPI Backend Service
- **Module ID:** FGI-MOD-09
- **Framework:** FastAPI, Uvicorn, Pydantic v2
- **Port:** `8000` (Localhost)
- **Contract:** Documented OpenAPI specification at `/docs`.

### 2.5 Frontend Client
- **Module ID:** FGI-MOD-10
- **Framework:** React 18, Vite, TypeScript
- **Port:** `5173` (with proxy to backend `:8000`)
- **Theme:** Obsidian Cybernetics (Zero-Radius Dark Minimal).
