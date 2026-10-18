# FraudGraph Intelligence (FGI)

**Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation**

*Academic Course:* VIT Vellore — CSI4007 GPU Programming  
*Project ID:* FGI  
*Platform Identity:* Professional, research-oriented graph-based financial fraud analysis and GPU performance engineering platform.

---

## 1. Executive Summary

Financial fraud detection increasingly relies on Graph Neural Networks (GNNs) and structural graph embeddings to capture complex relational anomalies (e.g., identity spoofing, synthetic fraud rings, shared credit instruments, and rapid fund dissipation). However, message passing across power-law transaction graphs exhibits severe memory-access irregularity:
1. **Divergent neighborhood degrees** that cause warp serialization on GPUs.
2. **Uncoalesced global memory reads** during neighbor feature gathering from high-dimensional feature matrices.
3. **Severe PCI-e transfer bottlenecks** during graph staging.

**FraudGraph Intelligence** investigates, measures, and optimizes **Parallel Neighbor Aggregation** using memory-aware CUDA C++ kernels. By implementing and benchmarking thread-per-node, memory-coalesced warp-per-node, and shared-memory tiling strategies against a validated CPU reference implementation, this project demonstrates how hardware-conscious parallel algorithms accelerate fraud classification while maintaining numerical precision.

---

## 2. Core Technical Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    FRAUDGRAPH INTELLIGENCE                                         |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   +-----------------------+      +---------------------------+      +--------------------------+   |
|   |  Dataset Engineering  | ---> |   CSR Graph Representation| ---> |   Parallel Aggregation   |   |
|   |  (IEEE-CIS Subset)    |      | (row_ptr, col_idx, feat)  |      |  (CPU vs CUDA Kernels)   |   |
|   +-----------------------+      +---------------------------+      +--------------------------+   |
|                                                                                   |                |
|                                                                                   v                |
|   +-----------------------+      +---------------------------+      +--------------------------+   |
|   |  Stitch React Frontend| <--- |   FastAPI Engine (/v1)    | <--- |   Fraud Evaluation &     |   |
|   | (Obsidian Cybernetics)|      | (Telemetry, Graph, Jobs)  |      |   Benchmark Metrics      |   |
|   +-----------------------+      +---------------------------+      +--------------------------+   |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

- **Frontend:** React 18, TypeScript, Vite, Obsidian Cybernetics Design System (zero-radius, dark obsidian minimal, Newsreader serif + Geist sans).
- **Backend:** Python 3.14, FastAPI, Pydantic v2, REST API under `/api/v1`.
- **Compute Engine (CPU):** Deterministic CSR graph aggregator in NumPy/SciPy.
- **Compute Engine (GPU):** Custom CUDA C++ kernels (`baseline`, `coalesced`, `shared-memory`) with Colab GPU workflow for verified cloud execution when local `nvcc` is unavailable.
- **Evaluation:** scikit-learn baseline classifier, PR-AUC, F1-score, Precision, Recall, Confusion Matrix.

---

## 3. Directory Layout

```
D:\FraudGraph Intelligence
├── .env.example              # Documented configuration variables
├── .gitignore                # Git ignore rules for venv, node_modules, datasets
├── README.md                 # Project overview and instructions
├── docs/                     # Authoritative engineering and research documentation
│   ├── PHASE_STATUS.md       # Implementation progress tracker
│   ├── ENVIRONMENT_SETUP.md  # Hardware and toolchain diagnostics
│   ├── PROJECT_CHARTER.md    # Course identity and scope
│   ├── RESEARCH_PLAN.md      # Mathematical models and hypotheses
│   ├── RESEARCH_GAP.md       # Analysis of existing limitations
│   ├── SYSTEM_ARCHITECTURE.md# High-level architecture and data flows
│   ├── DATA_DICTIONARY.md    # Feature schemas and graph derivations
│   ├── ALGORITHM_SPECIFICATION.md # Mathematical specs for kernels
│   ├── API_CONTRACT.md       # OpenAPI schema and endpoints
│   ├── DESIGN_SYSTEM.md      # Obsidian Cybernetics tokens and layout
│   ├── EXPERIMENT_PROTOCOL.md# Benchmark execution rules
│   ├── TEST_PLAN.md          # Multi-layer test strategy
│   ├── SECURITY_AND_PRIVACY.md # Data governance and integrity rules
│   ├── KNOWN_LIMITATIONS.md  # Architectural and hardware limits
│   ├── DECISIONS.md          # Architectural Decision Records (ADRs)
│   └── CHANGELOG.md          # Release history
├── backend/                  # FastAPI backend and engines
│   ├── app/                  # REST API implementation
│   ├── cpu_engine/           # CSR graph reference aggregator
│   ├── gpu_engine/           # CUDA C++ kernels and Colab runner
│   └── tests/                # Backend unit and integration tests
├── frontend/                 # React + TypeScript + Vite UI
└── stitch_fraudgraph_intelligence_platform/ # Stitch design export source
```

---

## 4. Quickstart

### 4.1 Backend (FastAPI)
```powershell
# Activate local virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies (if updating)
pip install -r backend/requirements.txt

# Run backend development server
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive API docs: `http://127.0.0.1:8000/docs`

### 4.2 Frontend (React + Vite)
```powershell
cd frontend
npm install
npm run dev
```
UI Application: `http://127.0.0.1:5173/`

### 4.3 Running Tests
```powershell
# Run backend pytest suite
.\.venv\Scripts\pytest backend/tests -v

# Run frontend typecheck
cd frontend
npm run typecheck
```
