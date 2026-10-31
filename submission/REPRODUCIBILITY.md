# FraudGraph Intelligence (FGI): Complete Reproducibility Guide

**Document Status:** Official Reproducibility Specification & Execution Runbook  
**Course:** CSI4007 — GPU Programming Capstone  
**Institution:** Vellore Institute of Technology (VIT), Vellore Campus  
**Workspace:** `D:\FraudGraph Intelligence`  
**Git Baseline Commit:** `8afa28cf1172c006f0242f530267e8cc4466f9d6` (Branch `main`)  
**Last Verified:** 2026-10-25T17:51:15+05:30  

---

## 1. System Requirements & Tested Environments

The platform has been rigorously tested and verified across two primary operating environments:

### Environment A: Local Development & API Host (Tested)
- **OS:** Microsoft Windows 11 Enterprise (64-bit AMD64)
- **CPU:** AMD64 x86_64 Multi-Core Processor (16 GB System RAM)
- **Python:** Python 3.14.3 (compatible with Python 3.10+)
- **Node.js:** Node.js v22.14.0 (compatible with Node 18+)
- **Package Managers:** pip 25.0+, npm 10.9.2
- **GPU Status:** No local GPU required for CPU engine, API services, and analytical dashboard.

### Environment B: Remote CUDA Benchmark Host (Tested)
- **OS:** Ubuntu Linux 22.04 LTS (x86_64) on Google Colab
- **GPU Accelerator:** NVIDIA Tesla T4 (Turing, SM 7.5, 40 SMs, 14.9 GB VRAM)
- **NVIDIA Driver:** 535.104.05
- **CUDA Toolkit:** CUDA 12.2 / 12.x (`nvcc` V12.2.140)
- **Build Tools:** `make`, `gcc` 11.4+, `g++` 11.4+

---

## 2. Fast-Path Verification (Smoke Test Checklist)

To verify the integrity of the repository in under 2 minutes:

### Step 1: Verify Canonical Fixture Hash
Verify that the canonical dataset fixture matches the authoritative cryptographic signature:
```powershell
# Windows PowerShell
Get-FileHash "data\fixtures\synthetic_fraud_10k.npz" -Algorithm SHA256

# Linux / macOS Bash
sha256sum data/fixtures/synthetic_fraud_10k.npz
```
**Expected Output:**
`AECAFE3B1FD608C18D62AA1586D0AB6F88B867BD92B20E0F6BD133606C7C4430`

### Step 2: Run Backend Pytest Suite
Execute the 190-item backend verification test suite:
```bash
python -m pytest -o filterwarnings="ignore::DeprecationWarning"
```
**Expected Output:** `188 passed, 2 skipped in ~90s` (Exit Code 0).  
*(Note: 2 tests gracefully skip when no local NVIDIA GPU is detected).*

### Step 3: Run Frontend Verification Suites
Execute the frontend Node test suites, TypeScript typecheck, and production build:
```bash
# 1. Run Polling Lifecycle & State Consistency Suite (21 tests)
node frontend/tests/polling_lifecycle.test.mjs

# 2. Run API Client & Subgraph Endpoint Tests (3 tests)
node frontend/tests/subgraph_transactions_client.test.mjs

# 3. Verify TypeScript Type Safety
cd frontend
npm run typecheck

# 4. Verify Production Asset Bundling
npm run build
```
**Expected Output:** All 24 Node tests pass; TypeScript reports 0 errors; Vite builds 1,621 modules cleanly into `frontend/dist/`.

---

## 3. Local Installation & Service Execution Guide

### 3.1 Python Backend Environment Setup
1. Clone or navigate to the repository:
   ```bash
   cd "D:\FraudGraph Intelligence"
   ```
2. (Optional) Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate
   ```
3. Install required backend dependencies:
   ```bash
   pip install fastapi uvicorn pydantic numpy scipy pandas scikit-learn pytest matplotlib
   ```

### 3.2 Starting the FastAPI Backend Service
Launch the backend API on port 8000:
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, the interactive OpenAPI documentation is accessible at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

#### Smoke-Checking API Endpoints via cURL:
```bash
# System Health & Hardware Detection
curl -s http://localhost:8000/api/v1/health

# Consolidated Dashboard Telemetry
curl -s http://localhost:8000/api/v1/dashboard/summary

# Dataset Registry
curl -s http://localhost:8000/api/v1/datasets

# Topological Subgraph Extraction (BFS Depth 2, Node 0)
curl -s "http://localhost:8000/api/v1/datasets/synthetic_fraud_10k/graph/subgraph?center_node_id=0&max_depth=2&max_edges=100"

# Transaction Ledger (Pagination & Fraud Filter)
curl -s "http://localhost:8000/api/v1/datasets/synthetic_fraud_10k/transactions?offset=0&limit=10&is_fraud=true"
```

### 3.3 Starting the React/Vite Frontend Dashboard
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm dependencies (if not already installed):
   ```bash
   npm install
   ```
3. Start the Vite development server on port 5173:
   ```bash
   npm run dev
   ```
4. Open your browser and navigate to:
   - **Analytical Dashboard:** `http://localhost:5173/`

---

## 4. Google Colab CUDA Acceleration Workflow

Because local development machines frequently lack discrete NVIDIA GPUs, FraudGraph Intelligence provides an automated Google Colab workflow to compile and execute the low-level CUDA kernels on a cloud Tesla T4 GPU.

### 4.1 Colab Workflow Files
All Colab automation utilities reside in `backend/gpu_engine/colab_runner/`:
- `fgi_colab_workflow.ipynb`: Step-by-step interactive Jupyter notebook.
- `package_colab_bundle.py`: Bundles CUDA source code, Makefile, fixtures, and runner scripts into a single zip file.
- `run_colab_benchmarks.py`: Automated headless benchmark runner executing all 4 kernel implementations across uniform and skewed graphs.
- `ingest_colab_results.py`: Unpacks and validates Colab benchmark JSON results into `experiments/results/`.

### 4.2 Step-by-Step Colab Execution Runbook
1. **Package the Colab Bundle:**
   ```bash
   python backend/gpu_engine/colab_runner/package_colab_bundle.py
   ```
   *Creates `experiments/colab_bundle_fgi_phase6.zip`.*
2. **Open Google Colab:**
   - Navigate to [Google Colab](https://colab.research.google.com/).
   - Upload and open `backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb`.
3. **Configure Hardware Accelerator:**
   - In the Colab menu, select **Runtime $\to$ Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU**.
4. **Upload and Unpack Bundle in Colab:**
   ```python
   # Inside Colab cell
   from google.colab import files
   uploaded = files.upload() # Upload colab_bundle_fgi_phase6.zip

   !unzip -q colab_bundle_fgi_phase6.zip -d /content/fgi
   %cd /content/fgi
   ```
5. **Compile CUDA Kernels:**
   ```bash
   !make -C backend/gpu_engine/cuda
   ```
6. **Execute GPU Benchmarks:**
   ```bash
   !python backend/gpu_engine/colab_runner/run_colab_benchmarks.py
   ```
7. **Download Generated Results:**
   The runner saves JSON results matching `exp_gpu_*_<timestamp>.json` and packages them into a zip file for download.
8. **Ingest Results into Repository:**
   ```bash
   python backend/gpu_engine/colab_runner/ingest_colab_results.py --zip-path <downloaded_zip>
   ```

---

## 5. Troubleshooting & Common Failure Modes

| Symptom / Error Message | Root Cause | Verified Remediation |
| :--- | :--- | :--- |
| `WinError 10048: Only one usage of each socket address is normally permitted` | Port 8000 or 5173 is already in use by a zombie process. | Identify the process holding port 8000 via `netstat -ano \| findstr :8000` and terminate it, or launch uvicorn on an alternate port (`--port 8001`). |
| `local_gpu_detected: None` / `cuda_compiler_status: NO_GPU_DETECTED_COLAB_READY` | Windows host does not have an active NVIDIA GPU or NVCC compiler. | **Expected behavior.** The platform operates in CPU host mode. All GPU benchmarks are evaluated via verified Colab Tesla T4 artifacts. |
| `DeprecationWarning: Starlette test client...` during pytest | Recent Starlette versions emit deprecation notices. | Pass the flag `-o filterwarnings="ignore::DeprecationWarning"` to pytest. |
| `422 Unprocessable Entity` on API requests | Query parameters failed Pydantic validation (e.g. `limit > 100` or `max_depth > 3`). | Keep `limit` between 1 and 100, and `max_depth` between 1 and 3 as specified in `docs/API_CONTRACT.md`. |
| `frontend/tests/api_integration.test.mjs` fails connection | Backend server on port 8000 or Vite server on port 5173 is not running. | Start backend (`uvicorn`) and frontend (`npm run dev`) before executing live socket integration tests. |

---
*End of Reproducibility Guide.*
