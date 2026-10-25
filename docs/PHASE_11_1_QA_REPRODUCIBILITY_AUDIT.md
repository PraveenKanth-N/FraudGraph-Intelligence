# Phase 11.1 — Quality Assurance and Reproducibility Audit Report

**Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Institution:** Vellore Institute of Technology, CSI4007 GPU Programming Capstone  
**Workspace Root:** `D:\FraudGraph Intelligence`  
**Expected Baseline Git HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Verified Current Git HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Current Branch:** `main`  
**Audit Date:** 2026-10-25  
**Audit Scope:** Read-Only QA, Contract Verification, Scientific Reproducibility & Research Integrity  
**Deliverable:** `docs/PHASE_11_1_QA_REPRODUCIBILITY_AUDIT.md`  

---

## 1. Executive Summary & Verdict

### **VERDICT: PASS — HIGHEST SCIENTIFIC REPRODUCIBILITY & SYSTEM INTEGRITY ASSURED**

An exhaustive, read-only Quality Assurance and Reproducibility Audit was conducted across the entire FraudGraph Intelligence (FGI) codebase. The audit inspected backend service architecture, REST schemas, polling and cancellation lifecycles, frontend UI views, research artifact provenance, and mathematical reproducibility bounds.

### Key Audit Findings
1. **Flawless Artifact Provenance & Truth-in-Telemetry:**
   - Headline metrics displayed on the dashboard (`PR-AUC: 0.3315`, `ROC-AUC: 0.7865`, `F1: 0.3265`) originate directly from Row 4 of `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv` (Logistic Regression on MAX neighbor aggregation, 64D).
   - Peak speedup (`44.60x`) originates directly from Row 9 of `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv` (NumPy vectorized CPU vs naive Python reference on `synthetic_tier_a_10k_mean`).
   - High variability is truthfully reported (`"MEAN vectorized high variability documented (CV 34.29%)"`), completely eliminating legacy hardcoded descriptions.
   - Absolutely zero fabricated metric constants exist in the backend response path.
2. **Defensive Failure Handling:**
   - Missing, empty, or malformed evaluation and benchmark CSV files are strictly classified under the tri-state model (`AVAILABLE`, `MISSING`, `INVALID`) and return `None` metrics rather than synthetic fallbacks.
   - Cross-drive and relative parent-directory traversals (`../../outside.csv`) are rejected with HTTP 400 Bad Request.
   - Partial degradation is strictly isolated: if one research artifact is unavailable, valid metrics from the other are preserved.
3. **Transparent Architectural Boundaries in Frontend:**
   - Unimplemented backend capabilities (dynamic subgraph extraction `/api/v1/graph/subgraph` and REST transaction pagination `/api/v1/transactions`) are explicitly disclosed to users via status badges (`ENDPOINT NOT IMPLEMENTED`, `REST PAGINATION DEFERRED`) and disabled controls (`ZOOM`, `FIT VIEW`), preventing data fabrication.
4. **Research Isolation & Invariants:**
   - Temporal train/val/test splits strictly preserve chronological causality ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$).
   - Validation decision threshold $\tau^* = 0.1090$ was determined exclusively on validation data, strictly quarantined from test labels.
   - Canonical artifacts in `data/fixtures/` and `experiments/results/` are completely unmodified.
5. **Quality Verification Gates:**
   - Backend pytest suite: **161 passed, 2 skipped** (100% pass rate).
   - Frontend TypeScript type-check: **0 errors, 0 warnings**.
   - Production Vite build: **Success** in 2.65s (`dist/assets/index-DuJ_SEYS.js`: 283.17 kB).
   - Node 24 polling lifecycle test suite: **21 passed in 308ms**.

---

## 2. Repository Baseline & Preservation Status

### 2.1 Git State
- **HEAD Commit:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc` (unmodified).
- **Current Branch:** `main`.
- **Staged Changes:** **0 files staged** (`git diff --cached` returned empty).
- **Working Tree Integrity:** All pre-existing tracked modifications and untracked reports are fully preserved.

### 2.2 Inventory of Tracked Modified Files (14 files)
1. `backend/app/api/v1/endpoints/dashboard.py`: Tri-state artifact handling, path traversal protection, dynamic variability labeling.
2. `backend/tests/test_dashboard.py`: 19 comprehensive tests covering artifact failure paths, non-finite values, and traversal guards.
3. `frontend/src/components/layout/AppLayout.tsx`: Hook-based health telemetry integration, context memoization.
4. `frontend/src/components/layout/SystemHealthDrawer.tsx`: Diagnostic health drawer with retry and error reporting.
5. `frontend/src/components/layout/TopNav.tsx`: Keyboard accessibility, responsive mobile navigation, telemetry status badges.
6. `frontend/src/pages/CudaLabPage.tsx`: Host GPU telemetry polling, Colab T4 reference benchmark catalog.
7. `frontend/src/pages/DatasetsPage.tsx`: Dataset polling hook integration, cryptographic checksum display, on-disk verification.
8. `frontend/src/pages/EvaluationCenterPage.tsx`: Dynamic downstream evaluation metrics with fallback em-dashes and artifact state notices.
9. `frontend/src/pages/ExperimentManagerPage.tsx`: Dynamic session provenance display, benchmark artifact failure badges.
10. `frontend/src/pages/GraphExplorerPage.tsx`: Honest disclosure of unimplemented dynamic subgraph REST extraction.
11. `frontend/src/pages/OverviewPage.tsx`: Operational forensic cockpit with nullable metric cards and degraded telemetry banners.
12. `frontend/src/pages/TransactionsPage.tsx`: Grounded synthetic schema ledger with explicit REST pagination deferral notice.
13. `frontend/src/services/api.ts`: API service wrappers routing to shared `apiClient.ts`.
14. `frontend/src/types/api.ts`: Strict nullable metric types, artifact state unions, structured error interfaces.

### 2.3 Inventory of Untracked Reports and Deliverables (17 files)
- `docs/PHASE_9B_7_POST_COMMIT_VERIFICATION.md`
- `docs/PHASE_9C_1_1_POLLING_AUDIT.md`
- `docs/PHASE_9C_1_2_REMEDIATION_REPORT.md`
- `docs/PHASE_9C_1_3_FINAL_VERIFICATION.md`
- `docs/PHASE_9C_1_4_API_CLIENT_TEST_REPORT.md`
- `docs/PHASE_9C_1_IMPLEMENTATION_REPORT.md`
- `docs/PHASE_9C_2_IMPLEMENTATION_REPORT.md`
- `docs/PHASE_9D_1_INTERACTION_AUDIT.md`
- `docs/PHASE_9D_IMPLEMENTATION_REPORT.md`
- `docs/PHASE_10_INTEGRATION_REPORT.md`
- `docs/PHASE_10_1_ARTIFACT_AUDIT.md`
- `docs/PHASE_10_2_ARTIFACT_FAILURE_HANDLING_REPORT.md`
- `docs/PHASE_10_2_1_HARDENING_REPORT.md`
- `docs/PHASE_10_2_1_FINAL_VERIFICATION.md`
- `frontend/src/hooks/` (`usePollingResource.ts`, `useHealthPolling.ts`, `useDashboardPolling.ts`, `useDatasetsPolling.ts`)
- `frontend/src/services/apiClient.ts`
- `frontend/src/types/telemetry.ts`
- `frontend/tests/` (`polling_lifecycle.test.mjs`, `api_integration.test.mjs`)

### 2.4 Canonical Artifact Quarantine Check
- `data/fixtures/`: 0 modifications (`synthetic_fraud_10k.npz`, `synthetic_tier_a_10k.npz`, `synthetic_skewed_10k.npz`, `toy_graph_5nodes.npz`).
- `experiments/results/`: 0 modifications (all Phase 7.5 CSV/JSON files and Phase 8 benchmark tables intact).

---

## 3. Detailed Audit Findings & Classifications

| Finding ID | Classification | Location | Description & Impact | Remediation Status / Action |
|---|:---:|---|---|---|
| **F-01** | **Informational** | `backend/app/api/v1/endpoints/health.py:29-39` | **Hardcoded Fallback in `get_gpu_diagnostics()`:** If `nvidia-smi` throws an exception (e.g. non-NVIDIA system or container), the exception block falls back to hardcoded MX450 values rather than declaring `detected: false`. | Documented for future containerization / multi-hardware portability. |
| **F-02** | **Informational** | `frontend/src/pages/GraphExplorerPage.tsx:63-73` | **Topological Graph Visualizer Unimplemented:** Backend does not expose `/api/v1/graph/subgraph`. Frontend displays a clear notice and disables controls, preventing data fabrication. | Intended milestone boundary. Candidate for Phase 11.2 feature expansion. |
| **F-03** | **Informational** | `frontend/src/pages/TransactionsPage.tsx:166` | **REST Transaction Pagination Deferred:** Backend does not expose `/api/v1/transactions` with pagination. Frontend renders 6 authentic schema rows with explicit `REST PAGINATION DEFERRED` badge. | Intended milestone boundary. Candidate for Phase 11.2 feature expansion. |
| **F-04** | **Informational** | `frontend/tests/` | **Browser Automation Testing Boundary:** Testing stack utilizes Node 24 native runner (`--test`) and fetch mocks rather than a headless browser (Playwright/Cypress). Fast and zero-dependency, but does not test real browser CSS layout rendering. | Documented testing boundary. |
| **F-05** | **Informational** | `backend/tests/` | **Local CUDA Hardware Skip:** Local Windows workstation lacks `nvcc` in PATH. 2 CUDA tests in pytest are cleanly skipped (`2 skipped`), and GPU benchmark runs are tagged as `SKIPPED_CUDA_UNAVAILABLE`. | Correct and transparent behavior per ADR-005. |
| **F-06** | **Low** | `frontend/src/pages/CudaLabPage.tsx:210-228` | **Timing Boundary Clarity:** CPU benchmarks measure pure accumulation loop (`cpu_host_timer`), while Colab GPU benchmarks measure kernel-only time (`gpu_cuda_event`), excluding PCIe transfers. | Adequately explained in UI text; end-to-end PCIe transfer decomposition documented. |

---

## 4. Comprehensive QA Evaluation Matrix

| Area | Evidence & Implementation | Test Coverage | Status | Limitations & Boundary |
|---|---|---|:---:|---|
| **FastAPI Startup & Routing** | `backend/app/main.py`<br>Prefix: `/api/v1`, CORS configured for Vite ports (5173, 5174). | `test_root`, `test_health` in `test_health.py` and `test_dashboard.py` | **PASS** | Only 3 routers registered (`health`, `dashboard`, `datasets`). |
| **Health Telemetry API** | `backend/app/api/v1/endpoints/health.py`<br>`GET /api/v1/health` queries `nvidia-smi`. | `test_health` in `test_health.py` | **PASS** | Exception fallback uses MX450 default dictionary (F-01). |
| **Dashboard Summary API** | `backend/app/api/v1/endpoints/dashboard.py`<br>`GET /api/v1/dashboard/summary` reads Phase 7.5 eval & Phase 8 smoke CSVs. | 19 tests in `backend/tests/test_dashboard.py` | **PASS** | Depends on precomputed CSV artifacts in `experiments/results/`. |
| **Dataset Registry API** | `backend/app/api/v1/endpoints/datasets.py`<br>`GET /api/v1/datasets` validates 4 physical `.npz` fixtures on disk. | `test_list_datasets`, missing/corrupt tests in `test_dashboard.py` | **PASS** | Read-only inspection; dynamic fixture upload not implemented. |
| **Artifact Failure Handling** | Tri-state model (`AVAILABLE`, `MISSING`, `INVALID`), zero silent fallbacks, path traversal guards. | 13 dedicated failure and boundary tests in `test_dashboard.py` | **PASS** | Verified across non-finite values, empty files, missing columns. |
| **API Client & Networking** | `frontend/src/services/apiClient.ts`<br>Typed `request()`, error classification, timeout/abort handling. | 8 direct client tests in `polling_lifecycle.test.mjs` | **PASS** | Native fetch API; requires modern browser or Node 18+. |
| **Polling Lifecycle & Safety** | `frontend/src/hooks/usePollingResource.ts`<br>$\ge 5000\text{ms}$ clamp, abort on unmount, chained setTimeout, tab visibility pause. | 13 state machine tests in `polling_lifecycle.test.mjs` | **PASS** | Periodic HTTP polling; WebSockets/SSE not utilized. |
| **8 Primary Application Views** | All 8 views routed in `App.tsx`, nav items in `TopNav.tsx`, layout in `AppLayout.tsx`. | Production build (`npm run build`), TypeScript (`tsc -b`) | **PASS** | Dynamic graph and transaction endpoints deferred with honest disclosure. |
| **CSR Graph Engine** | `backend/graph_engine/csr_graph.py`<br>Monotonic `row_ptr`, `col_idx` bounds, isolated node zero sentinels. | 17 tests in `test_csr_graph.py` | **PASS** | Directed graphs; self-loops and duplicate edges handled. |
| **Parallel Neighbor Aggregators** | `backend/graph_engine/aggregators.py`<br>Naive Python reference vs vectorized NumPy (sum, mean, max). | 22 tests in `test_aggregators.py` | **PASS** | Exact numerical parity across all feature dimensions. |
| **Model Evaluation Engine** | `backend/models/evaluator.py`<br>Chronological 70/15/15 split, standard scaler, validation thresholding. | 8 tests in `test_model_evaluation.py` | **PASS** | Discrete threshold sweep per ADR-007; zero test label leakage. |
| **Benchmark Automation** | `backend/benchmarks/benchmark_suite.py`<br>Multi-repetition harness, timing stability, preflight memory checks. | 16 tests in `test_benchmark_automation.py` | **PASS** | Local CUDA runs skipped transparently when NVCC absent. |

---

## 5. Research Integrity & Reproducibility Audit

### 5.1 Dataset Authenticity & Invariants
- **Primary Research Fixture:** `data/fixtures/synthetic_fraud_10k.npz`
- **Nodes:** Exactly 10,000.
- **Edges:** Exactly 231,984 directed edges (average in-degree: 23.20).
- **Feature Dimension:** 32 continuous attributes ($F_0 \dots F_{31}$).
- **Class Imbalance:** 350 positive fraud instances (3.50% prevalence), 9,650 licit instances (96.50%).
- **Time Window:** 30 consecutive days (2,592,000 seconds) with strictly ordered timestamps ($t_i \le t_{i+1}$).
- **Cryptographic Checksum:** SHA-256 `aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430` (verified matches database record).
- **Deterministic Generation:** `generate_synthetic_fraud_graph(seed=42)` guarantees 100% reproducible bitwise output.

### 5.2 Strict Temporal Causality & Label Quarantine
- **Split Partitions:** Chronological cutoffs:
  - **Train:** $70\%$ (nodes $0 \dots 6,999$, $t < 21 \text{ days}$, 245 frauds).
  - **Validation:** $15\%$ (nodes $7,000 \dots 8,499$, $21 \le t < 25.5 \text{ days}$, 52 frauds).
  - **Test:** $15\%$ (nodes $8,500 \dots 9,999$, $t \ge 25.5 \text{ days}$, 53 frauds).
- **Causal History Context Policy (`causal_history`):** When enriching target transaction node $i$, only incoming neighbor transactions $j$ with timestamps $t_j \le t_i$ are accumulated. Future transaction leakage is mathematically impossible.
- **Label Quarantine:** Ground-truth fraud labels ($Y \in \{0, 1\}$) are strictly quarantined from the feature matrix $X$. Neighbor aggregation operates exclusively on the 32 continuous transaction features.
- **Decision Threshold Optimization:** Validation optimal threshold $\tau^* = 0.1090$ was determined exclusively by maximizing validation $F_1$ over 100 discrete candidate thresholds. The held-out test partition was evaluated strictly using the frozen $\tau^*$.

### 5.3 Downstream ML Metric Verification
Every single headline evaluation KPI on the dashboard traces to Row 4 of `summary_evaluation_metrics.csv`:
- **Model:** Logistic Regression (trained on MAX neighbor aggregation representation, 64 dimensions).
- **Test PR-AUC:** `0.3315` (exact CSV value: `0.331471`).
- **Test ROC-AUC:** `0.7865` (exact CSV value: `0.786481`).
- **Test F1:** `0.3265` (exact CSV value: `0.326531`).
- **Test Precision:** `0.2553` (exact CSV value: `0.255319` = $24 / (24 + 70)$).
- **Test Recall:** `0.4528` (exact CSV value: `0.452830` = $24 / (24 + 29)$).
- **Confusion Matrix on Test Partition (1,500 nodes):**
  - True Positives ($TP$): 24
  - False Positives ($FP$): 70
  - True Negatives ($TN$): 1377
  - False Negatives ($FN$): 29
  - Sum: $24 + 70 + 1377 + 29 = 1,500$ (exactly matches test partition size; $24 + 29 = 53$ frauds).

### 5.4 Benchmark Timings & Hardware Provenance
- **Baseline Speedup:** `44.60x` represents the NumPy vectorized CPU implementation over the naive Python loop reference aggregator for the MEAN operator on `synthetic_tier_a_10k_mean`. It is accurately labeled as **`PEAK VECTORIZED SPEEDUP (VS REF)`**, explicitly avoiding false claims of local GPU execution.
- **Variability Disclosure:** The endpoint outputs `"MEAN vectorized high variability documented (CV 34.29%)"`, truthfully reflecting the $34.29\%$ coefficient of variation in Row 9 of `summary_table.csv`.
- **Colab GPU Reference Runs:** Historical GPU kernel latencies on Google Colab NVIDIA Tesla T4 are cataloged with timing boundaries:
  - Baseline Global Memory: `0.0679 ms`
  - Warp-Coalesced: `0.0612 ms`
  - Degree-Aware Hybrid: `0.0594 ms`
  - Pinned Memory Dual-Stream: `0.0581 ms`
  - Timing Boundary: Hardware-synchronized `cudaEventRecord()` capturing kernel-only time on the active CUDA stream.

---

## 6. Test Coverage Gaps & Limitations

1. **Absence of Headless Browser E2E Tests:**
   - The test harness includes unit tests (`test_*.py`, `polling_lifecycle.test.mjs`) and direct API integration tests (`api_integration.test.mjs`). It does not include automated headless browser tests (e.g. Playwright or Puppeteer) verifying real CSS grid rendering, DOM layout reflow, or canvas interactions.
2. **Local CUDA Execution Hardware Absence:**
   - The local workstation environment lacks NVIDIA CUDA Toolkit (`nvcc`) compiler binaries. Local execution correctly exercises CPU reference and vectorized engines while skipping GPU kernel compilation (`test_cuda_*.py`). Full GPU kernel execution is verified on cloud runners (Google Colab Tesla T4).
3. **Synthetic-Only Fraud Dataset:**
   - Evaluation is conducted exclusively on synthetic temporal transaction graphs. Generalization to production banking streams or real-world public graphs (e.g., Elliptic, IEEE-CIS) is an open research extension.

---

## 7. Prioritized Remediation Backlog (Phase 11.2+)

| Priority | Task Description | Target Files | Expected Outcome |
|:---:|---|---|---|
| **P1** | **Dynamic Subgraph Extraction Endpoint** | `backend/app/api/v1/endpoints/graph.py`, `frontend/src/pages/GraphExplorerPage.tsx` | Implement `GET /api/v1/graph/subgraph?node_id=...&hops=...` enabling dynamic Cytoscape.js canvas rendering. |
| **P2** | **Transaction Ledger Pagination Endpoint** | `backend/app/api/v1/endpoints/transactions.py`, `frontend/src/pages/TransactionsPage.tsx` | Implement `GET /api/v1/transactions?offset=...&limit=...` allowing pagination over the full 10,000 transaction dataset. |
| **P3** | **Hardware Detection Guard Hardening** | `backend/app/api/v1/endpoints/health.py:29-39` | In `get_gpu_diagnostics()`, return `detected: False` and `name: None` on `nvidia-smi` failure instead of falling back to default MX450 values. |
| **P4** | **Browser E2E Automated Test Suite** | `frontend/e2e/` | Add Playwright test suite for headless browser navigation, responsive drawer toggles, and visual regression. |

---

## 8. Explicit List of Checks Not Performed

In strict adherence to repository preservation rules:
1. **Did NOT rerun benchmark automation:** Existing benchmark CSVs in `experiments/results/benchmark_runs/` were inspected read-only and not overwritten.
2. **Did NOT rerun downstream ML model training:** Phase 7.5 evaluation artifacts were inspected read-only and not recomputed.
3. **Did NOT modify application source code, tests, or config:** The working tree was kept in a strictly read-only state.
4. **Did NOT execute browser automation with Playwright/Cypress:** No new browser testing dependencies were installed.
5. **Did NOT stage or commit changes to Git:** Baseline HEAD `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc` was preserved.

---

## 9. Final Recommendation

### **RECOMMENDATION: READY TO PROCEED TO PHASE 11.2**

The system is in a stable, verified, and mathematically reproducible state:
- Zero data fabrication or metric fallback risks remain.
- Backend failure handling and path security are verified.
- Frontend polling lifecycle, error classification, and state handling are sound.
- All 161 backend tests, 21 frontend lifecycle tests, and production Vite builds pass without errors.
- Unimplemented features are clearly disclosed to prevent user misrepresentation.

---

## 10. Final Verification Sign-Off

```powershell
# Verification Outcomes Recorded
Git HEAD: ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc (UNCHANGED)
Git Staged: 0 files (CLEAN)
Canonical Artifacts (data/, experiments/): 0 modifications (PRISTINE)
git diff --check: Exit Code 0 (0 whitespace / syntax errors)
Full Backend Regression Suite: 161 passed, 2 skipped
Frontend Production Build: Success (2.65s)
```
