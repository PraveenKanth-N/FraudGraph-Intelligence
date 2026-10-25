# Phase 10.1 — Dashboard Artifact Provenance and Release-Readiness Audit Report

**Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Institution:** Vellore Institute of Technology, CSI4007 GPU Programming Capstone  
**Workspace Root:** `D:\FraudGraph Intelligence`  
**Expected Baseline HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Current Branch:** `main`  
**Date:** 2026-10-25  
**Audit Status:** COMPLETE — READ-ONLY PROVENANCE & RELEASE-READINESS AUDIT  
**Deliverable:** `docs/PHASE_10_1_ARTIFACT_AUDIT.md`  

---

## 1. Executive Summary & Audit Verdict

### **VERDICT: PASS — RIGOROUS VERIFICATION CONFIRMS COMPLETE ARTIFACT PROVENANCE & FIDELITY**

This audit performed a thorough, read-only inspection of the data pipeline connecting repository evaluation/benchmark artifacts to the FastAPI backend and React frontend. Every displayed KPI metric, table entry, speedup ratio, and hardware label was traced directly to physical repository files.

### Key Audit Findings
1. **Flawless Mathematical Tracing:**
   - The headline evaluation metrics (`PR-AUC: 0.3315`, `ROC-AUC: 0.7865`, `F1: 0.3265`) were confirmed to originate from Line 4 of `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv`, representing the **Logistic Regression** classifier trained on the **MAX neighbor aggregation** representation (64 dimensions) under **causal history streaming**.
   - The headline speedup ratio (`44.60x`) was confirmed to originate from Line 9 of `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv`, representing the CPU vectorized NumPy implementation over the naive Python loop reference aggregator for the **MEAN operator** on `synthetic_tier_a_10k_mean`. It is accurately labeled in the UI as **`PEAK VECTORIZED SPEEDUP (VS REF)`**, explicitly avoiding false claims of local or universal GPU speedup.
2. **Strict Hardware Provenance Separation:**
   - Host GPU diagnostics (`/api/v1/health`) truthfully reflect local hardware (`NVIDIA GeForce MX450`, free VRAM, compiler status `UNAVAILABLE_LOCAL_COLAB_READY`).
   - Benchmark and evaluation KPIs are classified as `HISTORICAL_ARTIFACT` (`HISTORICAL`), ensuring recorded cloud benchmarks (Google Colab Tesla T4) are never presented as local execution measurements.
3. **Workspace Path Containment:**
   - `WORKSPACE_ROOT` traverses exactly five parent directories from `backend/app/api/v1/endpoints/dashboard.py`, cleanly anchoring file resolution to `D:\FraudGraph Intelligence`. Subpaths are hardcoded constants; no user-supplied parameters can escape the intended workspace.
4. **Audit Finding A-01 (Residual Risk & Proposed Remediation):**
   - In `backend/app/api/v1/endpoints/dashboard.py`, if `summary_evaluation_metrics.csv` or `summary_table.csv` is deleted or corrupted, the `try...except` block falls back to default constant variables (`top_pr = 0.3315`, `top_speedup = 44.60`) rather than raising a 500 error or marking status as degraded.
   - While these fallbacks currently mirror the true historical artifacts, they create a risk of silent stale reporting if artifacts are missing. A minimal remediation is proposed in Section 6.

---

## 2. Workspace Root Resolution & Path Containment

### Path Resolution Analysis
In `backend/app/api/v1/endpoints/dashboard.py` (line 12):
```python
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
```
- Location of `__file__`: `D:\FraudGraph Intelligence\backend\app\api\v1\endpoints\dashboard.py`
- Directory of `__file__`: `backend/app/api/v1/endpoints`
- Level 1 (`..`): `backend/app/api/v1`
- Level 2 (`..`): `backend/app/api`
- Level 3 (`..`): `backend/app`
- Level 4 (`..`): `backend`
- Level 5 (`..`): `D:\FraudGraph Intelligence` (Repository Root)

### Traversal Containment Check
- **Evaluation Artifact:** `os.path.join(WORKSPACE_ROOT, "experiments", "results", "evaluation_reporting_7_5", "summary_evaluation_metrics.csv")`
- **Benchmark Artifact:** `os.path.join(WORKSPACE_ROOT, "experiments", "results", "benchmark_runs", "session_20261024_230008_smoke", "summary_table.csv")`
- **Dataset Fixture Target Directory:** `os.path.join(WORKSPACE_ROOT, "data", "fixtures", file_name)`
- **Containment Verdict:** **SAFE.** All subpaths are static, hardcoded string literals within the repository root. Neither `dashboard.py` nor `health.py` accept external query parameters or path variables for file resolution. In `datasets.py`, the optional `base_dir` parameter only alters the directory prefix, while `file_name` is strictly drawn from a frozen whitelist (`CATALOG_FIXTURES`), preventing path traversal attacks.

---

## 3. Comprehensive Metric-to-Source Provenance Tracing

### 3.1 Evaluation Metrics Provenance Trace

| UI Label & Metric | Value Displayed | Source File | Source Row & Selector | Exact Physical Value in CSV | Mathematical Definition & Methodological Protocol |
|---|:---:|---|---|---|---|
| **Top Model PR-AUC** | `0.3315` | `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv` | Row 4: `context_policy == 'causal_history'`, `model_name == 'logistic_regression'`, `representation == 'max'` | `test_pr_auc`: `0.331471` (rounds to 0.3315) | Area under the Precision-Recall curve evaluated on the chronological test partition (1,500 transactions, 53 frauds) using trapezoidal integration. |
| **Top Model ROC-AUC** | `0.7865` | `summary_evaluation_metrics.csv` | Row 4 (same row) | `test_roc_auc`: `0.786481` (rounds to 0.7865) | Area under the Receiver Operating Characteristic curve evaluated on the chronological test partition. |
| **Top Model Test F1** | `0.3265` | `summary_evaluation_metrics.csv` | Row 4 (same row) | `test_f1`: `0.326531` (rounds to 0.3265) | Harmonic mean of Precision and Recall at the validation-optimal decision threshold $\tau^* = 0.1090$: $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$. |
| **Test Precision** | `0.2553` | `summary_evaluation_metrics.csv` | Row 4 (same row) | `test_precision`: `0.255319` | $TP / (TP + FP) = 24 / (24 + 70) = 24 / 94 = 0.255319...$ |
| **Test Recall** | `0.4528` | `summary_evaluation_metrics.csv` | Row 4 (same row) | `test_recall`: `0.45283` | $TP / (TP + FN) = 24 / (24 + 29) = 24 / 53 = 0.452830...$ |
| **Confusion Matrix ($TP, FP, TN, FN$)** | `24, 70, 1377, 29` | `summary_evaluation_metrics.csv` | Row 4 (same row) | `test_true_positive`: 24<br>`test_false_positive`: 70<br>`test_true_negative`: 1377<br>`test_false_negative`: 29 | Total test support: $24 + 70 + 1377 + 29 = 1,500$ transactions ($53$ frauds, $1,447$ licit). Perfectly matches the temporal test split. |
| **Optimal Decision Threshold ($\tau^*$)** | `0.1090` | `summary_evaluation_metrics.csv` | Row 4 (same row) | `val_optimal_threshold`: `0.109` | Selected exclusively on the validation set to maximize validation $F_1$, strictly quarantined from test labels. |
| **ADR-007 Notice** | Rendered in warning callout | `summary_evaluation_metrics.csv` / `DECISIONS.md` | ADR-007 Protocol | Verified in `test_reporting.py` | "Continuous PR curve omitted per ADR-007; discrete validation thresholding policy applied." |

### 3.2 Benchmark & Speedup Provenance Trace

| UI Label & Metric | Value Displayed | Source File | Source Row & Selector | Exact Physical Value in CSV | Physical Meaning & Attribution Fidelity |
|---|:---:|---|---|---|---|
| **Peak Vectorized Speedup** | `44.60x` (`44.6x`) | `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv` | Row 9: `workload_name == 'synthetic_tier_a_10k_mean'`, `operator == 'mean'`, `engine == 'cpu_vectorized'` | `speedup_vs_cpu_ref`: `44.60x` | Ratio of naive Python loop reference latency ($114.8598\text{ ms}$) to NumPy vectorized latency ($2.5756\text{ ms}$): $114.8598 / 2.5756 = 44.595\text{x}$. Labeled as **PEAK VECTORIZED SPEEDUP (VS REF)**. |
| **Top Speedup Workload** | `synthetic_tier_a_10k_mean (cpu_vectorized)` | `summary_table.csv` | Row 9 (same row) | `workload_name`: `synthetic_tier_a_10k_mean`<br>`engine`: `cpu_vectorized` | Honestly identifies the specific workload and engine producing the 44.60x speedup. |
| **Benchmark Session ID** | `session_20261024_230008_smoke` | `summary_table.csv` | Session directory name | Directory: `experiments/results/benchmark_runs/session_20261024_230008_smoke` | Provenance identifier of the recorded Phase 8 smoke execution session. |
| **Total Benchmark Runs** | `36` | `summary_table.csv` | Full table row count | Exactly 36 rows in CSV | 18 workloads for `synthetic_tier_a_10k` + 18 workloads for `toy_graph_5nodes`. |
| **Measured CPU Runs** | `12` | `summary_table.csv` | Rows with `measurement_state == 'MEASURED'` | 12 rows (6 reference + 6 vectorized) | 2 engines $\times$ 3 operators $\times$ 2 graph fixtures = 12 measured CPU runs. |
| **Skipped CUDA Runs** | `24` | `summary_table.csv` | Rows with `measurement_state == 'SKIPPED_CUDA_UNAVAILABLE'` | 24 rows | 4 CUDA engines (`cuda_baseline`, `cuda_coalesced`, `cuda_degree_aware`, `cuda_streams`) $\times$ 3 operators $\times$ 2 fixtures = 24 skipped GPU runs. |
| **Timing Stability Audit** | `CV 189.0%` | `summary_table.csv` | Row 14: `synthetic_tier_a_10k_sum`, `cpu_vectorized` | `cv_percent`: `189.0`, `timing_stability`: `HIGH_VARIABILITY` | Coefficient of Variation ($CV = \frac{\sigma}{\mu} = \frac{7.3154}{3.8706} \times 100\% = 189.0\%$). Accurately highlighted as `SUM vectorized high variability documented (CV 189.0%)`. |

---

## 4. Verification of UI Telemetry Attribution & Hardware Separation

| UI View & Component | Displayed Hardware / Telemetry Context | State Classification | Truthful Attribution Verification |
|---|---|:---:|---|
| **TopNav (`TopNav.tsx`)** | `LOCAL: GeForce MX450` or host probe state | `LIVE_ENDPOINT` (`AVAILABLE` / `DEGRADED`) | Directly displays `compute_engine.local_gpu_detected` returned by live probe. No claim of local CUDA acceleration. |
| **Diagnostics Drawer (`SystemHealthDrawer.tsx`)** | `NVIDIA GeForce MX450`<br>`COMPUTE CAPABILITY: 7.5`<br>`COMPILER STATUS: UNAVAILABLE_LOCAL_COLAB_READY`<br>`FREE VRAM: 1907.0 MB` | `LIVE_ENDPOINT` (`AVAILABLE`) | Binds to live `check_health()` probe. Accurately alerts user that local NVCC compiler is unavailable and cloud execution is ready. |
| **Overview Dashboard (`OverviewPage.tsx`)** | Telemetry Badge: `HISTORICAL ARTIFACTS`<br>KPI 3: `PEAK VECTORIZED SPEEDUP 44.6x`<br>Panel: `LOCAL HOST ACCELERATOR` vs `REMOTE BENCHMARK PLATFORM` | `HISTORICAL_ARTIFACT` (`HISTORICAL`) | Distinct separation between local MX450 host GPU and canonical remote Colab Tesla T4 platform ($0.0679\text{ ms}$ kernel latency). Zero false attribution. |
| **CUDA Lab (`CudaLabPage.tsx`)** | Card 1: `LOCAL HOST ACCELERATOR` (`MX450`, `UNAVAILABLE_LOCAL_COLAB_READY`)<br>Card 2: `CANONICAL RESEARCH PLATFORM` (`NVIDIA Tesla T4 - Colab Cloud Runner`) | `LIVE_ENDPOINT` + `HISTORICAL` | Explicit badge: `HISTORICAL BENCHMARK RIG`. Notes that local Windows host does not execute CUDA kernels and directs to Colab workflow. |
| **Experiment Manager (`ExperimentManagerPage.tsx`)** | Session: `session_20261024_230008_smoke`<br>Table: All CUDA engines marked `CUDA UNAVAILABLE` with latency `--` | `HISTORICAL` | Zero synthesized GPU latency figures. 24 GPU workloads explicitly marked as skipped due to absent local compiler. |
| **Evaluation Center (`EvaluationCenterPage.tsx`)** | Model comparison table (10 rows)<br>Top model: `Logistic Regression (max)` | `HISTORICAL` | Frozen benchmark evaluation results derived from Phase 7.5 CSV artifact. Interactive threshold slider ($\tau \in [0.01, 0.99]$) explores local parameters without faking test re-evaluation. |
| **Dataset Registry (`DatasetsPage.tsx`)** | 4 fixtures (`synthetic_fraud_10k`, `tier_a`, `skewed`, `toy`) with live SHA-256 and CSR dimensions | `LIVE_ENDPOINT` (`AVAILABLE`) | Live filesystem inspection. Badges dynamically compute `VERIFIED`, `CORRUPT`, or `MISSING`. Verified scope note warns against certifying external commercial ground truth. |

---

## 5. Review of Phase 10 Path-Resolution Fix and Regression Test

### The Remediation
- **File:** `backend/app/api/v1/endpoints/dashboard.py`
- **Before:** `WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))` (Resolved to `D:\FraudGraph Intelligence\backend`, missing `experiments/`)
- **After:** `WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))` (Resolves to `D:\FraudGraph Intelligence`, containing `experiments/`)

### The Regression Test
- **File:** `backend/tests/test_dashboard.py`
- **Test:** `test_dashboard_artifact_path_and_real_parsing`
- **Assertions:**
  1. `assert os.path.exists(os.path.join(WORKSPACE_ROOT, "experiments"))`
  2. `assert os.path.exists(eval_csv)`
  3. `assert os.path.exists(bench_csv)`
  4. `assert eval_summary.top_test_pr_auc == 0.3315`
  5. `assert eval_summary.top_test_roc_auc == 0.7865`
  6. `assert eval_summary.top_test_f1 == 0.3265`
  7. `assert bench_summary.total_runs == 36`
  8. `assert bench_summary.measured_cpu_runs == 12`
  9. `assert bench_summary.skipped_gpu_runs == 24`

---

## 6. Audit Finding A-01: Silent Fallback on Missing Artifacts (Residual Risk & Proposed Remediation)

### The Issue
In `backend/app/api/v1/endpoints/dashboard.py`:
- `_load_evaluation_summary()` initializes default values (`top_pr = 0.3315`, `top_roc = 0.7865`, `top_f1 = 0.3265`). If the CSV file is deleted or corrupt, the `except Exception:` block catches the failure and silently returns these numbers.
- `_load_benchmark_summary()` initializes default values (`total_runs = 36`, `measured = 12`, `skipped = 24`, `top_speedup = 44.60`). If the benchmark CSV is deleted or corrupt, it silently returns these numbers.

### Contrast with Dataset Endpoint
In `datasets.py`, if a fixture file is missing, it sets `is_available = False`, `size_bytes = 0`, `sha256 = ""`, and dimensions to `0`, allowing the UI to render `MISSING` badges and dashes (`—`).

### Proposed Remediation (For Future Phase / Release Gate)
Add an explicit artifact health indicator or raise an `HTTPException(status_code=500, detail="Repository evaluation artifact unreadable")` when artifacts cannot be located, or include an `artifact_state: "VERIFIED" | "MISSING"` flag in `EvaluationSummaryModel` and `BenchmarkSummaryModel`. When missing, the UI will display a degraded banner rather than reporting stale constants.

---

## 7. Verification Test Suite Execution Results

All verification commands required by the audit were executed on the unmodified codebase:

| Verification Suite | Command | Result | Timing |
|---|---|:---:|---|
| **Dashboard Pytest Tests** | `.\.venv\Scripts\pytest.exe backend/tests/test_dashboard.py -v` | **7 passed, 0 failed** | 1.71s |
| **Backend Reporting Tests** | `.\.venv\Scripts\pytest.exe backend/tests/test_health.py backend/tests/test_reporting.py -v` | **13 passed, 0 failed** | 69.25s |
| **Frontend TypeScript Check** | `npx tsc --noEmit` (in `frontend/`) | **0 errors, 0 warnings** | 3.52s |
| **Frontend Production Build** | `npm run build; git restore tsconfig.tsbuildinfo` | **Build Success (281.03 kB bundle)** | 1.98s |
| **Git Diff Whitespace Check** | `git diff --check` | **0 whitespace errors** | 0.12s |

---

## 8. Current Git HEAD & Working-Tree Status

```powershell
Set-Location 'D:\FraudGraph Intelligence'
git rev-parse HEAD
# Output: ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc (Phase 9B Commit)

git status --short
```

- **Modified Tracked Files (14):**
  - `backend/app/api/v1/endpoints/dashboard.py` (WORKSPACE_ROOT fix)
  - `backend/tests/test_dashboard.py` (Regression test)
  - `frontend/src/components/layout/AppLayout.tsx`
  - `frontend/src/components/layout/SystemHealthDrawer.tsx`
  - `frontend/src/components/layout/TopNav.tsx`
  - `frontend/src/pages/CudaLabPage.tsx`
  - `frontend/src/pages/DatasetsPage.tsx`
  - `frontend/src/pages/EvaluationCenterPage.tsx`
  - `frontend/src/pages/ExperimentManagerPage.tsx`
  - `frontend/src/pages/GraphExplorerPage.tsx`
  - `frontend/src/pages/OverviewPage.tsx`
  - `frontend/src/pages/TransactionsPage.tsx`
  - `frontend/src/services/api.ts`
  - `frontend/src/types/api.ts`
- **Untracked Deliverables & Reports (14):**
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
  - `frontend/src/hooks/`
  - `frontend/src/services/apiClient.ts`
  - `frontend/src/types/telemetry.ts`
  - `frontend/tests/` (contains `polling_lifecycle.test.mjs` and `api_integration.test.mjs`)
- **Staged Changes:** None (`git diff --cached` is empty).
- **History Operations:** Zero commits, zero pushes, zero resets/stashes.
- **Phase 11:** Unstarted. Standing by for review.
