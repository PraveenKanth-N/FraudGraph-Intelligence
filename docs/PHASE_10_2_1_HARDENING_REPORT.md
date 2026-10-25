# Phase 10.2.1 — Dashboard Label Correction and Regression Test Hardening Report

**Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Institution:** Vellore Institute of Technology, CSI4007 GPU Programming Capstone  
**Workspace Root:** `D:\FraudGraph Intelligence`  
**Expected Baseline HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Current Branch:** `main`  
**Date:** 2026-10-25  
**Execution Status:** COMPLETE — VERIFIED & HARDENED  
**Deliverable:** `docs/PHASE_10_2_1_HARDENING_REPORT.md`  

---

## 1. Executive Summary & Objective

Phase 10.2.1 addresses the findings and testing gaps identified during the independent review of Phase 10.2:
1. **Dynamic High-Variability Labeling:** Eliminated the legacy hardcoded `"SUM vectorized"` description and fallback constant `"189.0"` in `backend/app/api/v1/endpoints/dashboard.py`. The label is now derived dynamically from the selected high-variability CSV row's actual `operator`, `engine`, and `cv_percent` fields. Missing or malformed fields honestly declare unavailability rather than fabricating labels.
2. **Permanent Regression Test Coverage:** Extended `backend/tests/test_dashboard.py` with 6 new permanent test suites covering:
   - Non-finite evaluation literals (`nan`, `inf`, `-inf`) and negative probabilities (`-0.05`).
   - Non-positive and non-finite benchmark speedups (`-5.0x`, `0.0x`, `infx`).
   - Missing required evaluation target row (`causal_history,logistic_regression,max`).
   - Missing required benchmark workload row (`synthetic_tier_a_10k_mean,cpu_vectorized`).
   - Relative parent-directory traversal (`../../outside.csv`).
   - Dynamic operator and CV extraction (verifying `MEAN` vs `SUM`, and graceful handling of missing/invalid fields).

---

## 2. Implementation Details

### 2.1 Dynamic High-Variability Labeling (`backend/app/api/v1/endpoints/dashboard.py`)

Previously, line 455 contained:
```python
# PREVIOUS VULNERABLE LOGIC
high_var_rows = [r for r in reader if r.get("timing_stability") == "HIGH_VARIABILITY"]
if high_var_rows:
    cv_val = high_var_rows[0].get("cv_percent", "189.0")
    high_var_label = f"SUM vectorized high variability documented (CV {cv_val}%)"
else:
    high_var_label = "All measured workloads STABLE"
```
In the authentic smoke session artifact `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv`, the first row where `timing_stability == "HIGH_VARIABILITY"` is Row 9 (`synthetic_tier_a_10k_mean`), representing the `mean` operator under `cpu_vectorized` with a CV of `34.29%`. Hardcoding `"SUM vectorized"` resulted in inaccurate telemetry text (`SUM vectorized ... (CV 34.29%)`).

**Hardened Implementation:**
```python
high_var_rows = [r for r in reader if r.get("timing_stability") == "HIGH_VARIABILITY"]
if high_var_rows:
    selected_var_row = high_var_rows[0]
    raw_op = selected_var_row.get("operator", "").strip()
    raw_engine = selected_var_row.get("engine", "").strip()
    raw_cv = selected_var_row.get("cv_percent", "").rstrip("%").strip()

    op_valid = bool(raw_op)
    cv_valid = False
    if raw_cv:
        try:
            cv_num = float(raw_cv)
            if math.isfinite(cv_num) and cv_num >= 0.0:
                cv_valid = True
        except (ValueError, TypeError):
            pass

    if not op_valid and not cv_valid:
        high_var_label = "High variability documented (operator and CV unavailable)"
    elif not op_valid:
        high_var_label = f"High variability documented (operator unavailable, CV {raw_cv}%)"
    elif not cv_valid:
        engine_desc = raw_engine.replace("cpu_", "").strip()
        desc = f"{raw_op.upper()} {engine_desc}".strip()
        high_var_label = f"{desc} high variability documented (CV unavailable)"
    else:
        engine_desc = raw_engine.replace("cpu_", "").strip()
        desc = f"{raw_op.upper()} {engine_desc}".strip()
        high_var_label = f"{desc} high variability documented (CV {raw_cv}%)"
else:
    high_var_label = "All measured workloads STABLE"
```
**Key Invariants Enforced:**
- No hardcoded `"SUM vectorized"` prefix.
- Operator is derived from `operator` field (`raw_op.upper()`).
- Engine is derived from `engine` field (`raw_engine.replace("cpu_", "")`).
- CV is validated for numeric finiteness and non-negativity ($CV \ge 0.0$).
- Missing operator or invalid CV gracefully returns explicit `"unavailable"` descriptions rather than crashing or guessing.

---

### 2.2 Permanent Regression Test Additions (`backend/tests/test_dashboard.py`)

Six new permanent test functions were added to `backend/tests/test_dashboard.py`:

1. `test_dashboard_evaluation_artifact_non_finite_and_negative_metrics(tmp_path)`:
   - Evaluates temporary CSV fixtures containing `nan`, `inf`, `-inf`, and negative probability `-0.05`.
   - Confirms all values return HTTP 200 with status `"degraded"`, `artifact_state == "INVALID"`, and `top_test_pr_auc is None`.
2. `test_dashboard_benchmark_speedup_negative_zero_and_infinite(tmp_path)`:
   - Evaluates temporary CSV fixtures containing `-5.0x`, `0.0x`, and `infx`.
   - Confirms all values return HTTP 200 with status `"degraded"`, `artifact_state == "INVALID"`, `top_speedup_vs_cpu_ref is None`, and clear error diagnostics.
3. `test_dashboard_evaluation_target_row_missing(tmp_path)`:
   - Evaluates a CSV with valid headers and data rows, but omitting the target row (`causal_history,logistic_regression,max`).
   - Confirms `artifact_state == "INVALID"` and error diagnostic `"target evaluation row (causal_history, logistic_regression, max) not found"`.
4. `test_dashboard_benchmark_target_workload_row_missing(tmp_path)`:
   - Evaluates a CSV with valid headers and data rows, but omitting the target row (`synthetic_tier_a_10k_mean,cpu_vectorized`).
   - Confirms `artifact_state == "INVALID"` and error diagnostic `"target benchmark workload synthetic_tier_a_10k_mean (cpu_vectorized) not found"`.
5. `test_dashboard_relative_parent_path_traversal()`:
   - Tests `eval_csv_path=../../outside.csv` and `bench_csv_path=../../outside.csv`.
   - Confirms both queries return HTTP 400 Bad Request with detail `"escapes authorized workspace boundaries"`.
6. `test_dashboard_variability_label_derives_operator_and_cv(tmp_path)`:
   - **Case A (MEAN):** Verifies label is `"MEAN vectorized high variability documented (CV 34.29%)"`, with `"SUM"` absent.
   - **Case B (SUM):** Verifies label is `"SUM vectorized high variability documented (CV 189.0%)"`.
   - **Case C (Missing Operator):** Verifies label declares operator unavailable.
   - **Case D (Invalid CV):** Verifies label declares CV unavailable.

In addition, `test_dashboard_summary()` was updated to assert:
```python
assert "MEAN vectorized" in bm["high_variability_label"]
assert "34.29%" in bm["high_variability_label"]
assert "SUM" not in bm["high_variability_label"]
```

---

## 3. Verification & Test Gate Results

| Verification Suite | Target | Result | Details |
|---|---|:---:|---|
| **Focused Dashboard Tests** | `pytest backend/tests/test_dashboard.py -v` | **PASS** | **19 passed in 3.40s** (all 6 new hardening tests + 13 baseline tests) |
| **Complete Backend Regression Suite** | `pytest -v` | **PASS** | **161 passed, 2 skipped** (100% test pass rate across aggregators, CSR graph, evaluation, benchmarks, reporting, generator, and dashboard) |
| **Frontend TypeScript Type Check** | `npx tsc --noEmit` (in `frontend/`) | **PASS** | **0 errors, 0 warnings** |
| **Git Whitespace & Syntax Check** | `git diff --check` | **PASS** | **Exit code 0**, 0 whitespace errors |

---

## 4. Current Repository State

- **Baseline HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc` (unmodified).
- **Git Staging Policy:** 0 files staged (`git add`), 0 commits made, 0 pushes.
- **Modified Tracked Files (14):**
  - `backend/app/api/v1/endpoints/dashboard.py`
  - `backend/tests/test_dashboard.py`
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
- **Untracked Deliverables & Reports (16):**
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
  - `docs/PHASE_10_2_1_HARDENING_REPORT.md` (this report)
  - `frontend/src/hooks/`
  - `frontend/src/services/apiClient.ts`
  - `frontend/src/types/telemetry.ts`
  - `frontend/tests/`

---

## 5. Unresolved Limitations

1. **Artifact-Driven Headline Metrics:** Headline KPIs in `/api/v1/dashboard/summary` rely on precomputed evaluation and benchmark run CSVs located in `experiments/results/`. Real-time retraining or on-the-fly execution is not part of the dashboard summary endpoint design.
2. **CUDA Hardware Exemption on Host:** The host system (Windows / NVIDIA GeForce MX450 without local NVCC) skips CUDA GPU kernel execution as documented by design; all 24 GPU benchmark runs are reported as `SKIPPED_CUDA_UNAVAILABLE` with zero fabricated metrics.

---

## 6. Conclusion

Phase 10.2.1 is complete. The high-variability label is now dynamically derived from authentic CSV row fields with complete fallback honesty. The dashboard endpoint is fortified with permanent regression tests covering non-finite numbers, non-positive speedups, missing target rows, relative traversal, and dynamic label derivation. All 161 backend tests pass without error.
