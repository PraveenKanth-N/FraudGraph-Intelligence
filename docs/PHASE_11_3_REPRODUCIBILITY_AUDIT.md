# Phase 11.3 — Reproducibility and Research Integrity Audit Report

**Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Institution:** Vellore Institute of Technology, CSI4007 GPU Programming Capstone  
**Workspace Root:** `D:\FraudGraph Intelligence`  
**Expected Git HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Audit Date:** October 25, 2026  
**Audit Mode:** READ-ONLY (No code modified, no artifacts regenerated)  
**Deliverable File:** `docs/PHASE_11_3_REPRODUCIBILITY_AUDIT.md`  

---

## 1. Executive Summary & Audit Overview

This audit provides an independent, evidence-backed evaluation of the reproducibility and research integrity of **FraudGraph Intelligence (FGI)** across four foundational research pillars:
1. **Synthetic Dataset Generation & Provenance**
2. **Graph Construction & Temporal Integrity**
3. **Model Evaluation & Anti-Leakage Protocol**
4. **CUDA Benchmarking, Timing Boundaries & Hardware Detection**

### Summary Scorecard

| Area | Status | Key Evidence / Observations |
|---|:---:|---|
| **Synthetic Dataset Determinism** | **PASS** | `generate_synthetic_fraud_graph(seed=42)` reproduces 100% identical arrays and byte-for-byte identical archive SHA256 (`aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430`). |
| **Dataset Manifest Completeness** | **PASS with gaps** | Records seed (42), dimensions, entity keys, degree stats, and schemas; lacks explicit `generator_version` and embedded archive hash. |
| **Graph Temporal Causality** | **PASS** | Zero future edges ($t_j \le t_i$), zero self-loops; verified on all 231,984 edges of canonical fixture. |
| **CSR Structural Invariants** | **PASS** | Validated $row\_ptr$ monotonic offsets ($N=10,000, M=231,984$), $col\_idx < N$. |
| **Graph Label Quarantine** | **PASS** | Edge construction relies strictly on entity keys and timestamps; labels array is never accessed during graph building or BFS traversal. |
| **Chronological Split Isolation** | **PASS** | Continuous non-overlapping windows: Train $[0, 7000)$, Val $[7000, 8500)$, Test $[8500, 10000)$ with monotonic timestamps ($t_0 < t_1 < \dots < t_{9999}$). |
| **Train-Only Preprocessing** | **PASS** | `StandardScaler` fitted strictly on `X_train`; validation and test sets transformed using training statistics. |
| **Validation-Only Threshold Selection** | **PASS** | F1 and target-recall thresholds selected via grid search on `y_val` and `val_probs`, frozen before test evaluation. |
| **Model Metrics Agreement** | **PASS** | All 20 rows of `summary_evaluation_metrics.csv` match `fraud_model_evaluation_20261024_212032.json` with zero discrepancies; $TP+FP+TN+FN == 1500$. |
| **Benchmark Timing Traceability** | **PASS** | Kernel execution (`gpu_cuda_event`), transfers (H2D/D2H), and host timers (`cpu_host_timer`) are distinctly isolated. |
| **CUDA-Unavailable Handling** | **PASS** | 24 GPU workloads correctly recorded as `SKIPPED_CUDA_UNAVAILABLE` with `speedup_vs_cpu_ref = N/A` rather than fabricated. |
| **High-Variability Documentation** | **PASS** | Workloads with $CV > 30\%$ (up to $189.0\%$) are explicitly tagged `HIGH_VARIABILITY` and reported transparently in API. |
| **Hardware Detection Fallback** | **FAIL (HD-01)** | `backend/app/api/v1/endpoints/health.py` lines 29–39 catches `Exception` and hardcodes `"detected": True`, `"name": "NVIDIA GeForce MX450"`, reporting fake hardware if `nvidia-smi` is absent. |
| **Browser E2E Testing Scope** | **PASS (Clarified)** | Automated browser E2E test frameworks (Playwright/Cypress) do not exist; frontend verification relies on unit/client tests, TypeScript checks, and build validation. |

---

## 2. Pillar 1: Synthetic Dataset Reproducibility

### 2.1 Random Seed Control & Configuration
- **Code Reference:** `backend/cpu_engine/synthetic_fraud_generator.py` (Lines 38, 93, 218, 406).
- **Audit Verification:**
  - Random number generation is anchored to NumPy's modern Generator API: `rng = np.random.default_rng(seed)`.
  - Default seed is explicitly set to `42`.
  - Seed is configurable via function parameter and CLI argument (`--seed`).
  - Recorded in graph metadata: `metadata["seed"] = seed`.
- **Verdict:** **PASS**

### 2.2 Deterministic Reproduction Test
- **Methodology:** Generated a fresh synthetic graph in memory using `generate_synthetic_fraud_graph(seed=42)` and serialized to a temporary directory without modifying canonical files.
- **Empirical Array Verification:**
  - `row_ptr`: Array identical (`np.array_equal == True`). Shape `(10001,)`, `int32`.
  - `col_idx`: Array identical (`np.array_equal == True`). Shape `(231984,)`, `int32`.
  - `features`: Array identical (`np.array_equal == True`). Shape `(10000, 32)`, `float32`.
  - `labels`: Array identical (`np.array_equal == True`). Shape `(10000,)`, `int32`. Exactly 350 fraud nodes, 9,650 licit nodes.
  - `timestamps`: Array identical (`np.array_equal == True`). Shape `(10000,)`, `float64`.
  - **Archive Checksum Match:** The generated `.npz` file yielded SHA256:
    `aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430`.
    This matches the canonical archive checksum byte-for-byte.
- **Verdict:** **PASS**

### 2.3 Manifest Inspection & Provenance Gaps
- **Code Reference:** `backend/cpu_engine/synthetic_fraud_generator.py` (Lines 290–315).
- **Manifest Content in Canonical Fixture:**
  - `num_nodes`: `10000`
  - `num_edges`: `231984`
  - `feature_dim`: `32`
  - `is_directed`: `True`
  - `metadata`: Contains `seed: 42`, `target_fraud_rate: 0.035`, `time_span_days: 30.0`, `temporal_window_days: 7.0`, `relational_keys: ["card_id", "device_id"]`, and research disclaimer.
  - `fraud_stats`: `fraud_count: 350`, `licit_count: 9650`, `fraud_percentage: 3.5`.
  - `temporal_stats`: `min_timestamp_sec: 0.0`, `max_timestamp_sec: 2592000.0`.
- **Provenance Gaps Identified (`PV-01`):**
  1. *Generator Version Omission:* The manifest lacks an explicit generator version string (e.g. `generator_version: "1.0.0"`).
  2. *Self-Contained Checksum Omission:* The embedded JSON manifest does not contain the archive's SHA256 hash (it is tracked externally across endpoints and test files).
- **Verdict:** **PASS (With Documented Minor Provenance Gaps)**

---

## 3. Pillar 2: Graph Construction & Temporal Integrity

### 3.1 Causal Time Direction & Future Edge Prevention
- **Code Reference:** `backend/cpu_engine/synthetic_fraud_generator.py` (Lines 188–209, 245–254).
- **Construction Mechanics:** Transactions sharing `card_id` or `device_id` are partitioned and sorted by timestamp. For each transaction $i$ at timestamp $t_i$, prior transactions $j$ within the lookback window ($t_i - t_j \le 7.0\text{ days}$) form directed aggregation edges $(i, j)$.
- **Empirical Check on Canonical Fixture:**
  - Verified across all 231,984 edges:
    - Self-loops ($i == j$): **0** (Invariant satisfied).
    - Future-directed edges ($t_j > t_i$): **0** (Strict causality satisfied).
    - Unique timestamps: 10,000 unique values out of 10,000 transactions ($t_0 < t_1 < \dots < t_{9999}$).
- **Verdict:** **PASS**

### 3.2 CSR Structural Invariants
- **Code Reference:** `backend/cpu_engine/csr_graph.py` (Lines 100–165).
- **Invariant Verification:**
  - $row\_ptr[0] == 0$ and $row\_ptr[N] == M$ ($row\_ptr[10000] == 231,984$).
  - Monotonicity: $row\_ptr[i+1] \ge row\_ptr[i]$ for all $i \in [0, 9999]$.
  - Adjacency bounds: $0 \le col\_idx[p] < 10,000$ for all $p \in [0, 231983]$.
- **Verdict:** **PASS**

### 3.3 Graph Label Quarantine
- **Code Reference:** `backend/cpu_engine/synthetic_fraud_generator.py` (Lines 186–210) & `backend/app/api/v1/endpoints/graph.py` (Lines 142–177).
- **Verification:**
  - During graph generation, edge construction algorithms inspect only `card_ids`, `device_ids`, and `timestamps`. The `labels` array is never accessed.
  - In the subgraph extraction API (`graph.py`), BFS traversal expands neighbors solely from `row_ptr` and `col_idx`.
  - Mathematical isolation was confirmed in `backend/tests/test_graph_api.py` (`test_label_quarantine`), where inverted labels produced 100% identical graph topology.
- **Verdict:** **PASS**

### 3.4 Chronological Split Isolation & Context Semantics (ADR-007)
- **Code Reference:** `backend/cpu_engine/synthetic_fraud_generator.py` (Lines 367–396) & `backend/ml_engine/feature_enrichment.py` (Lines 120–185).
- **Chronological Partitions:**
  - Train: $[0, 7000)$ (7,000 transactions, 239 frauds = 3.41% prevalence, $t \in [0.0\text{s}, t_{6999}]$).
  - Validation: $[7000, 8500)$ (1,500 transactions, 58 frauds = 3.87% prevalence, $t \in [t_{7000}, t_{8499}]$).
  - Test: $[8500, 10000)$ (1,500 transactions, 53 frauds = 3.53% prevalence, $t \in [t_{8500}, 2592000.0\text{s}]$).
- **Context Policy Transparency:**
  - `causal_history`: Evaluates production streaming behavior where a test transaction at $t_i$ aggregates past historical neighbors $j < i$ from the shared transaction ledger (including historical training transactions).
  - `train_only_context`: Evaluates sensitivity under strict inductive isolation where test aggregation is artificially restricted to training-window nodes only.
  - Both policies are evaluated in parallel and reported in separate result tables.
- **Verdict:** **PASS**

---

## 4. Pillar 3: Model Evaluation Integrity

### 4.1 Preprocessing Isolation
- **Code Reference:** `backend/ml_engine/model_evaluation.py` (Lines 270–281).
- **Verification:**
  - `StandardScaler()` is fitted strictly on `X_train`:
    ```python
    scaler = StandardScaler()
    x_train_fit = scaler.fit_transform(x_train)
    x_val_trans = scaler.transform(x_val)
    x_test_trans = scaler.transform(x_test)
    ```
  - Validation and test splits are strictly out-of-sample transforms. Zero transductive feature leakage occurs.
- **Verdict:** **PASS**

### 4.2 Threshold Selection Isolation
- **Code Reference:** `backend/ml_engine/model_evaluation.py` (Lines 57–135, 314–316).
- **Verification:**
  - Decision threshold optimization is performed exclusively on validation data:
    `opt_f1_th, _ = select_optimal_f1_threshold(y_val, val_probs)`
    `rec_target_th = select_recall_target_threshold(y_val, val_probs, target_recall=0.50)`
  - The resulting thresholds are frozen before calculating test metrics. Test ground-truth labels are never used to tune decision boundaries.
- **Verdict:** **PASS**

### 4.3 Metric Agreement with Source Artifacts
- **Artifacts Compared:**
  - Source JSON: `experiments/results/fraud_model_evaluation_20261024_212032.json`
  - Reporting CSV: `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv`
- **Verification Results:**
  - Evaluated across all 20 rows (10 `causal_history` + 10 `train_only_context`):
    - `test_pr_auc`: Exact match ($\Delta < 10^{-6}$).
    - `test_roc_auc`: Exact match ($\Delta < 10^{-6}$).
    - `test_f1`: Exact match ($\Delta < 10^{-6}$).
    - Confusion Matrix ($TP, FP, TN, FN$): Exact integer match across all rows.
    - Confusion Matrix Sum: $TP + FP + TN + FN == 1500$ (exactly matches test split size).
- **Verdict:** **PASS**

### 4.4 Model Parameters & Research Generalization Claims
- **Artifact Inspection:**
  - `model_parameters`: Random seeds (`random_state: 42`) and hyperparameters (`C: 1.0`, `solver: lbfgs`, `n_estimators: 100`) are recorded for every model run.
  - Generalization Qualifier: The canonical evaluation JSON includes an explicit disclaimer:
    > *"Controlled synthetic benchmark fixture for evaluating graph-based neighbor aggregation. The 3.5% fraud prevalence reflects real-world class imbalance, but does NOT represent live banking fraud distributions or empirical records."*
  - The results are framed as synthetic algorithmic benchmarks rather than claims of live commercial fraud generalization.
- **Verdict:** **PASS**

---

## 5. Pillar 4: CUDA Benchmark Integrity

### 5.1 Timing Boundary Isolation
- **Artifact Inspection:** `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv` & `experiments/results/exp_gpu_baseline_tier_a_max_20261024_142037.json`.
- **Verification:**
  - In historical Colab GPU runs (`Tesla T4`), pure kernel time (`kernel_mean: 0.06766 ms`), host-to-device transfer (`h2d_transfer_mean: 0.5614 ms`), device-to-host transfer (`d2h_transfer_mean: 0.3344 ms`), and total end-to-end time (`end_to_end_mean: 0.9634 ms`) are distinctly measured and reported.
  - In `summary_table.csv`, the `timing_boundary` column explicitly tags `cpu_host_timer` vs. `gpu_cuda_event`.
- **Verdict:** **PASS**

### 5.2 CUDA-Unavailable Handling & Truth in Performance
- **Inspection:** `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv`.
- **Verification:**
  - On the local development host lacking an active CUDA compiler, all 24 GPU workloads were marked `measurement_state: SKIPPED_CUDA_UNAVAILABLE`.
  - Speedup vs. CPU reference was recorded as `N/A`. No synthetic or fabricated GPU metrics were reported.
  - The 12 CPU workloads were genuinely measured (`MEASURED`) using `cpu_host_timer`.
- **Verdict:** **PASS**

### 5.3 High-Variability Timing Documentation
- **Inspection:** In `summary_table.csv`, 3 workloads exhibited high latency variance ($CV > 30\%$):
  1. `synthetic_tier_a_10k_sum` (`cpu_vectorized`): $CV = 189.0\%$, speedup $25.93\times$.
  2. `toy_graph_5nodes_mean` (`cpu_vectorized`): $CV = 59.6\%$, speedup $0.25\times$.
  3. `synthetic_tier_a_10k_mean` (`cpu_vectorized`): $CV = 34.29\%$, speedup $44.60\times$.
- **Verification:**
  - All three rows are tagged `HIGH_VARIABILITY` in the `timing_stability` column.
  - Phase 10.2.1 dashboard endpoint derivation extracts the operator and CV directly from the row, returning an honest label (`"SUM vectorized high variability documented (CV 189.0%)"`), preventing misrepresentation as stable performance.
- **Verdict:** **PASS**

---

## 6. Pillar 5: Hardware Detection & Testing Scope Limitations

### 6.1 Hardware Detection Fallback Defect (`HD-01`)
- **Code Reference:** `backend/app/api/v1/endpoints/health.py` (Lines 29–39).
- **Inspection Findings:**
  ```python
  def get_gpu_diagnostics() -> dict:
      try:
          output = subprocess.check_output(
              ["nvidia-smi", "--query-gpu=..."], ...
          )
          ...
      except Exception:
          return {
              "detected": True,
              "name": "NVIDIA GeForce MX450",
              "driver_version": "581.83",
              "compute_capability": "7.5",
              "vram_total_mb": 2048.0,
              "vram_free_mb": 1907.0,
              "local_nvcc_available": False,
              "cuda_compiler_status": "UNAVAILABLE_LOCAL_COLAB_READY"
          }
  ```
- **Defect Analysis:**
  - When `nvidia-smi` raises an exception (e.g. on a CPU-only VM, inside a container without NVIDIA drivers, or on a non-NVIDIA machine), the `except Exception:` block catches the failure and returns `"detected": True` with hardcoded laptop GPU details (`"NVIDIA GeForce MX450"`, `2048 MB VRAM`).
  - **Severity:** HIGH. Can mislead users or automated checks into believing a physical NVIDIA GPU is present when it is absent.
- **Verdict:** **FAIL (Defect HD-01)**

### 6.2 Browser E2E Testing Classification
- **Inspection of `frontend/package.json` and `frontend/tests/`:**
  - Dependencies: Only `react`, `react-dom`, `react-router-dom`, `lucide-react`, `clsx`.
  - DevDependencies: `@types/*`, `@vitejs/plugin-react`, `typescript`, `vite`.
  - Test Frameworks Present: Node.js native test runner (`node:test`) executing `.mjs` scripts.
  - Test Frameworks Absent: No Playwright, Cypress, Puppeteer, Selenium, or Vitest.
- **Test Taxonomy:**
  1. *Backend Tests (178 tests):* Pytest executing in-memory algorithms and FastAPI `TestClient(app)`.
  2. *Frontend Unit/Client Tests (24 tests):* Node.js native runner (`node --test`) testing lifecycle timers, visibility, and API client request serialization.
  3. *API Integration Tests (1 test file):* `frontend/tests/api_integration.test.mjs` executing live HTTP requests from Node.js runtime against localhost:8000.
  4. *Actual Browser E2E Tests (0 tests):* **None exist.** No headless or headed browser DOM automation is implemented in the repository.
- **Verdict:** **PASS (Accurately Classified & Documented)**

---

## 7. Explicitly Unperformed Checks in This Audit

To preserve repository safety and honor read-only audit rules:
1. **No Hardware Benchmark Execution:** Did not execute live GPU or CPU benchmark runs that would overwrite `experiments/results/benchmark_runs/`.
2. **No Model Re-Training:** Did not execute full model training runs that would overwrite `experiments/results/fraud_model_evaluation_*.json`.
3. **No Headed Browser Testing:** Did not launch real browser window sessions (no browser automation framework is configured).
4. **No Code Fix Application:** Did not modify `backend/app/api/v1/endpoints/health.py` to fix defect `HD-01` during this audit pass.

---

## 8. Prioritized Remediation Backlog

| Priority | ID | Category | Description | Proposed Action |
|:---:|:---:|---|---|---|
| **P1** | **HD-01** | Hardware Detection | `health.py` lines 29–39 returns hardcoded `"detected": True`, `"NVIDIA GeForce MX450"` on `nvidia-smi` exception. | Update `except Exception:` in `get_gpu_diagnostics()` to return `"detected": False`, `"name": None`, `"vram_total_mb": None`, `"vram_free_mb": None`, `"cuda_compiler_status": "UNAVAILABLE"`. |
| **P2** | **PV-01** | Provenance | `synthetic_fraud_10k.npz` manifest lacks explicit `generator_version` string. | Update `save_labeled_graph` in `synthetic_fraud_generator.py` to record `generator_version: "1.0.0"` in future fixture generations. |
| **P3** | **TS-01** | Testing Scope | No automated browser E2E test suite (Playwright/Cypress) exists. | Evaluate adding lightweight Playwright browser tests in a future milestone if GUI CI/CD automation is required. |

---

## 9. Final Safety Verification

- **Git HEAD:** Verified unchanged at `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`.
- **Git Staging (`git status --short`):** Zero files staged.
- **Pre-existing Working-Tree Changes:** 14 modified files and all untracked docs/tests preserved.
- **Canonical Datasets & Experiments:** Both `data/` and `experiments/` remain 100% clean and untouched (`git status data experiments` clean).
- **Git Diff Check:** `git diff --check` passed with 0 conflict/whitespace errors.
- **Audit Stop Boundary:** Work concluded with report generation. Phase 11.4 has not been started.
