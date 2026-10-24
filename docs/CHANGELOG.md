# Changelog — FraudGraph Intelligence (FGI)

All notable changes to the FraudGraph Intelligence platform will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to Semantic Versioning.

---

## [Unreleased]

### Planned for Phase 9
- Stitch UI Design & Frontend Implementation (React + Vite + TypeScript application, Obsidian Cybernetics design system tokens, Cytoscape.js graph canvas, ECharts analytics).

---

## [0.8.0] - 2026-10-24 — Phase 8: Benchmark Automation & Experiment Management

### Added
- **Benchmark Engine Architecture (`backend/benchmark_engine/`):**
  - **Matrix & Configuration Specification (`benchmark_matrix.py`):** Defines formal taxonomy separating Workload Configurations ($W$), Implementation Runs ($E$), Repetitions ($R$), and Execution Status (`MEASURED`, `NOT_RUN`, `SKIPPED_CUDA_UNAVAILABLE`, `SKIPPED_OOM_GUARD`, `FAILED_TOLERANCE`, `FAILED_NON_FINITE`, `FAILED_SHAPE_MISMATCH`, `FAILED_EXECUTION`).
  - Predefined benchmark suite profiles: `smoke` (6 workloads, 36 planned runs, 720 repetitions), `scaling_features` (10 workloads, 60 planned runs), `scaling_graphs` (6 workloads, 36 planned runs), and `power_law_vs_uniform` (6 workloads, 36 planned runs).
  - **Operator-Specific Numerical Correctness Verifier (`benchmark_verifier.py`):** Replaces rigid absolute thresholds with operator-appropriate absolute and relative tolerances ($|Y_{\text{test}} - Y_{\text{ref}}| \le \text{atol} + \text{rtol} \times |Y_{\text{ref}}|$):
    - `mean`: $\text{atol} = 10^{-5}, \text{rtol} = 10^{-5}$
    - `max`: $\text{atol} = 10^{-5}, \text{rtol} = 0.0$
    - `sum`: $\text{atol} = 10^{-4}, \text{rtol} = 10^{-4}$ (reflecting FP32 accumulator drift on high-degree nodes)
    - Invariant: isolated nodes ($d_i = 0$) strictly verified to output exact zero vector $\vec{0}$.
    - Non-finite check: immediate rejection on `NaN` or `Inf`.
  - **Hardware Telemetry & Memory Guards (`benchmark_telemetry.py`):** Collects CPU model, core count, platform, Python/NumPy versions, Host RAM, GPU model, and VRAM via `nvidia-smi`.
    - Host-RAM preflight guard (80% safety margin) and GPU-VRAM preflight guard (85% safety margin) preventing out-of-memory crashes before execution. Workload-level preflight check executes before graph loading or allocation.
    - Fail-closed memory safety policy: When host RAM telemetry is missing or invalid, large or unverified workloads are strictly blocked (`BLOCKED_TELEMETRY_UNAVAILABLE`). Bounded smoke workloads (<=100 MB) are permitted under an explicit configurable fallback policy (`allow_bounded_fallback=True`) with explicit caution logging.
    - High-precision statistical calculations (mean, std with sample `ddof=1`, median, p95, min, max, cv%) with timing stability flagging (`is_stable`, `stability_label`).
    - Compulsory data traffic ($B_{\text{compulsory}}$) and algorithmic data traffic ($B_{\text{algorithmic}}$) modeling, distinguishing model-derived bandwidth estimates from measured hardware DRAM performance counters.
  - **Experiment Session & Provenance Manager (`experiment_manager.py`):** Manages isolated session directories (`experiments/results/benchmark_runs/session_{session_id}/`), atomic `.tmp` file serialization, cryptographic SHA-256 fixture checksumming, git environment capture, and structured session manifests supporting partial recovery and existing manifest loading.
  - **Benchmark Runner & Orchestrator (`benchmark_runner.py`):** Unified execution orchestrator caching Phase 4 CPU reference ground truth, distinguishing timing boundaries (`cpu_host_timer` vs `gpu_cuda_event`), recording actual physical graph dimensions with 100% fidelity, transparently bypassing CUDA runs on non-CUDA hosts with explicit notifications, and providing a clean CLI entrypoint.
  - **Benchmark Reporting Engine (`benchmark_report.py`):** Automatically produces machine-readable `summary_table.csv` and publication-quality `session_report.md`.
- **Automated Verification Test Suite (`backend/tests/test_benchmark_automation.py`):**
  - 15 dedicated unit and invariant tests covering workload matrix schema, profile counts, operator tolerances, zero-degree sentinels, non-finite rejection, shape mismatches, statistical moments, stability flagging, throughput, compulsory/algorithmic bandwidth, memory preflights across valid/unavailable/invalid telemetry, workload-level OOM guards, atomic writes, Colab ingestion schema compliance, transparent CUDA skipping, and fixture immutability.
- **Canonical Smoke Benchmark Artifacts (`experiments/results/benchmark_runs/session_20261024_230008_smoke/`):**
  - Completed smoke benchmark execution on Tier A ($N=10,000, M=100,000, D=64$) and Toy ($N=5, M=6, D=16$) graphs.
  - Demonstrates **44.59x speedup** for vectorized CPU `mean` (3.88M nodes/sec) and authoritative **25.92x speedup** for vectorized CPU `sum` (2.58M nodes/sec, based on all 20 unpruned repetitions) over reference CPU loop, with 100% numerical verification.
  - Generates full session manifest, 36 individual raw measurement JSON records, summary CSV, and Markdown report.
- **Authoritative Documentation:**
  - Authored comprehensive specification: [`docs/BENCHMARK_AUTOMATION.md`](file:///D:/FraudGraph%20Intelligence/docs/BENCHMARK_AUTOMATION.md).
  - Clarified statistical policy for Tier A vectorized sum timing variability: authoritative primary speedup is 25.92x; 19-run metric is strictly a secondary diagnostic.
  - Documented upstream Pytest warning resolution narrowing specifically to Starlette's deprecation warning from FastAPI.

---

## [0.7.3] - 2026-10-24 — Phase 7.5: Research-Grade Evaluation Reporting & Visualizations

### Added
- **Evaluation Reporting Engine (`backend/ml_engine/reporting.py`, `backend/ml_engine/__init__.py`):**
  - Implemented publication-grade evaluation reporting and visualization pipeline translating canonical Phase 7.4 evaluation artifact into multi-format figures, machine-readable CSV table, and cryptographic manifest.
  - Strict schema and data validation (`validate_evaluation_artifact`): enforces split sample counts, confusion matrix sums ($\text{TP}+\text{FP}+\text{TN}+\text{FN} = N$), and support alignment.
  - Full-precision CSV summary table generator (`generate_summary_csv`): produces 20-row summary covering all metric columns across `causal_history` and `train_only_context` policies.
  - Headless rendering engine: generates 7 publication figures in 300 DPI PNG, vector SVG, and vector PDF formats.
  - Cryptographic provenance tracking (`generate_manifest_and_metadata`): records input and output SHA-256 hashes, runtime environment metadata, and per-figure field mappings and interpretations.
- **Publication Figures Generated (`experiments/results/evaluation_reporting_7_5/`):**
  - `fig1_test_pr_auc_comparison`: Grouped bar comparison of test PR-AUC with minority prevalence baseline ($\pi \approx 0.0353$).
  - `fig2_test_roc_auc_comparison`: Test ROC-AUC comparison highlighting metric compression in class-imbalanced regimes.
  - `fig3_pr_curve_omission_notice`: Methodological notice documenting why continuous PR curves are omitted in accordance with scientific integrity safeguards (no continuous scores stored in summary artifact).
  - `fig4_confusion_matrices`: $2 \times 2$ grid of confusion matrix heatmaps at validation-frozen $\tau^*$ thresholds.
  - `fig5_operating_tradeoffs`: Detection rates (Precision, Recall, F1) and error volume counts (FP, FN).
  - `fig6_runtime_breakdown`: Three-panel profile of CPU enrichment, model training, and qualified hardware execution boundaries (explicitly non-equivalent host loop vs isolated CUDA kernel).
  - `fig7_context_policy_sensitivity`: Comparative sensitivity analysis of `causal_history` vs `train_only_context`.
- **Machine-Readable Experiment Artifacts:**
  - `summary_evaluation_metrics.csv` (SHA-256: `86523fd6d6f998602124ba101972eba15dc5b27bc97c852b3dee475dd6f77ba4`).
  - `manifest.json`: Dynamic verification manifest.
- **Automated Verification Test Suite (`backend/tests/test_reporting.py`):**
  - 11 dedicated unit and invariant tests verifying artifact parsing, schema error raising, metric reading, confusion matrix arithmetic, split count consistency, CSV fidelity, headless rendering, artifact immutability, checksum recording, omission notice compliance, and output determinism.
- **Comprehensive Documentation:**
  - Authored comprehensive evaluation report [`docs/PHASE_7_5_EVALUATION_REPORT.md`](file:///D:/FraudGraph%20Intelligence/docs/PHASE_7_5_EVALUATION_REPORT.md).
  - Updated [`docs/PHASE_STATUS.md`](file:///D:/FraudGraph%20Intelligence/docs/PHASE_STATUS.md) with Section 11.9 marking Phase 7.5 and overall Phase 7 complete.

---

## [0.7.2] - 2026-10-24 — Phase 7.4: Controlled Fraud Model Evaluation

### Added
- **Model Evaluation Engine (`backend/ml_engine/model_evaluation.py`, `backend/ml_engine/__init__.py`):**
  - Implemented reproducible downstream fraud classification evaluation pipeline comparing raw tabular features against graph-enriched representations (`mean`, `max`, `sum`, `multi_operator`).
  - Validation-only threshold selection protocol (`select_optimal_f1_threshold`, `select_recall_target_threshold`): freezes decision thresholds on validation partition predictions via a 100-step grid search, preventing test-set overfitting.
  - Strict leakage controls: `StandardScaler` fitted exclusively on training partition features ($X_{\text{train}}$), enforcing strict train-only preprocessing isolation; zero label leakage.
  - Comprehensive metric computation (`compute_metrics`): evaluates PR-AUC, ROC-AUC, threshold precision/recall/F1, confusion matrices, and precision at target recall (0.50), guarded against zero-division errors.
  - Timing instrumentation: captures microsecond-level timing breakdowns for feature enrichment ($t_{\text{enrich}}$), preprocessing ($t_{\text{prep}}$), training ($t_{\text{train}}$), and inference ($t_{\text{infer}}$).
  - Multi-cohort execution (`run_comprehensive_evaluation`): evaluates complete $2\text{ models} \times 5\text{ representations}$ matrix across both primary streaming (`causal_history`) and sensitivity (`train_only_context`) regimes.
- **Automated Verification Test Suite (`backend/tests/test_model_evaluation.py`):**
  - 8 dedicated unit and invariant tests verifying threshold optimization, zero-division handling, metric bounds, confusion matrix arithmetic, train-only scaler isolation, evaluation record schema, context policy invariance, and multi-model matrix execution.
- **Empirical Experiment Artifacts & Documentation:**
  - Machine-readable benchmark report saved at [`experiments/results/fraud_model_evaluation_20261024_212032.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/fraud_model_evaluation_20261024_212032.json) (SHA-256: `155304b7453b15dda1f103817e32471a2ffb2357de9cb298cc7b470d8d0efc90`).
  - Authored comprehensive evaluation report [`docs/PHASE_7_4_EVALUATION.md`](file:///D:/FraudGraph%20Intelligence/docs/PHASE_7_4_EVALUATION.md).
  - Updated [`docs/PHASE_STATUS.md`](file:///D:/FraudGraph%20Intelligence/docs/PHASE_STATUS.md) with Section 11.7 detailing Phase 7.4 metrics.

---

## [0.7.1] - 2026-10-24 — Phase 7.3: Graph Feature Enrichment Pipeline

### Added
- **Feature Enrichment Module (`backend/ml_engine/feature_enrichment.py`, `backend/ml_engine/__init__.py`):**
  - Implemented multi-operator feature enrichment engine supporting five representation formats:
    1. `raw`: Base transaction features $X \in \mathbb{R}^{N \times D}$.
    2. `mean`: $[X \,\|\, Y_{\text{mean}}] \in \mathbb{R}^{N \times 2D}$.
    3. `max`: $[X \,\|\, Y_{\text{max}}] \in \mathbb{R}^{N \times 2D}$.
    4. `sum`: $[X \,\|\, Y_{\text{sum}}] \in \mathbb{R}^{N \times 2D}$.
    5. `multi_operator` (alias `combined`): $[X \,\|\, Y_{\text{mean}} \,\|\, Y_{\text{max}}] \in \mathbb{R}^{N \times 3D}$.
  - Explicit graph context policy enforcement:
    - `causal_history`: Preserves full historical causal lookup ($t_j < t_i, j < i$) for streaming transaction simulation.
    - `train_only_context`: Prunes all edges pointing to validation or test nodes ($j \ge \text{train\_end}$), ensuring strict historical isolation where validation/test nodes aggregate strictly from training entities.
  - Backend dispatch interface supporting `cpu_reference`, `cpu_vectorized` (default), and native CUDA runners (`cuda_baseline`, `cuda_coalesced`, `cuda_streams`, `cuda_degree_aware`).
  - Strict preservation of float32 precision, zero-degree sentinels ($d_i = 0 \implies \vec{0}$), 100% finiteness, and complete label quarantine.
- **Automated Verification Test Suite (`backend/tests/test_feature_enrichment.py`):**
  - 16 comprehensive unit and invariant tests covering representation dimensions, concatenation ordering, CPU reference vs vectorized numerical parity ($\text{atol} \le 10^{-5}$), zero-degree sentinels, boundary/empty graph handling, argument validation, float32 finiteness, context policies, train/val/test partition splits, label quarantine, fixture immutability, and conditional live CUDA execution.
- **Documentation Deliverables:**
  - Authored comprehensive specification [`docs/FEATURE_ENRICHMENT.md`](file:///D:/FraudGraph%20Intelligence/docs/FEATURE_ENRICHMENT.md).
  - Updated [`docs/PHASE_STATUS.md`](file:///D:/FraudGraph%20Intelligence/docs/PHASE_STATUS.md) with Section 11.5 marking Phase 7.3 complete.

---

## [0.7.0] - 2026-10-24 — Phase 7.1 & 7.2: Dataset Audit and Deterministic Labeled Synthetic Fraud Fixture

### Added
- **Synthetic Fraud Generator (`backend/cpu_engine/synthetic_fraud_generator.py`):**
  - High-performance deterministic transaction graph generator producing labeled fixtures with controlled fraud prevalence and non-trivial graph patterns.
  - Data generating process modeling Poisson transaction timestamps, Zipfian credit card and device distributions, and 20 coordinated fraud rings (245 transactions) alongside 105 opportunistic frauds ($N=10,000$, $M=231,984$, $D=32$, 350 positive fraud labels = exactly 3.50% prevalence).
  - Subtle feature space separation with bounded single-feature Pearson correlation ($r \approx 0.120 < 0.35$), ensuring task cannot be trivially solved by univariate tabular heuristics.
  - Temporal lookback sliding window ($W=100$ transactions) enforcing strict causal ordering ($t_j < t_i$, $j < i$) directed from query transaction $i$ to historical neighbor $j$, preventing future-to-past temporal leakage across all 231,984 edges.
  - Serialization functions `save_labeled_graph()` and `load_labeled_graph()` outputting structured `.npz` archives with `row_ptr`, `col_idx`, `features`, `labels`, `timestamps`, and JSON `manifest`.
  - Chronological dataset splitting function `temporal_train_test_split()` producing strictly non-overlapping train (70%, 239 frauds), validation (15%, 58 frauds), and test (15%, 53 frauds) partitions.
- **Labeled Fixture (`data/fixtures/synthetic_fraud_10k.npz`):**
  - Generated deterministic canonical fixture ($N=10,000, M=231,984, D=32$, 2.14 MB, seed=42) with full backward compatibility with standard `CSRGraph.load()`.
- **Automated Verification Test Suite (`backend/tests/test_synthetic_fraud_generator.py`):**
  - 11 comprehensive tests validating determinism, seed divergence, exact fraud prevalence (3.50%), feature matrix invariants, 8 structural CSR invariants, temporal causality ($t_j < t_i$), label leakage quarantine, round-trip serialization, `CSRGraph.load()` backward compatibility, chronological partition splits, and Phase 3 & 6 fixture preservation (`test_existing_graph_fixtures_remain_unchanged`).
  - Recorded reproducible preliminary tabular classifier verification on canonical split (Logistic Regression: ROC-AUC 0.7855, PR-AUC 0.3088; Random Forest: ROC-AUC 0.7786, PR-AUC 0.2445 vs random baseline 0.0353) strictly labeled as synthetic verification metrics.
- **Documentation Deliverables:**
  - Authored comprehensive specification [`docs/SYNTHETIC_FRAUD_DATASET.md`](file:///D:/FraudGraph%20Intelligence/docs/SYNTHETIC_FRAUD_DATASET.md).
  - Updated [`docs/PHASE_STATUS.md`](file:///D:/FraudGraph%20Intelligence/docs/PHASE_STATUS.md) with Section 11 documenting Phase 7.1 audit findings and Phase 7.2 fixture statistics.

---

## [0.6.2] - 2026-10-24 — Phase 6, Experiment 6.3: Degree-Aware Bucketing & Hybrid CUDA Dispatch

### Added
- **Degree-Aware Hybrid CUDA Engine (`backend/gpu_engine/cuda/`):**
  - Implemented `fgi_degree_aware_kernels.cuh` and `fgi_degree_aware_kernels.cu` introducing a 3-way topological dispatch architecture:
    1. *Zero-Degree Bucket ($d_i = 0$):* Direct zero-initialization via asynchronous `cudaMemsetAsync` on stream, setting exact FGI $\vec{0}$ sentinels with 0 instruction overhead.
    2. *Standard-Degree Bucket ($1 \le d_i \le 32$):* Dedicated `standard_degree_warp_kernel` mapping 1 warp per node, eliminating intra-block warp retirement stalls across bounded degrees.
    3. *Hub-Degree Bucket ($d_i > 32$):* Dedicated cooperative block kernel (`hub_degree_block_kernel`, 256 threads = 8 warps) interleaving neighbor lists with shared-memory reduction for `MEAN` and `MAX`, and strictly ordered sequential accumulation for `SUM` to preserve exact bitwise parity ($0.00\text{e}+00$ error) against single-threaded reference arithmetic.
  - Implemented standalone CLI runner `main_degree_aware.cu` with pinned memory, explicit bucketing timing measurement ($t_{\text{bucketing}}$ included in end-to-end latency), and structured JSON export conforming to `EXP-GPU-DEGREE-AWARE-OPT-6.3`.
  - Added `fgi_degree_aware_runner` build target to `Makefile` and `kernel="degree_aware"` adapter to `cuda_runner.py`.
- **Heavy-Tailed Skewed Graph Fixture (`data/fixtures/synthetic_skewed_10k.npz`):**
  - Generated deterministic heavy-tailed power-law graph ($N=10,000, M=94,970, D=32$) containing a 6,340-edge super-hub and 56 nodes with degree $> 32$.
- **Automated Verification Test Suite (`backend/tests/test_cuda_degree_aware.py`):**
  - 8 comprehensive tests validating source tokens, Makefile multi-target build, runner interface, bundle packaging, JSON schema conformity, partition invariants, CPU reference parity, and extreme hub stress testing.
- **Physical GPU Benchmark Results on NVIDIA Tesla T4 (Driver 580.82.07, CUDA 13.0, Cohort `20261024_142037`):**
  - 100% numerical verification across all 12 toy checks and all 24 Tier A and Skewed graph benchmark runs ($\text{atol} \le 1.0 \times 10^{-5}$, max absolute error $\le 1.19 \times 10^{-7}$).
  - **Skewed Graph Acceleration (`synthetic_skewed_10k`, max degree 6,340, 56 hubs):**
    - `MEAN`: Kernel latency dropped from $0.220\text{ ms} \rightarrow \mathbf{0.065\text{ ms}}$ (**$3.39\times$ faster than Streams**, **$107.71\times$ faster than Baseline**); E2E latency dropped to $\mathbf{0.396\text{ ms}}$ (**$1.22\times$ faster than Streams**, **$19.92\times$ faster than Baseline**).
    - `MAX`: Kernel latency dropped from $0.234\text{ ms} \rightarrow \mathbf{0.111\text{ ms}}$ (**$2.12\times$ faster than Streams**, **$65.21\times$ faster than Baseline**); E2E latency dropped to $\mathbf{0.443\text{ ms}}$ (**$1.14\times$ faster than Streams**, **$18.37\times$ faster than Baseline**).
  - **Explicit Empirical Regressions Disclosed (Negative Findings):**
    - `SUM` on Skewed Graph: Kernel latency regressed from $0.212\text{ ms}$ (Streams) to $0.464\text{ ms}$ ($0.46\times$ speedup / $2.19\times$ slowdown; E2E $0.803\text{ ms}$ vs $0.498\text{ ms}$, $0.62\times$ / $1.61\times$ slowdown) because float32 non-associative parallel tree reordering accumulated $3.97 \times 10^{-4}$ error in Run `141134`, requiring Warp 0 serial accumulation to preserve exact $0.00\text{e}+00$ reference parity, which left 7/8 warps idle in the 256-thread block.
    - Tier A End-to-End Cases: All 3 operators regressed end to end ($0.343\text{ ms}$ vs $0.281\text{ ms}$ for Mean, $0.396\text{ ms}$ vs $0.283\text{ ms}$ for Sum, $0.383\text{ ms}$ vs $0.278\text{ ms}$ for Max; $22\% - 40\%$ slowdown) due to non-amortizable host degree bucketing overhead ($t_{\text{bucketing}} = 0.053 - 0.109\text{ ms}$) on topologically uniform graphs lacking hub nodes.
- **Documentation Deliverables:**
  - Authored comprehensive report [`docs/EXPERIMENT_6_3_DEGREE_AWARE_DISPATCH.md`](file:///D:/FraudGraph%20Intelligence/docs/EXPERIMENT_6_3_DEGREE_AWARE_DISPATCH.md).
  - Updated [`docs/PHASE_STATUS.md`](file:///D:/FraudGraph%20Intelligence/docs/PHASE_STATUS.md) marking Phase 6 complete.

---

## [0.6.1] - 2026-10-24 — Phase 6, Experiment 6.2: Pinned Host Memory & Asynchronous CUDA Streams

### Added
- **Pinned Host Memory & Asynchronous Streams CUDA Runner (`backend/gpu_engine/cuda/`):**
  - Implemented standalone CLI driver [`main_streams.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_streams.cu) using page-locked host memory (`cudaMallocHost` / `cudaFreeHost`) to enable direct PCIe DMA transfers without OS pageable bounce buffers.
  - Asynchronous transfer and kernel execution sequencing via dedicated non-blocking CUDA stream (`cudaStreamNonBlocking`, `cudaMemcpyAsync`).
  - Strict preservation of baseline (`fgi_baseline_kernels.cu`, `main_baseline.cu`) and Phase 6.1 (`fgi_coalesced_kernels.cu`, `main_coalesced.cu`) sources.
  - Multi-target build support in [`Makefile`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/Makefile) compiling `fgi_baseline_runner`, `fgi_coalesced_runner`, and `fgi_streams_runner` for Tesla T4 (`compute_75,code=sm_75`).
- **Python Adapter & Colab Workflow Integration (`backend/gpu_engine/`):**
  - Added `kernel="streams"` support to [`cuda_runner.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda_runner.py).
  - Updated [`package_colab_bundle.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/package_colab_bundle.py) to package all three runners into `experiments/colab_bundle_fgi_phase6.zip` (3.56 MB).
  - Updated [`run_colab_benchmarks.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/run_colab_benchmarks.py) and [`fgi_colab_workflow.ipynb`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb) for 3-way comparative benchmarking.
- **Automated Verification Test Suite (`backend/tests/test_cuda_streams.py`):**
  - 8 comprehensive tests validating source tokens, Makefile targets, runner interface, bundle packaging, schema conformity, PCIe bandwidth math, ingested toy validations, and measured Tier A benchmark results.
  - Full backend test suite passing 72 tests (1 live GPU test appropriately skipped).
- **Physical GPU Benchmark Results on NVIDIA Tesla T4 (Driver 580.82.07, CUDA 13.0):**
  - Numerical correctness passed on all 9 toy-graph checks and all 9 Tier A benchmarks ($N=10,000, M=94,945, D=32$) with error $0.00 \le 1.0 \times 10^{-5}$ across Mean, Sum, and Max.
  - H2D transfer time dropped from $0.538 - 0.551\text{ ms}$ to $0.156 - 0.160\text{ ms}$ (**$3.41\times - 3.53\times$ speedup**).
  - D2H transfer time dropped from $0.317 - 0.350\text{ ms}$ to $0.105\text{ ms}$ (**$3.02\times - 3.34\times$ speedup**).
  - Measured PCIe bandwidth: **$10.66 - 10.87\text{ GB/s}$ H2D**, **$12.11 - 12.20\text{ GB/s}$ D2H** effective payload transfer throughput, compared to $2.95 - 3.16\text{ GB/s}$ (H2D) and $3.65 - 4.08\text{ GB/s}$ (D2H) under pageable host memory.
  - Pure kernel speedup vs baseline maintained at **$3.88\times - 4.06\times$** (invariant relative to Phase 6.1 coalesced, operator deltas between $0.28\%$ and $0.82\%$, delta $\le 0.00014\text{ ms}$).
  - **End-to-End Wall-Clock Speedup:** **$3.35\times - 3.44\times$ vs Baseline**, **$3.22\times - 3.27\times$ vs Coalesced 6.1** ($0.94 - 0.97\text{ ms} \rightarrow 0.28\text{ ms}$).
  - Ingested 18 verified JSON files into `experiments/results/` and preserved archive at `experiments/fgi_phase6_streams_gpu_results_20261024_114748.zip`. Phase 6.2 gate passed.
- **Documentation Deliverables:**
  - Authored [`docs/EXPERIMENT_6_2_PINNED_MEMORY_STREAMS.md`](file:///D:/FraudGraph%20Intelligence/docs/EXPERIMENT_6_2_PINNED_MEMORY_STREAMS.md).
  - Updated [`docs/PHASE_STATUS.md`](file:///D:/FraudGraph%20Intelligence/docs/PHASE_STATUS.md) with Section 10.5 and updated Phase 6 gate status.

---

## [0.6.0] - 2026-10-19 — Phase 6, Experiment 6.1: Warp-Coalesced Neighbor Aggregation

### Added
- **Optimized Warp-Coalesced CUDA Engine (`backend/gpu_engine/cuda/`):**
  - Implemented `aggregate_coalesced_warp_kernel` in [`fgi_coalesced_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_coalesced_kernels.cu) mapping 1 warp (32 lanes) per output node, achieving 100% coalesced 128-byte global memory transactions for canonical feature dimension $D=32$.
  - Column-strided edge loop ensuring generalized compatibility for arbitrary feature dimensions ($D \ne 32$).
  - Full compliance with FGI zero-sentinel policy for empty neighborhoods and negative-only neighborhoods for Max aggregation.
  - Implemented standalone CLI driver [`main_coalesced.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_coalesced.cu) with standard comma formatting and segregated analytical bandwidth calculation reporting.
  - Updated Makefile to compile both `fgi_baseline_runner` and `fgi_coalesced_runner` targeting Tesla T4 `compute_75,code=sm_75`.
- **Google Colab Cloud Revalidation & Automation (`backend/gpu_engine/colab_runner/`):**
  - Updated [`package_colab_bundle.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/package_colab_bundle.py) producing `experiments/colab_bundle_fgi_phase6.zip` (3.56 MB).
  - Turnkey execution and verification script [`run_colab_benchmarks.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/run_colab_benchmarks.py) and notebook [`fgi_colab_workflow.ipynb`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb).
  - `#include <cstdint>` and `std::uint64_t` repair ensuring clean compilation on GCC / CUDA 13.0 (V13.0.88).
- **Automated Verification Test Suite (`backend/tests/test_cuda_coalesced.py`):**
  - 9 comprehensive unit and schema tests validating source tokens, Makefile targets, runner interface, bundle packaging, schema validation, canonical Tier A fixture, bandwidth formulas, throughput formatting, and ingested result assertions.
  - Backend regression test suite passes 64 tests (1 live GPU test appropriately skipped).
- **Physical GPU Revalidation Results on NVIDIA Tesla T4 (Driver 580.82.07, CUDA 13.0):**
  - Revalidation run (`20261019_175441`) confirmed 100% numerical correctness ($\text{err} = 0.00 \le 1.0 \times 10^{-5}$) across toy and 10K node graphs for all operators.
  - Pure kernel acceleration: Mean $3.72\times$ ($0.0689\text{ ms} \rightarrow 0.0185\text{ ms}$), Sum $4.07\times$ ($0.0686\text{ ms} \rightarrow 0.0168\text{ ms}$), Max $4.03\times$ ($0.0673\text{ ms} \rightarrow 0.0167\text{ ms}$).
  - End-to-end wall-clock analysis: Sum $1.09\times$, Max $1.07\times$, Mean $0.99\times$ (accounting for PCIe transfer overhead occupying $93\%-98\%$ of execution time).
  - Disclosed analytical bandwidth estimates ($160.77 - 178.37\text{ GB/s}$ compulsory DRAM traffic, $747 - 829\text{ GB/s}$ algorithmic work rate hitting the 4MB on-chip L2 cache).
  - Ingested 12 verified JSON files into `experiments/results/` while preserving Run 1 evidence. Phase 6.1 gate passed.

---

## [0.5.0] - 2026-10-18 — Phase 5: Baseline CUDA Implementation & Colab Workflow

### Added
- **Native Baseline CUDA C++ Engine (`backend/gpu_engine/cuda/`):**
  - Implemented `aggregate_baseline_kernel` in [`backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu) with 1 thread mapped per output node.
  - Supports Sum, Mean, and Max aggregation operators in full compliance with the FGI mathematical specification.
  - Explicit empty-neighborhood handling writing $\vec{0}$ for isolated nodes ($d_i = 0$), preventing division-by-zero in Mean and $-\infty$ propagation in Max.
  - First-neighbor initialization for Max aggregation to correctly handle negative-only neighborhoods.
  - Created [`backend/gpu_engine/cuda/include/fgi_cuda_common.cuh`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/include/fgi_cuda_common.cuh) with robust `CUDA_CHECK` and `CUDA_CHECK_KERNEL` error handling macros.
  - Implemented standalone CLI driver executable in [`backend/gpu_engine/cuda/src/main_baseline.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_baseline.cu) with 64-bit safe allocations, hardware CUDA event timing, decoupled transfer metrics ($t_{\text{H2D}}$, $t_{\text{kernel}}$, $t_{\text{D2H}}$, $t_{\text{e2e}}$), and machine-readable JSON export.
  - Authored multi-architecture `Makefile` targeting SM 7.0, 7.5, and 8.0.
- **Google Colab Cloud Execution Adapter (`backend/gpu_engine/colab_runner/`):**
  - Implemented [`package_colab_bundle.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/package_colab_bundle.py) producing `experiments/colab_bundle_fgi_phase5.zip` (3.55 MB) containing CUDA sources, headers, Makefile, raw binary fixtures (`toy_graph_5nodes`, `synthetic_tier_a_10k`), and precomputed Phase 4 CPU references.
  - Implemented automated Colab runner script [`run_colab_benchmarks.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/run_colab_benchmarks.py) enforcing a two-stage correctness gate before benchmark execution.
  - Created interactive Jupyter notebook [`fgi_colab_workflow.ipynb`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb) for turnkey execution on Colab T4/A100 runtimes.
  - Created [`ingest_colab_results.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/ingest_colab_results.py) for validating and archiving downloaded benchmark JSON results into `experiments/results/`.
  - Authored comprehensive workflow guide in [`backend/gpu_engine/colab_runner/README.md`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/README.md).
- **Host-Side Python Adapter (`backend/gpu_engine/cuda_runner.py`):**
  - Python interface exposing `execute_cuda_aggregation`, `is_nvcc_available`, and `validate_gpu_against_cpu`.
  - Graceful fallback raising actionable runtime errors pointing to the Colab workflow when local compilation tools are absent.
- **Automated Verification Test Suite (`backend/tests/test_cuda_baseline.py`):**
  - 12 comprehensive unit tests covering source completeness, Colab bundle packaging, notebook JSON structure, JSON schema validation, numerical comparison logic with tolerance ($\text{atol} \le 10^{-5}$), and graceful missing-binary handling.
  - Total backend test suite now passes 55 tests (with 1 live GPU test appropriately skipped).
- **Empirical GPU Benchmarking & Verification on NVIDIA Tesla T4:**
  - Validated baseline 1-thread-per-node CUDA kernel on physical Tesla T4 GPU (40 SMs, 14.9 GB VRAM) compiled with CUDA compilation tools release 13.0, V13.0.88.
  - Correctness verification on `toy_graph_5nodes` achieved zero absolute error ($\text{atol} = 0.0 \le 10^{-5}$) across Sum, Mean, and Max.
  - Scale benchmark on `synthetic_tier_a_10k` ($N=10,000, M=94,945, D=32$) demonstrated $1.39\text{B edges/sec}$ throughput at $\sim 0.068\text{ ms}$ kernel latency ($203.4\text{ GB/s}$ effective bandwidth).
  - Ingested 6 verified JSON telemetry archives into `experiments/results/` meeting all schema criteria. Phase 5 gate successfully passed.


---

## [0.4.0] - 2026-10-18 — Phase 4: CPU Reference Aggregation Engine

### Added
- **CPU Reference Aggregation Engine (`backend/cpu_engine/aggregators.py`):**
  - Implemented Sum, Mean, and Max neighbor aggregation functions accepting validated `CSRGraph`.
  - Implemented dual execution architectures: straightforward node-iterative reference (`method="reference"`) and high-performance SciPy SpMM sparse-dense multiplication (`method="vectorized"`).
  - Explicit zero-sentinel handling: empty neighborhoods ($d_i = 0$) strictly return a zero vector ($\vec{0} \in \mathbb{R}^D$) across all operators, preventing division-by-zero or $-\infty$ propagation in Max aggregation.
  - Guaranteed single-precision floating-point arithmetic (`np.float32`) and input array immutability.
- **CPU Benchmark Harness (`backend/cpu_engine/benchmark_cpu.py`):**
  - High-precision benchmarking using monotonic nanosecond timer (`time.perf_counter_ns`).
  - Warm-up and repeated trials measuring mean, standard deviation, median (p50), 95th percentile, min, and max latencies.
  - Throughput calculations (nodes/sec and directed CSR adjacency entries/sec).
  - Cross-verification between reference and vectorized outputs with maximum absolute error checking ($\text{atol} \le 10^{-5}$).
  - Automatic JSON export under `experiments/results/` marked with state `MEASURED`.
- **Comprehensive Unit & Correctness Test Suite (`backend/tests/test_aggregators.py`):**
  - 22 dedicated test cases covering all 18 core verification requirements from the specification.
  - Verified hand-calculated 5-node toy fixture across Sum, Mean, and Max.
  - Verified isolated nodes, empty graphs, single-node graphs, asymmetric directed edges, symmetric undirected edges, and star graphs with 500 peripheral nodes.
  - Total backend test suite now passes 44/44 tests cleanly in 2.36s.
- **Comprehensive Architecture Documentation (`docs/CPU_REFERENCE_ENGINE.md`):** Complete mathematical definitions, algorithmic details, tolerance standards, and benchmark results.

### Fixed
- **Resolved Phase 3 Fixture Inconsistencies (`data/fixtures/`):**
  - Corrected `toy_graph_5nodes.npz`: explicitly verified that Node 4 is the isolated node (out-degree 0), while Node 0 has degree 2 (connected to 1 and 2).
  - Clarified edge count semantics: 8 directed CSR adjacency entries representing 4 undirected edge pairs.
  - Populated `toy_graph_5nodes.npz` with distinct, non-zero `float32` features (positive, negative, and zero values) to enable meaningful numerical verification.
  - Corrected `synthetic_tier_a_10k.npz` feature dimension documentation from $D=16$ to $D=32$ (1.62 MB compressed archive).


---

## [0.3.0] - 2026-10-18 — Phase 3: Dataset Pipeline & Graph Representation

### Added
- **Canonical CSR Graph Representation (`backend/cpu_engine/csr_graph.py`):**
  - Fully documented `CSRGraph` data class encapsulating `row_ptr` (`int32`), `col_idx` (`int32`), and `features` (`float32`).
  - Strict validation enforcing all 8 structural CSR invariants (start at 0, monotonic non-decreasing, end matches edge count, valid col index range, feature dimension match, finite values, and undirected symmetry).
  - Explicit handling for isolated nodes ($d_i = 0$), duplicate edge deduplication, self-loop pruning/retention, and directed/undirected edge semantics.
  - Serialization to `.npz` format pairing binary arrays with structured JSON manifest metadata (`save()` and `load()`).
  - Conversion methods between CSR and coordinate edge lists (`from_edge_list()`, `to_edge_list()`).
- **Deterministic Synthetic Graph Generator (`backend/cpu_engine/synthetic_generator.py`):**
  - High-performance vectorized Barabási–Albert generator producing power-law degree distributions ($\gamma \approx 2.5$) in $< 1$ second for 10,000 nodes.
  - Support for uniform random graphs (Erdős–Rényi style), star graphs (extreme degree skew stress-testing), and empty graphs.
  - Guaranteed finite `float32` node features and zero external network dependencies for CI/CD.
- **Temporal Transaction Graph Builder (`backend/cpu_engine/transaction_graph_builder.py`):**
  - Financial tabular-to-graph constructor enforcing strict temporal causality ($t_u \le t_v$).
  - Missing/null sentinel rejection (`NaN`, empty strings, `-1`) preventing artificial hub clique formation.
  - Quarantine of target label `isFraud` completely outside the graph structure into a separate array.
  - Offline fallback checking local raw directory without making unverified network requests.
- **Fixture Generation Utility (`backend/cpu_engine/generate_fixtures.py`):**
  - Generated `data/fixtures/toy_graph_5nodes.npz` (5 nodes, 4 edges, $D=4$) for instant edge-case testing.
  - Generated `data/fixtures/synthetic_tier_a_10k.npz` (10,000 nodes, 94,945 edges, $D=16$, 1.62 MB) for Tier A scalability benchmarks.
- **Comprehensive Test Suite (`backend/tests/test_csr_graph.py`):**
  - 20 unit, invariant, and edge-case tests verifying row pointers, out-of-bounds column indices, NaN/Inf feature rejection, empty neighborhoods, round-trip serialization, and temporal leakage prevention.
  - Test suite passes 22/22 tests in 2.03s.
- **Formal Graph Data Contract (`docs/GRAPH_DATA_CONTRACT.md`):** Authoritative contract specifying memory layouts, byte offsets, invariant definitions, and serialization schemas.
- **Updated Data Dictionary (`docs/DATA_DICTIONARY.md`):** Complete field-level mappings, types, nullability rules, and leakage prevention protocols.
- **Architectural Decisions (`docs/DECISIONS.md`):** Added `ADR-008` (32-bit Integer Indexing for CSR Arrays) and `ADR-009` (Temporal Causality and Sentinel Rejection).


---

## [0.2.0] - 2026-10-18 — Phase 2: Literature Review & Research Specification

### Added
- **Literature Review (`docs/LITERATURE_REVIEW.md`):** Comprehensive survey of verified publications:
  - Bell & Garland (SC '09) on CSR-Vector GPU parallelization.
  - NVIDIA CUDA Programming Guide & Best Practices Guide on memory coalescing and warp scheduling.
  - Wang et al. (PPoPP '16) on Gunrock fine-grained graph load balancing.
  - Huang et al. (SC '20, *GE-SpMM*) on CSR coalesced row caching for GNNs.
  - Hu et al. (SC '20, *FeatGraph*) on co-optimizing graph and feature dimension schedules.
  - Dou et al. (CIKM '20, *CARE-GNN*) on fraudster camouflage and degree normalization.
  - Wang et al. (ICDM '19, *SemiGNN*) on relational fraud detection at scale.
  - Weber et al. (KDD '19, *Elliptic*) on financial forensics benchmark metrics (PR-AUC and minority F1).
- **Evidence-Backed Research Gap Analysis (`docs/RESEARCH_GAP.md`):** Established four explicit gaps connecting academic literature to FGI engineering implementations.
- **Formal Research Specification (`docs/RESEARCH_SPECIFICATION.md`):** Defined $RQ_1$–$RQ_3$, formal testable hypotheses ($H_1$–$H_3$), independent/dependent variables, supporting/rejecting criteria, and separated engineering vs. experimental vs. modest research contributions.
- **Dataset Evaluation & Selection (`docs/DATASET_SELECTION.md`):** Evaluated IEEE-CIS, Elliptic, and synthetic fixtures; established leak-free temporal splitting rules and adopted the Dual-Dataset strategy.
- **Clarified Algorithm Specification (`docs/ALGORITHM_SPECIFICATION.md`):** Formulated exact mathematical definitions of CSR arrays, Mean/Sum/Max aggregators, zero-degree sentinel handling ($d_i = 0 \implies \vec{0}$), and numerical equivalence criteria ($\text{atol} \le 10^{-5}$).
- **Benchmark Experiment Protocol (`docs/EXPERIMENT_PROTOCOL.md`):** Established tri-state measurement standards (`MEASURED`, `SAMPLE`, `NOT_RUN`), hardware CUDA event timing isolation, warm-up policies, OOM protection, and machine-readable JSON schema.
- **Architectural Decisions (`docs/DECISIONS.md`):** Appended `ADR-005` (Dual-Dataset Strategy), `ADR-006` (Mathematical Aggregation Scope & Zero-Degree Sentinel), and `ADR-007` (Decoupled GPU Benchmarking).

---

## [0.1.0] - 2026-10-18 — Phase 1: Project Foundation & Dependency Setup

### Added
- **Repository Architecture:** Created modular project layout covering `backend/`, `frontend/`, `data/`, `experiments/`, and `docs/`.
- **Version Control & Hygiene:** Initialized Git repository on `main` branch with comprehensive `.gitignore` and `.env.example`.
- **Python Virtual Environment:** Established dedicated local `.venv` under Python 3.14.3.
- **Backend Dependencies:** Installed and verified FastAPI, Uvicorn, Pydantic, NumPy, SciPy, Pandas, Scikit-Learn, Pytest, and HTTPX.
- **Health & Telemetry API:** Implemented FastAPI application with `/` root and `/api/v1/health` endpoint reporting system status and NVIDIA GeForce MX450 GPU detection.
- **Backend Test Suite:** Added `backend/tests/test_health.py` and `pytest.ini` with automated verification (2/2 tests passed).
- **Frontend Scaffolding:** Configured React 18, TypeScript, and Vite with zero-radius Obsidian Cybernetics design tokens (`frontend/src/styles/design-tokens.css`). Verified clean production build (`npm run build`).
- **Stitch Design Alignment:** Inspected Stitch design export `stitch_fraudgraph_intelligence_platform` and recorded approved tokens and screen route mappings in `docs/DESIGN_SYSTEM.md`.
- **Documentation Suite:** Authored foundational technical documentation.

---

## [0.0.1] - 2026-10-18 — Phase 0: Workspace Audit & Feasibility

### Added
- Comprehensive system audit of Windows 11 host, 116.8 GB free disk on D:, Python 3.14, Node.js v24, Git, and NVIDIA GeForce MX450 GPU (Turing, Compute Capability 7.5).
- Identified local absence of `nvcc` and MSVC build tools; formulated two-tier Google Colab execution strategy for CUDA acceleration.
- Audited Stitch MCP connection and verified access to project `9197530824498398398`.
- Initialized `docs/ENVIRONMENT_SETUP.md` and `docs/PHASE_STATUS.md`.
