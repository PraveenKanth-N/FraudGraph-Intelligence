# Phase Status — FraudGraph Intelligence

**Authoritative Record of Implementation Progress and Phase Gates**
*Project ID:* FGI  
*Academic Course:* VIT Vellore — CSI4007 GPU Programming  
*Workspace Root:* `D:\FraudGraph Intelligence`  
*Current Local Time:* 2026-10-18T21:30:00+05:30  
*Responsible Agent:* Antigravity  

---

## 1. Phase Gate Master Table

| Phase | Description | Module IDs Covered | Gate Status | Completion Criteria / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | Workspace Audit & Feasibility | FGI-MOD-01 | **PASSED (Complete)** | Environment audited, D: verified (116.8 GB free), tools detected, Stitch connected, reference assets recorded, execution strategy formulated. |
| **Phase 1** | Project Foundation & Dependency Setup | FGI-MOD-01 | **PASSED (Complete)** | Git repository initialized, `.venv` created, baseline directory scaffolding established, complete documentation suite created, health endpoint verified via pytest (2/2 passing), frontend scaffold verified via TypeScript/Vite build (0 errors). |
| **Phase 2** | **Literature Review & Research Specification** | FGI-MOD-02 | **PASSED (Complete)** | Rigorous review of verified literature (Bell & Garland '09, NVIDIA CUDA Guides, Huang et al. '20, Hu et al. '20, Wang et al. '16, Dou et al. '20, Weber et al. '19, Wang et al. '19); evidence-backed gap analysis (`RESEARCH_GAP.md`); testable hypotheses & variable matrices (`RESEARCH_SPECIFICATION.md`); candidate dataset evaluation & Dual-Dataset strategy (`DATASET_SELECTION.md`); mathematical CSR & zero-degree operator specifications (`ALGORITHM_SPECIFICATION.md`); benchmark protocol & tri-state schema (`EXPERIMENT_PROTOCOL.md`); ADRs 005–007 recorded (`DECISIONS.md`). |
| **Phase 3** | **Dataset Pipeline & Graph Representation** | FGI-MOD-02, FGI-MOD-03 | **PASSED (Complete)** | Canonical CSR graph implementation (`csr_graph.py`) enforcing 8 invariants; deterministic synthetic power-law/star/uniform generator (`synthetic_generator.py`); temporal transaction graph builder (`transaction_graph_builder.py`) with zero leakage; 20 unit/invariant tests passing (22/22 total suite in 2.03s); `toy_graph_5nodes.npz` and `synthetic_tier_a_10k.npz` (10K nodes, 95K edges) generated; `GRAPH_DATA_CONTRACT.md` authored; ADRs 008-009 recorded. |
| **Phase 4** | **CPU Reference Implementation** | FGI-MOD-04 | **PASSED (Complete)** | Implemented Sum, Mean, Max aggregators in `aggregators.py`; dual reference and vectorized SpMM modes; explicit empty-neighborhood zero sentinel ($d_i = 0 \implies \vec{0}$); 22 tests in `test_aggregators.py` (total 44/44 backend tests passing in 2.36s); high-precision benchmark harness in `benchmark_cpu.py` with JSON output in `experiments/results/`; resolved Phase 3 fixture inconsistencies (toy isolated node 4, non-zero features); `CPU_REFERENCE_ENGINE.md` authored. |
| **Phase 5** | **Baseline CUDA Implementation** | FGI-MOD-05 | **PASSED (Complete)** | Baseline 1-thread-per-node CUDA C++ kernel in `fgi_baseline_kernels.cu`; standalone CLI driver in `main_baseline.cu`; Colab runner package in `colab_bundle_fgi_phase5.zip` (3.55 MB); executed on physical NVIDIA Tesla T4 GPU (40 SMs, 14.9 GB VRAM); correctness gate PASSED across Sum, Mean, Max (error = 0.0, tol <= 1e-5); synthetic Tier A (10K nodes, 95K edges, D=32) achieved 1.39B edges/sec throughput (kernel latency 0.068 ms); all 6 results ingested into `experiments/results/`; regression suite 55 passed; `CUDA_BASELINE.md` and `ADR-010` updated. |
| **Phase 6** | **CUDA Optimization Laboratory** | FGI-MOD-06 | **PASSED (Complete — Exp 6.1, 6.2, 6.3 Empirically Validated)** | Optimization plan in `PHASE6_OPTIMIZATION_PLAN.md`; Exp 6.1 (Warp Coalescing) achieved ~4.0x kernel speedup; Exp 6.2 (Pinned Streams) achieved 3.35x-3.44x end-to-end speedup vs Baseline (3.22x-3.27x vs 6.1); Exp 6.3 (Degree-Aware Hybrid Dispatch) validated on Tesla T4 (Driver 580.82.07, CUDA 13.0), eliminating SIMT degree divergence on heavy-tailed power-law graphs to achieve 107.7x kernel speedup over Baseline (3.38x over Coalesced/Streams) and 19.92x E2E speedup on MEAN (18.37x on MAX), with 100% numerical verification (atol <= 1e-5); 80 backend tests passing; `EXPERIMENT_6_1_WARP_COALESCED.md`, `EXPERIMENT_6_2_PINNED_MEMORY_STREAMS.md`, and `EXPERIMENT_6_3_DEGREE_AWARE_DISPATCH.md` authored. |
| **Phase 7** | Fraud Scoring & Evaluation | FGI-MOD-07 | **IN PROGRESS (Phases 7.1, 7.2, 7.3 Complete)** | Dataset audit completed (7.1); deterministic labeled synthetic fraud fixture (`synthetic_fraud_10k.npz`) generated with zero temporal leakage (7.2); feature enrichment pipeline (`feature_enrichment.py`) implemented supporting raw, mean, max, sum, and combined representations with causal and train-only context policies (7.3); 27 unit tests passing across Phase 7 (108 total suite). Downstream classifier baselines (7.4-7.5) pending. |
| **Phase 8** | Benchmark Automation & Experiment Management | FGI-MOD-08 | **PENDING** | Multi-run automated harness, graph-scaling experiments, latency/throughput speedup metrics, machine-readable JSON/CSV storage. |
| **Phase 9** | Stitch UI Design & Frontend Implementation | FGI-MOD-10 | **PENDING** | React + Vite + TypeScript application, Obsidian Cybernetics design system tokens, Cytoscape.js graph canvas, ECharts analytics. |
| **Phase 10**| Backend & Frontend Integration | FGI-MOD-09, FGI-MOD-10 | **PENDING** | FastAPI REST endpoints (`/api/v1`), interactive graph queries, live experiment trigger/ingestion, diagnostics console. |
| **Phase 11**| Quality Assurance & Reproducibility | FGI-MOD-11 | **PENDING** | Automated frontend tests, backend pytest suite, CUDA correctness test suite, accessibility and performance audit. |
| **Phase 12**| Research Report & Final Release | FGI-MOD-12 | **PENDING** | Comprehensive academic report draft, publication-quality plots, reproducibility bundle, viva voce defense slides. |

---

## 2. Phase 2 Deliverables Summary

### 2.1 Research Foundation & Literature Survey
- **`docs/LITERATURE_REVIEW.md`:** Authoritative survey of verified publications spanning GPU sparse linear algebra, GNN memory bottlenecks, SIMT thread divergence, and financial fraud detection. All citations independently verified via publisher DOI / official technical documentation. Unverified papers explicitly quarantined in a "Verification Needed" section.
- **`docs/RESEARCH_GAP.md`:** Evidence-backed analysis contrasting tabular trees (relational blindness), generic GNN frameworks (framework churn and uncoalesced memory reads), and vendor SpMM libraries (inflexible operators).
- **`docs/RESEARCH_SPECIFICATION.md`:** Final research questions ($RQ_1$–$RQ_3$), formal hypotheses ($H_1$–$H_3$) with independent/dependent variables and empirical rejection criteria. Modest, defensible contributions separated into engineering, experimental, and research components.

### 2.2 Dataset Strategy & Leakage Controls
- **`docs/DATASET_SELECTION.md`:** Rigorous comparative analysis of IEEE-CIS, Elliptic, and synthetic graph generators. Established the **Dual-Dataset Strategy**:
  1. *Controlled Synthetic Fixtures:* For deterministic, zero-network-dependency CI/CD tests and micro-architectural scaling experiments.
  2. *Empirical Financial Benchmark:* Managed subset of IEEE-CIS with strict chronological splitting and target leakage prevention.

### 2.3 Mathematical Algorithms & Experiment Protocols
- **`docs/ALGORITHM_SPECIFICATION.md`:** Clarified mathematical definitions of CSR arrays (`row_ptr`, `col_idx`, `features`), Mean/Sum/Max aggregators, and explicit zero-degree node handling ($d_i = 0 \implies \vec{0}$). Numerical tolerance established at $\text{atol} \le 10^{-5}$.
- **`docs/EXPERIMENT_PROTOCOL.md`:** Standardized benchmark methodology separating PCIe transfer from hardware CUDA event kernel timings. Defined tri-state measurement policy (`MEASURED`, `SAMPLE`, `NOT_RUN`) and machine-readable JSON schema.
- **`docs/DECISIONS.md`:** Added `ADR-005` (Dual-Dataset Strategy), `ADR-006` (Mathematical Aggregation Scope & Zero-Degree Sentinel), and `ADR-007` (Decoupled GPU Benchmarking).
- **`docs/CHANGELOG.md`:** Documented release `0.2.0` milestones.

---

## 3. Phase 2 Gate Acceptance
- **Gate Status:** **PASSED (Complete)**

---

## 4. Phase 3 Deliverables Summary

### 4.1 Graph Engine & Representation
- **`backend/cpu_engine/csr_graph.py`:** Canonical CSR graph class enforcing all 8 structural CSR invariants:
  1. `row_ptr[0] == 0`.
  2. Monotonic non-decreasing sequence ($row\_ptr[i] \le row\_ptr[i+1]$).
  3. Last entry equals edge count ($row\_ptr[N] == M$).
  4. Column index validity ($0 \le col\_idx[j] < N$).
  5. Contiguous array alignment and 32-bit integer typing (`np.int32`).
  6. Feature matrix shape ($N \times D$) and single-precision typing (`np.float32`).
  7. Finite numerical guarantee (zero NaNs or Infs).
  8. Bidirectional edge symmetry for undirected graphs.
  Includes `.save()` / `.load()` with JSON manifest, isolated node support, duplicate edge deduplication, and edge list conversions.

### 4.2 Deterministic Synthetic Generation
- **`backend/cpu_engine/synthetic_generator.py`:** High-speed vectorized Barabási–Albert preferential attachment generator producing scale-free power-law graphs ($\gamma \approx 2.5$) in under 1 second for 10,000 nodes.
- Also supports uniform random, star (hub stress-testing), and empty graph generation, all guaranteed finite and reproducible.

### 4.3 Temporal Pipeline & Leakage Controls
- **`backend/cpu_engine/transaction_graph_builder.py`:** Tabular transaction graph builder with:
  - Causal forward-in-time edge direction ($t_u \le t_v$).
  - Sentinel rejection (`NaN`, empty strings, `-1`) preventing artificial dense hub cliques.
  - Complete quarantine of target label `isFraud` outside node features.
  - Clean offline fallback when raw external files are absent.

### 4.4 Fixtures & Test Verification
- **Generated Fixtures:**
  - `data/fixtures/toy_graph_5nodes.npz` (5 nodes, 8 directed CSR entries representing 4 undirected pairs, isolated node 4, $D=4$).
  - `data/fixtures/synthetic_tier_a_10k.npz` (10,000 nodes, 94,945 directed edges, $D=32$, 1.62 MB).
- **Automated Tests (`backend/tests/test_csr_graph.py`):** 20 comprehensive unit tests covering all 8 invariants, edge cases, round-trips, and leakage rules. All 22/22 tests across the backend suite pass in 2.03s.
- **Frontend Build:** Clean compilation with 0 TypeScript/Vite errors in 1.85s.

### 4.5 Specifications & Architectural Records
- **`docs/GRAPH_DATA_CONTRACT.md`:** Authoritative specification for memory layouts, types, schemas, and invariants.
- **`docs/DATA_DICTIONARY.md`:** Updated with CSR representation details and tabular-to-graph mapping fields.
- **`docs/DECISIONS.md`:** Appended `ADR-008` (32-bit Integer Indexing for CSR) and `ADR-009` (Temporal Causality and Sentinel Rejection).
- **`docs/CHANGELOG.md`:** Added release `0.3.0` milestones.

---

## 5. Phase 3 Gate Acceptance
- **Gate Status:** **PASSED (Complete)**

---

## 6. Phase 4 Deliverables Summary

### 6.1 Aggregation Engine (`backend/cpu_engine/aggregators.py`)
- Implemented `aggregate_neighbors`, `aggregate_sum_reference`, `aggregate_mean_reference`, `aggregate_max_reference`, and `aggregate_vectorized`.
- Mathematical compliance:
  - Sum: $Y_{i, :} = \sum_{j \in \mathcal{N}(i)} X_{j, :}$ ($0$ if empty).
  - Mean: $Y_{i, :} = \frac{1}{d_i} \sum_{j \in \mathcal{N}(i)} X_{j, :}$ ($0$ if empty).
  - Max: $Y_{i, :} = \max_{j \in \mathcal{N}(i)} X_{j, :}$ with strict zero-sentinel policy for empty neighborhoods ($\vec{0} \in \mathbb{R}^D$).
- Dual execution architectures:
  - Node-iterative reference loop for direct mathematical clarity.
  - Vectorized SciPy CSR SpMM matrix multiplication for high-performance CPU baseline.
- Full type safety and defensive input validation (`TypeError`, `ValueError`).

### 6.2 Fixture Resolution & Accuracy
- Corrected `toy_graph_5nodes.npz`:
  - Verified that Node 4 is the isolated node ($d_4 = 0$), while Node 0 has degree 2 (connected to 1 and 2).
  - Clarified edge count: 8 directed CSR adjacency entries representing 4 undirected edge pairs.
  - Populated distinct, non-zero float32 features (positive, negative, and zero values) allowing hand-calculated verification.
- Corrected documentation for `synthetic_tier_a_10k.npz` to reflect exact feature dimension $D = 32$ (1.62 MB compressed archive).

### 6.3 Test Suite Verification (`backend/tests/test_aggregators.py`)
- 22 comprehensive tests covering all 18 specification requirements (hand calculations, empty neighborhoods, isolated nodes, single nodes, self-loops, directed asymmetry, undirected symmetry, hub nodes, multiple dimensions $D \in \{1, 4, 16, 32, 128\}$, negative features, determinism, immutability, defensive input validation, non-finite input rejection, and cross-verification).
- Total backend test suite: **44/44 tests passed in 2.36s**.

### 6.4 CPU Benchmark Harness (`backend/cpu_engine/benchmark_cpu.py`)
- High-precision monotonic timer (`time.perf_counter_ns`).
- Empirical observations on Intel Core i5-1135G7 (8 cores):
  - `toy_graph_5nodes.npz`: Mean latency 0.033 ms (245,927 edges/s).
  - `synthetic_tier_a_10k.npz` (10K nodes, 94.9K edges, $D=32$):
    - Reference Mean: 133.177 ms (712,923 edges/s).
    - Vectorized Mean: 6.325 ms (15,011,945 edges/s).
    - Reference Sum: 166.877 ms (568,953 edges/s).
    - Vectorized Sum: 3.624 ms (26,196,494 edges/s).
    - Reference Max: 167.173 ms (567,945 edges/s).
- Benchmark results stored in `experiments/results/` marked with state `MEASURED`.

### 6.5 Documentation Deliverables
- **`docs/CPU_REFERENCE_ENGINE.md`:** Authoritative specification of formulas, architectures, numerical tolerance ($\text{atol} \le 10^{-5}$), and benchmark data.
- **`docs/ALGORITHM_SPECIFICATION.md`:** Updated with reference to `aggregators.py`.
- **`docs/EXPERIMENT_PROTOCOL.md`:** Updated with CPU benchmark harness usage and baseline metrics.
- **`docs/TEST_PLAN.md`:** Updated test pyramid and 44-test coverage details.
- **`docs/CHANGELOG.md`:** Logged release `0.4.0` milestones.

---

## 7. Phase 4 Gate Acceptance
- **Gate Status:** **PASSED (Complete)**

---

## 8. Phase 5 Deliverables Summary

### 8.1 Baseline CUDA C++ Implementation (`backend/gpu_engine/cuda/`)
- Implemented `aggregate_baseline_kernel` in [`backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_baseline_kernels.cu):
  - 1 thread mapped per output node ($i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$), block size 256.
  - Supports Sum, Mean, and Max aggregation operators.
  - Strict zero-sentinel handling: explicitly returns $\vec{0}$ for empty neighborhoods ($d_i = 0$).
  - Negative-only neighborhood handling for Max operator initialized with first neighbor's feature.
- Implemented error handling macros (`CUDA_CHECK`, `CUDA_CHECK_KERNEL`) in [`fgi_cuda_common.cuh`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/include/fgi_cuda_common.cuh).
- Implemented standalone C++ CLI driver in [`main_baseline.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_baseline.cu):
  - 64-bit safe allocation sizes on host.
  - Hardware CUDA event timing isolating $t_{\text{H2D}}$, $t_{\text{kernel}}$ (20 repetitions after 5 warmups), $t_{\text{D2H}}$, and $t_{\text{e2e}}$.
  - Multi-architecture Makefile targeting SM 7.0, 7.5, and 8.0.

### 8.2 Google Colab Cloud Workflow (`backend/gpu_engine/colab_runner/`)
- **Automated Packaging (`package_colab_bundle.py`):** Unpacks `toy_graph_5nodes` and `synthetic_tier_a_10k` into raw binary arrays with precomputed Phase 4 CPU reference outputs, packaging everything into `experiments/colab_bundle_fgi_phase5.zip` (3.55 MB).
- **Automated Colab Runner (`run_colab_benchmarks.py`):** Compiles CUDA binary with `nvcc`, executes two-stage correctness gate against CPU references ($\text{atol} \le 10^{-5}$), executes multi-repetition benchmark matrix, and archives results.
- **Turnkey Notebook (`fgi_colab_workflow.ipynb`):** Complete Colab notebook for one-click compilation, validation, and execution.
- **Results Ingestion Utility (`ingest_colab_results.py`):** Validates downloaded results against FGI schema, confirms `MEASURED` state, checks tolerance, and archives to `experiments/results/`.
- **Workflow Guide (`backend/gpu_engine/colab_runner/README.md`):** Complete execution and troubleshooting documentation.

### 8.3 Python Host Adapter & Test Verification
- Implemented [`backend/gpu_engine/cuda_runner.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda_runner.py) with graceful missing-binary error reporting pointing to Colab workflow.
- Implemented 12 comprehensive tests in [`backend/tests/test_cuda_baseline.py`](file:///D:/FraudGraph%20Intelligence/backend/tests/test_cuda_baseline.py). Full suite: **55 passed, 1 skipped (live GPU test appropriately skipped without local nvcc) in 3.11s**.

### 8.4 Architecture & Specification Documentation
- **`docs/CUDA_BASELINE.md`:** Authoritative specification of baseline kernel mapping, memory safety, Colab workflow, and baseline performance bottlenecks (uncoalesced memory reads, warp divergence).
- **`docs/DECISIONS.md`:** Recorded `ADR-010` (One-Thread-per-Node Baseline CUDA Kernel Mapping and Colab Validation Protocol).
- **`docs/KNOWN_LIMITATIONS.md`:** Added Section 4 detailing baseline architectural bottlenecks to be solved in Phase 6.
- **`docs/EXPERIMENT_PROTOCOL.md`:** Added Section 7 on decoupled execution metrics and honest speedup accounting.
- **`docs/TEST_PLAN.md`:** Updated test matrix with Phase 5 coverage.
- **`docs/CHANGELOG.md`:** Documented release `0.5.0`.

### 8.5 Empirical Tesla T4 GPU Benchmark Results & Ingestion
- **Hardware Telemetry Verified:** NVIDIA Tesla T4 (Compute Capability 7.5, 14,912.7 MB VRAM, 40 SMs, 320.0 GB/s peak bandwidth).
- **CUDA Compiler:** CUDA compilation tools release 13.0, V13.0.88 (flags: `-O3 -std=c++17`).
- **Stage 1 Correctness Gate (`toy_graph_5nodes`, $N=5, M=8, D=4$):**
  - Sum: Max absolute error = $0.0$, status `PASSED` ([`val_toy_sum.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/val_toy_sum.json)).
  - Mean: Max absolute error = $0.0$, status `PASSED` ([`val_toy_mean.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/val_toy_mean.json)).
  - Max: Max absolute error = $0.0$, status `PASSED` ([`val_toy_max.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/val_toy_max.json)).
- **Stage 2 Scale Benchmarks (`synthetic_tier_a_10k`, $N=10,000, M=94,945, D=32$):**
  - Sum: $t_{\text{kernel}} = 0.0681 \pm 0.0007\text{ ms}$, $1.394\text{ B edges/s}$, $203.44\text{ GB/s}$ effective BW, error = $0.0$ ([`exp_gpu_baseline_tier_a_sum_20261018_173315.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/exp_gpu_baseline_tier_a_sum_20261018_173315.json)).
  - Mean: $t_{\text{kernel}} = 0.0682 \pm 0.0007\text{ ms}$, $1.393\text{ B edges/s}$, $203.24\text{ GB/s}$ effective BW, error = $0.0$ ([`exp_gpu_baseline_tier_a_mean_20261018_173315.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/exp_gpu_baseline_tier_a_mean_20261018_173315.json)).
  - Max: $t_{\text{kernel}} = 0.0673 \pm 0.0022\text{ ms}$, $1.411\text{ B edges/s}$, $205.84\text{ GB/s}$ effective BW, error = $0.0$ ([`exp_gpu_baseline_tier_a_max_20261018_173315.json`](file:///D:/FraudGraph%20Intelligence/experiments/results/exp_gpu_baseline_tier_a_max_20261018_173315.json)).
- **Speedup vs Phase 4 CPU Reference (Intel i5-1135G7):**
  - Pure Kernel Speedup: $2,450.8\times$ (Sum), $1,953.9\times$ (Mean), $2,484.0\times$ (Max).
  - Pure Kernel Speedup vs CPU Vectorized SpMM: $53.2\times$ (Sum), $92.8\times$ (Mean).
  - End-to-End Speedup (including PCIe): $179.2\times$ (Sum), $139.4\times$ (Mean), $166.3\times$ (Max) vs CPU Reference; $3.89\times$ (Sum), $6.62\times$ (Mean) vs CPU SpMM.
- **Ingestion & Archive:** All 6 result JSON files validated with strict schema conformity and archived into `experiments/results/`.

---

## 9. Phase 5 Gate Acceptance
- **Gate Status:** **PASSED (Complete)**
- **Verification Evidence:**
  1. Baseline CUDA C++ kernel and host CLI driver compiled without warnings using CUDA compilation tools release 13.0, V13.0.88.
  2. Numerical correctness validated on both toy and 10K node graphs with max absolute error equal to $0.0$, strictly satisfying the $\text{atol} \le 10^{-5}$ tolerance standard.
  3. Decoupled PCIe transfer and pure kernel event timing verified on physical Tesla T4 hardware.
  4. Ingested result files conform strictly to the `MEASURED` state schema in `docs/EXPERIMENT_PROTOCOL.md`.
  5. Full backend regression test suite passes 55/55 tests in 2.96s (1 live GPU test appropriately skipped on local host).
  6. Frontend builds cleanly with 0 TypeScript/Vite errors in 1.70s.
- **Next Phase:** Phase 6 (CUDA Optimization Laboratory) unlocked.

---

## 10. Phase 6 Deliverables & Experiment 6.1 Status

### 10.1 Optimization Plan
- **`docs/PHASE6_OPTIMIZATION_PLAN.md`:** Comprehensive planning specification investigating 7 candidate optimization areas (Memory Coalescing, Warp-per-Node Mapping, Warp Shuffle Reductions, Degree-Aware Bucketing, Shared-Memory Hub Caching, Launch Geometry Sweep, and Pinned DMA Streams).
- Reaffirms research integrity rules: all candidates are hypotheses until measured on GPU hardware; baseline kernel remains completely untouched; decoupled timing enforced.

### 10.2 Experiment 6.1 Implementation (Warp-Coalesced Aggregator Kernel)
- **Separate Kernel Implementation:** Authored in [`backend/gpu_engine/cuda/src/fgi_coalesced_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_coalesced_kernels.cu) and [`include/fgi_coalesced_kernels.cuh`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/include/fgi_coalesced_kernels.cuh).
  - Assigns 1 warp (32 threads) per output node.
  - For $D=32$, lane $t$ processes column $t$, achieving 100% coalesced 128-byte global memory reads.
  - Column-strided traversal (`for (int k = lane_id; k < feature_dim; k += 32)`) guarantees correct execution for arbitrary feature dimensions $D$.
  - Preserves exact zero-sentinels for empty neighborhoods and first-neighbor initialization for Max aggregation.
  - Eliminates intra-warp degree divergence during edge iteration while documenting that inter-warp divergence across disparate nodes remains present.
- **Standalone CLI Driver:** [`backend/gpu_engine/cuda/src/main_coalesced.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_coalesced.cu) supporting `--op`, `--warmup`, `--repeat`, `--block-size`, `--validate`, and JSON export conforming to `docs/EXPERIMENT_PROTOCOL.md`.
- **Build Integration:** Updated [`backend/gpu_engine/cuda/Makefile`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/Makefile) to build both `fgi_baseline_runner` and `fgi_coalesced_runner` concurrently. Resolved CUDA 13.0 (V13.0.88) compatibility on Colab Tesla T4 by targeting `-gencode arch=compute_75,code=sm_75` and removing unsupported `compute_70` architecture.
- **Colab Workflow Integration:**
  - Updated [`backend/gpu_engine/colab_runner/run_colab_benchmarks.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/run_colab_benchmarks.py) to compile both targets, verify correctness gates for both, and run comparative benchmarks on `synthetic_tier_a_10k` ($N=10,000, M=94,945, D=32$).
  - Updated [`backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb) for turnkey execution on Colab Tesla T4.
  - Packaged self-contained bundle [`experiments/colab_bundle_fgi_phase6.zip`](file:///D:/FraudGraph%20Intelligence/experiments/colab_bundle_fgi_phase6.zip) (3.56 MB).
- **Automated Verification:** Added 9 dedicated tests in [`backend/tests/test_cuda_coalesced.py`](file:///D:/FraudGraph%20Intelligence/backend/tests/test_cuda_coalesced.py) including canonical Tier A fixture checks ($N=10,000, M=94,945, D=32$), bandwidth metric math, string formatting, and raw ingested result assertions. Full backend test suite now passes **64/64 tests (1 live GPU test skipped)**.
- **Specification Report:** Authored [`docs/EXPERIMENT_6_1_WARP_COALESCED.md`](file:///D:/FraudGraph%20Intelligence/docs/EXPERIMENT_6_1_WARP_COALESCED.md).

### 10.3 Experiment 6.1 GPU Revalidation & Performance Audit
- **Revalidation Date:** October 19, 2026.
- **Target Hardware & Environment:** NVIDIA Tesla T4 (`sm_75`, 40 SMs, 14.9 GB VRAM, Driver 580.82.07, CUDA 13.0, V13.0.88).
- **Numerical Correctness:** PASSED on `toy_graph_5nodes` and `synthetic_tier_a_10k` ($N=10,000, M=94,945, D=32$) with reported error $0.00 \le 1.0 \times 10^{-5}$ across Mean, Sum, and Max for both Baseline and Coalesced engines.
- **Pure Kernel Acceleration (Revalidation Run `20261019_175441`):**
  - **Mean:** $0.06892\text{ ms} \rightarrow 0.01853\text{ ms}$ (**$3.72\times$ speedup**, $73.1\%$ latency reduction)
  - **Sum:** $0.06860\text{ ms} \rightarrow 0.01684\text{ ms}$ (**$4.07\times$ speedup**, $75.4\%$ latency reduction)
  - **Max:** $0.06726\text{ ms} \rightarrow 0.01670\text{ ms}$ (**$4.03\times$ speedup**, $75.2\%$ latency reduction)
- **End-to-End Latency, Transfer Overhead & Amdahl's Law:**
  - **Mean:** $0.96623\text{ ms} \rightarrow 0.97857\text{ ms}$ (**$0.99\times$ speedup** / $0.987\times$). Mean kernel latency improved significantly ($3.72\times$), but un-pinned synchronous PCIe transfers ($0.8973\text{ ms}$ in baseline vs $0.9600\text{ ms}$ in coalesced) constituted $\sim 93\% - 98\%$ of total wall-clock time. A $+0.0627\text{ ms}$ transfer jitter on the shared Colab PCIe bus slightly exceeded the $0.0504\text{ ms}$ kernel reduction.
  - **Sum:** $0.95632\text{ ms} \rightarrow 0.87603\text{ ms}$ (**$1.09\times$ speedup**). Steady PCIe transfers permitted kernel savings to translate into end-to-end acceleration.
  - **Max:** $0.94608\text{ ms} \rightarrow 0.88451\text{ ms}$ (**$1.07\times$ speedup**).
  - **Scientific Integrity:** We explicitly **do not claim that all operations improved end to end**. Without asynchronous PCIe streaming, whole-pipeline speedups remain capped by Amdahl's Law, experimentally motivating Candidate 7 (Pinned DMA Streams) for Phase 6.
- **Bandwidth Audit & Analytical Estimation Disclosures:**
  - Both compulsory and algorithmic bandwidths are **analytical model calculations** based on assumed data volumes divided by kernel execution time; they are **not** sampled hardware performance counters from Nsight Compute or DRAM bus probes, and they do not claim or establish physical memory-bus saturation.
  - **Estimated Compulsory DRAM Bandwidth ($2.98\text{ MB}$ unique footprint):** Baseline $43.24 - 44.30\text{ GB/s}$ ($13.5\% - 13.8\%$ of peak) $\rightarrow$ Coalesced $160.77 - 178.37\text{ GB/s}$ ($50.2\% - 55.7\%$ of Tesla T4 $320\text{ GB/s}$ peak), demonstrating a $\sim 4.0\times$ improvement in memory request efficiency.
  - **Estimated Algorithmic Memory Work Rate ($13.85\text{ MB}$ un-cached footprint):** Baseline $201.00 - 205.94\text{ GB/s}$ $\rightarrow$ Coalesced $747.41 - 829.23\text{ GB/s}$. This work rate exceeds peak DRAM bandwidth because the $1.28\text{ MB}$ feature table fits inside the Tesla T4's $4.0\text{ MB}$ unified L2 cache, enabling multi-TB/s on-chip cache reuse.
- **Software Fixes Verified:** `#include <cstdint>` and `std::uint64_t` typing added to resolve GCC/CUDA 13.0 compilation errors; standard `format_with_commas` helper verified; separate estimated bandwidth keys verified in JSON schema.
- **Scope Boundary:** Feature aggregation speedup evaluated exclusively; no claims made on fraud classification accuracy (Phase 7).
- **Telemetry Archive:** All 12 raw revalidation JSON files ingested into `experiments/results/` alongside preserved Run 1 records; archives preserved at `experiments/fgi_phase6_gpu_results.zip`, `experiments/fgi_phase6_revalidation_gpu_results_20261019_175441.zip`, and `experiments/fgi_phase6_gpu_results_20261019_160820.zip`.

### 10.4 Phase 6.1 Gate Status
- **Phase 6.1 Status:** **`PASSED (Experiment 6.1 Complete & Revalidated)`**

---

### 10.5 Experiment 6.2 Implementation & GPU Verification (Pinned Host Memory & Streams)
- **Objective:** Eliminate the PCIe data transfer bottleneck documented in Phase 6.1 ($\sim 93\% - 98\%$ of total wall-clock runtime) by replacing pageable memory with page-locked (pinned) host memory (`cudaMallocHost` / `cudaFreeHost`) and executing asynchronous transfers (`cudaMemcpyAsync`) within a dedicated non-default stream (`cudaStreamNonBlocking`).
- **Separate Implementation:** Authored in [`backend/gpu_engine/cuda/src/main_streams.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_streams.cu). Reuses verified `fgi::launch_coalesced_aggregation` without modifying baseline or Phase 6.1 CUDA sources.
- **Build & Adapter Integration:** Updated [`backend/gpu_engine/cuda/Makefile`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/Makefile) with target `fgi_streams_runner`; updated [`backend/gpu_engine/cuda_runner.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda_runner.py) to support `kernel="streams"`; updated Colab workflow script and notebook to execute 3-way benchmarking (Baseline vs Coalesced vs Streams).
- **Physical GPU Benchmark Results on NVIDIA Tesla T4 (Driver 580.82.07, CUDA 13.0, V13.0.88):**
  - **Date:** October 24, 2026 (Run `20261024_114748`).
  - **Numerical Correctness:** PASSED on all 9 toy-graph checks and all 9 Tier A benchmarks ($N=10,000, M=94,945, D=32$) with max absolute error equal to $0.00$ ($\text{atol} \le 1.0 \times 10^{-5}$) and zero NaNs/Infs.
  - **PCIe Transfer Latency Reduction (Pinned Memory DMA):**
    - **H2D Transfer Time:** Dropped from $0.538 - 0.551\text{ ms}$ (Baseline) to **$0.156 - 0.160\text{ ms}$** (**$3.41\times - 3.53\times$ speedup**).
    - **D2H Transfer Time:** Dropped from $0.317 - 0.350\text{ ms}$ (Baseline) to **$0.105\text{ ms}$** (**$3.02\times - 3.34\times$ speedup**).
    - **Measured PCIe Bandwidth:** Reached **$10.66 - 10.87\text{ GB/s}$ (H2D)** and **$12.11 - 12.20\text{ GB/s}$ (D2H)** effective payload transfer throughput, compared to $2.95 - 3.16\text{ GB/s}$ and $3.65 - 4.08\text{ GB/s}$ under pageable memory.
  - **Pure Kernel Invariance:**
    - Pure kernel latency remained mathematically invariant between Coalesced 6.1 and Streams 6.2 ($0.016731\text{ ms}$ vs $0.016869\text{ ms}$ for Sum, delta $0.82\%$; $0.017486\text{ ms}$ vs $0.017590\text{ ms}$ for Mean, delta $0.59\%$; $0.016512\text{ ms}$ vs $0.016558\text{ ms}$ for Max, delta $0.28\%$; maximum delta $\le 0.82\%$). Pure kernel speedup vs baseline remained **$3.88\times - 4.06\times$**.
  - **End-to-End Latency & Amdahl's Law Resolution:**
    - **Sum:** $0.9362\text{ ms} \rightarrow 0.2782\text{ ms}$ (**$3.37\times$ speedup vs Baseline**, **$3.26\times$ speedup vs Coalesced 6.1**).
    - **Mean:** $0.9418\text{ ms} \rightarrow 0.2809\text{ ms}$ (**$3.35\times$ speedup vs Baseline**, **$3.22\times$ speedup vs Coalesced 6.1**).
    - **Max:** $0.9671\text{ ms} \rightarrow 0.2809\text{ ms}$ (**$3.44\times$ speedup vs Baseline**, **$3.27\times$ speedup vs Coalesced 6.1**).
    - Directly confirms the Amdahl's Law hypothesis: compressing data transfer time by $3.4\times$ allows the $4.0\times$ kernel acceleration to translate into a **$3.35\times - 3.44\times$ whole-pipeline wall-clock speedup**.
- **Automated Verification:** Added 8 dedicated unit and schema tests in [`backend/tests/test_cuda_streams.py`](file:///D:/FraudGraph%20Intelligence/backend/tests/test_cuda_streams.py); full backend test suite passes **72/72 tests (1 live GPU test skipped)**; frontend typecheck and build pass cleanly with 0 errors.
- **Specification Report:** Authored [`docs/EXPERIMENT_6_2_PINNED_MEMORY_STREAMS.md`](file:///D:/FraudGraph%20Intelligence/docs/EXPERIMENT_6_2_PINNED_MEMORY_STREAMS.md).
- **Telemetry Archive:** All 18 raw JSON files ingested into `experiments/results/`; archives preserved at `experiments/fgi_phase6_gpu_results.zip` and timestamped backup `experiments/fgi_phase6_streams_gpu_results_20261024_114748.zip`.

### 10.6 Phase 6.2 Gate Status
- **Phase 6.2 Status:** **`PASSED (Experiment 6.2 Pinned Streams Complete & Revalidated)`**

---

### 10.7 Experiment 6.3 Implementation & GPU Verification (Degree-Aware Bucketing & Hybrid CUDA Dispatch)
- **Objective:** Evaluate Candidate 4 (Degree-Aware Bucketing) to test the research hypothesis that partitioning nodes by degree regime into dedicated execution kernels resolves inter-warp degree divergence caused by power-law hub nodes on heavy-tailed graph topologies.
- **Separate Implementation:** Authored in [`backend/gpu_engine/cuda/src/fgi_degree_aware_kernels.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/fgi_degree_aware_kernels.cu), [`include/fgi_degree_aware_kernels.cuh`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/include/fgi_degree_aware_kernels.cuh), and standalone driver [`backend/gpu_engine/cuda/src/main_degree_aware.cu`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/src/main_degree_aware.cu). All Phase 5, Phase 6.1, and Phase 6.2 source files remain **100% untouched**.
- **Build & Adapter Integration:** Updated [`backend/gpu_engine/cuda/Makefile`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda/Makefile) with target `fgi_degree_aware_runner`; updated [`backend/gpu_engine/cuda_runner.py`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/cuda_runner.py) to support `kernel="degree_aware"`; packaged turnkey 4-engine Colab runner bundle in [`experiments/colab_bundle_fgi_phase6.zip`](file:///D:/FraudGraph%20Intelligence/experiments/colab_bundle_fgi_phase6.zip) with normalized timestamps `(2026, 1, 1, 0, 0, 0)` preventing Makefile clock-skew warnings.
- **Physical GPU Benchmark Telemetry (NVIDIA Tesla T4, Driver 580.82.07, CUDA 13.0, Cohort `20261024_142037`):**
  - **Date & Hardware:** October 24, 2026 on Tesla T4 (`sm_75`, Compute Capability 7.5, 40 SMs, 15,360 MiB VRAM).
  - **Numerical Verification:** PASSED across all 12 toy checks and all 24 benchmark configurations on Tier A and Skewed graphs ($N=10,000$, $D=32$) with status `PASSED` and $\text{max\_abs\_error} \le 1.0 \times 10^{-5}$ ($0.00\text{e}+00$ on all cases except Skewed Mean at $1.19 \times 10^{-7}$).
  - **Demonstrated Accelerations on Heavy-Tailed Skewed Graph (`synthetic_skewed_10k`, max degree 6,340, 56 hubs):**
    - **`MEAN` Operator:**
      - Pure kernel latency dropped from $0.2202\text{ ms}$ (Streams) to **$0.0649\text{ ms}$** (**$3.39\times$ kernel speedup vs Streams**, **$107.71\times$ vs Baseline**).
      - End-to-end wall-clock latency (inclusive of $0.0621\text{ ms}$ host bucketing) dropped from $0.4834\text{ ms}$ (Streams) to **$0.3964\text{ ms}$** (**$1.22\times$ E2E speedup vs Streams**, **$19.92\times$ vs Baseline**).
      - Numerical error: $1.19 \times 10^{-7} \le 1.0 \times 10^{-5}$.
    - **`MAX` Operator:**
      - Pure kernel latency dropped from $0.2344\text{ ms}$ (Streams) to **$0.1106\text{ ms}$** (**$2.12\times$ kernel speedup vs Streams**, **$65.21\times$ vs Baseline**).
      - End-to-end wall-clock latency (inclusive of $0.0623\text{ ms}$ host bucketing) dropped from $0.5054\text{ ms}$ (Streams) to **$0.4432\text{ ms}$** (**$1.14\times$ E2E speedup vs Streams**, **$18.37\times$ vs Baseline**).
      - Numerical error: $0.00\text{e}+00$.
  - **Empirical Regressions & Negative Findings (Explicit Scientific Disclosure):**
    - **Regression 1: `SUM` on Skewed Graph:**
      - Pure kernel latency regressed from $0.2117\text{ ms}$ (Streams) to **$0.4641\text{ ms}$** (**$0.46\times$ speedup** / **$2.19\times$ slowdown**).
      - End-to-end latency regressed from $0.4975\text{ ms}$ (Streams) to **$0.8025\text{ ms}$** (**$0.62\times$ speedup** / **$1.61\times$ slowdown**).
      - *Observed Findings & Investigated Architectural Explanation:* In initial Run `141134`, 8-warp cooperative reduction reordered additions across the 6,340-edge super-hub, accumulating a non-associative float32 rounding difference of $3.97 \times 10^{-4}$ (relative error $4.56 \times 10^{-6} \le 1\text{ ULP}$ per addition) that was recorded as exceeding project tolerance ($\text{atol} \le 1.0 \times 10^{-5}$). To ensure exact $0.00\text{e}+00$ reference parity without relaxing tolerances, Warp 0 was assigned serial accumulation. Investigation of the subsequent $0.464\text{ ms}$ latency indicates that executing serial accumulation on 1 warp within a 256-thread block left 7 out of 8 warps inactive, reducing SM occupancy and latency hiding on the 56 hub blocks while serializing behind the standard-degree kernel on the same stream.
    - **Regression 2: All Tier A End-to-End Cases (Uniform Degree Distribution):**
      - `MEAN` E2E: $0.3433\text{ ms}$ vs $0.2810\text{ ms}$ Streams (**$0.82\times$ speedup** / **$22.2\%$ slowdown**).
      - `SUM` E2E: $0.3961\text{ ms}$ vs $0.2827\text{ ms}$ Streams (**$0.71\times$ speedup** / **$40.1\%$ slowdown**).
      - `MAX` E2E: $0.3832\text{ ms}$ vs $0.2785\text{ ms}$ Streams (**$0.73\times$ speedup** / **$37.4\%$ slowdown**).
      - *Root Cause Analysis:* On Tier A (where $94.95\%$ of nodes have degree 10 and 0 hub nodes exist), there are zero long-running warps to optimize. Host degree scanning and bucketing partition overhead ($t_{\text{bucketing}} = 0.053 - 0.109\text{ ms}$) and kernel launch indirection represent strictly non-amortizable overhead, demonstrating that degree-aware dispatch is an adaptive, topology-dependent strategy that should only be triggered when graph degree variance exceeds threshold.
- **Architectural Retention Decision:** Experiment 6.3 is retained in the FGI engine portfolio as a **specialized adaptive engine for skewed, heavy-tailed financial graphs** for `MEAN` and `MAX`, while Experiment 6.2 (Pinned Memory + Streams) remains the primary default canonical engine for quasi-uniform transaction graphs and `SUM`.
- **Telemetry Archive:** All 24 canonical benchmark records from cohort `20261024_142037` and 3 toy validation records ingested into `experiments/results/`; archives preserved at `experiments/fgi_phase6_gpu_results.zip` and `experiments/fgi_phase6_degree_aware_gpu_results_20261024_142037.zip`; historical aborted run `141134` safely quarantined in `experiments/results/raw/20261024_141134/` and `experiments/fgi_phase6_gpu_results_20261024_141134_aborted.zip`.

---

### 10.8 Phase 6 Status & Next Steps
- **Phase 6 Status:** **`PASSED (Experiments 6.1, 6.2, and 6.3 Complete & Empirically Validated)`**
- **Next Steps:** Phase 6 CUDA Optimization Laboratory is successfully closed. Transition to Phase 7 (Fraud Scoring & Classifier Pipeline) to evaluate graph-aggregated feature embeddings in fraud classification models.

---

## 11. Phase 7: Fraud Scoring & Evaluation Laboratory

### 11.1 Phase 7.1 Dataset and Label Audit (Read-Only)
- **Objective:** Establish the empirical ground truth regarding available datasets, targets, feature dimensions, and class balance prior to implementing classifier pipelines.
- **Audit Findings:**
  1. *Existing Fixtures (`data/fixtures/`):* `toy_graph_5nodes.npz`, `synthetic_tier_a_10k.npz`, and `synthetic_skewed_10k.npz` are unlabeled structural fixtures ($y$ is empty/unspecified). They exist strictly for numerical verification and CUDA performance benchmarking.
  2. *Raw Dataset Directory (`data/raw/`):* Currently unpopulated. Real external benchmarks (such as IEEE-CIS Fraud Detection) require large multi-gigabyte downloads and offline preprocessing.
  3. *Leakage Constraints:* Identified strict requirements that target labels must remain quarantined outside the node feature matrix $X \in \mathbb{R}^{N \times D}$, and that graph construction must strictly obey temporal ordering ($t_j \le t_i$) to prevent future-to-past information leakage during neighbor aggregation.
  4. *Downstream Interface Contract:* Confirmed that CPU reference aggregators (`sum`, `mean`, `max` from `backend/cpu_engine/aggregators.py`) and GPU runners produce output matrices of shape $(N, D)$, which concatenate naturally with raw node features $(N, D)$ into enriched representations $(N, 2D)$ for downstream tabular classifiers.

### 11.2 Phase 7.2 Deterministic Labeled Synthetic Fraud Fixture (`synthetic_fraud_10k.npz`)
- **Objective:** Implement a fully reproducible, self-contained, labeled synthetic transaction-graph generator producing a non-trivial benchmark fixture with controlled fraud prevalence and zero leakage.
- **Module Implementation:** Authored in [`backend/cpu_engine/synthetic_fraud_generator.py`](file:///D:/FraudGraph%20Intelligence/backend/cpu_engine/synthetic_fraud_generator.py).
  - *Data Generating Process (DGP):*
    - **Node Count & Features:** $N = 10,000$ transactions, $D = 32$ continuous features (`float32`).
    - **Arrival Process:** Poisson inter-arrival times generating monotonic arrival timestamps $t_i$.
    - **Entity Simulation:** Zipfian card ($K=800, \alpha=1.1$) and device ($M=400, \alpha=1.2$) associations.
    - **Fraud Generation:** Exactly 350 fraud transactions (**prevalence: 3.50%**), partitioned into:
      - 20 coordinated fraud rings (245 transactions across compromised card/device clusters).
      - 105 opportunistic/isolated frauds.
    - **Feature Generation:**
      - Features 0–3: Transaction amount, velocity, card-frequency shifts.
      - Features 4–7: Risk/consistency features with subtle mean shifts for fraudulent nodes.
      - Features 8–31: Background Gaussian noise ($\mathcal{N}(0, 1)$).
      - Bounded single-feature correlation: maximum absolute Pearson correlation with label $y$ is strictly bounded ($r \approx 0.120 < 0.35$), ensuring the classification problem is non-trivial and cannot be solved by a simple threshold on a single tabular feature.
    - **Temporal Graph Construction (Lookback Window):**
      - Edges formed between transactions sharing a card or device within a chronological lookback window ($W = 100$ transactions).
      - Strict causal ordering: for every directed edge $(i, j)$ in CSR (where row $i$ points to historical context neighbor $j$), $t_j < t_i$ ($j < i$) strictly holds. Verified across 100% of 231,984 edges.
  - *Graph Dimensions & Density:*
    - Nodes: $10,000$
    - Directed CSR Edges: $231,984$ (average in-degree: $23.20$)
    - Max Degree: $178$
    - Zero-Degree (Isolated) Nodes: $1,280$ ($12.80\%$)
  - *Serialization Schema (`data/fixtures/synthetic_fraud_10k.npz`):*
    - Includes `row_ptr` (`int32`), `col_idx` (`int32`), `features` (`float32`), `labels` (`int32`), `timestamps` (`float64`), and detailed JSON `manifest`.
    - Backward-compatible with `CSRGraph.load()` (which safely ignores extraneous label/timestamp keys while maintaining full structural fidelity).
  - *Chronological Splitting (`temporal_train_test_split`):*
    - **Train (70%):** Nodes 0–6,999 (7,000 nodes, 239 frauds, 3.41% prevalence)
    - **Validation (15%):** Nodes 7,000–8,499 (1,500 nodes, 58 frauds, 3.87% prevalence)
    - **Test (15%):** Nodes 8,500–9,999 (1,500 nodes, 53 frauds, 3.53% prevalence)
    - Strict chronological boundary: $\max(t_{\text{train}}) \le \min(t_{\text{val}}) \le \max(t_{\text{val}}) \le \min(t_{\text{test}})$.

### 11.3 Automated Verification Test Suite (`test_synthetic_fraud_generator.py`)
- Authored 11 comprehensive unit and invariant tests in [`backend/tests/test_synthetic_fraud_generator.py`](file:///D:/FraudGraph%20Intelligence/backend/tests/test_synthetic_fraud_generator.py):
  1. `test_determinism_and_reproducibility`: Exact bitwise reproducibility with identical seed.
  2. `test_seed_divergence`: Distinct seeds produce different graphs.
  3. `test_fraud_prevalence_exact`: Exact count (350) and prevalence (3.50%).
  4. `test_feature_matrix_invariants`: Shape $(10000, 32)$, float32, zero NaNs/Infs.
  5. `test_csr_structural_invariants`: Full compliance with all 8 CSR invariants.
  6. `test_temporal_causality_zero_leakage`: $100\%$ compliance with $t_j \le t_i$.
  7. `test_label_leakage_safeguards`: Labels quarantined; max feature correlation $< 0.35$.
  8. `test_save_load_roundtrip_npz`: Round-trip array fidelity across all 5 arrays + manifest.
  9. `test_compatibility_with_csr_graph_load`: Verifies backward compatibility with standard `CSRGraph.load()`.
  10. `test_temporal_train_test_split`: Verifies chronological bounds, non-overlap, and positive label presence across all splits.
  11. `test_existing_graph_fixtures_remain_unchanged`: Asserts that Phase 3 and Phase 6 fixtures (`toy_graph_5nodes.npz`, `synthetic_tier_a_10k.npz`, `synthetic_skewed_10k.npz`) remain identical and uncorrupted.
- **Preliminary Synthetic Classifier Baseline (Verification Only):** Evaluated standard scikit-learn models on the canonical split with `StandardScaler` fitted strictly on train (zero transductive leakage):
  - Logistic Regression ($C=1.0$): Test ROC-AUC $0.7855$, PR-AUC $0.3088$ (minority prevalence $\pi = 0.0353$).
  - Random Forest ($100$ trees): Test ROC-AUC $0.7786$, PR-AUC $0.2445$.
  - *Scientific Integrity Disclosure:* These preliminary metrics are measured strictly on the synthetic benchmark fixture (`synthetic_fraud_10k.npz`) for verification that the synthetic classification task is non-trivial prior to graph enrichment; they do not represent real-world fraud detection performance.
- **Test Suite Results:** All 11 tests pass in 1.76s; full backend suite passes **93 tests (1 skipped)**.
- **Specification Report:** Authored [`docs/SYNTHETIC_FRAUD_DATASET.md`](file:///D:/FraudGraph%20Intelligence/docs/SYNTHETIC_FRAUD_DATASET.md).

### 11.4 Phase 7.1 & 7.2 Gate Status
- **Phase 7.1 Status:** **`PASSED (Dataset Audit Complete)`**
- **Phase 7.2 Status:** **`PASSED (Deterministic Labeled Synthetic Fixture Verified)`**

---

### 11.5 Phase 7.3 Graph Feature Enrichment Pipeline Implementation & Parity
- **Objective:** Construct multi-representation feature tensors combining raw transaction features with neighbor-aggregated relational embeddings across CPU and native CUDA engines.
- **Module Implementation:** Authored in [`backend/ml_engine/feature_enrichment.py`](file:///D:/FraudGraph%20Intelligence/backend/ml_engine/feature_enrichment.py) and packaged via [`backend/ml_engine/__init__.py`](file:///D:/FraudGraph%20Intelligence/backend/ml_engine/__init__.py).
  - *Supported Representations:*
    1. **`raw`:** $X \in \mathbb{R}^{N \times D}$ (baseline tabular features).
    2. **`mean`:** $[X \,\|\, Y_{\text{mean}}] \in \mathbb{R}^{N \times 2D}$ (concatenation with neighbor mean aggregation).
    3. **`max`:** $[X \,\|\, Y_{\text{max}}] \in \mathbb{R}^{N \times 2D}$ (concatenation with neighbor max aggregation).
    4. **`sum`:** $[X \,\|\, Y_{\text{sum}}] \in \mathbb{R}^{N \times 2D}$ (concatenation with neighbor sum aggregation).
    5. **`multi_operator` / `combined`:** $[X \,\|\, Y_{\text{mean}} \,\|\, Y_{\text{max}}] \in \mathbb{R}^{N \times 3D}$ (multi-operator relational embedding fusion).
  - *Context Policy Support:*
    1. `causal_history` (default): Preserves full historical causal lookup ($t_j < t_i, j < i$) for streaming transaction simulation.
    2. `train_only_context`: Prunes all edges pointing to validation or test nodes ($j \ge \text{train\_end}$), ensuring validation and test nodes aggregate strictly from historical training entities.
  - *Numerical & Invariant Guarantees:*
    - 100% `float32` precision across all representations.
    - Zero NaNs or Infs guaranteed.
    - Zero-degree nodes mapped to exact $\vec{0}$ sentinels ($d_i = 0 \implies [X_{i, :} \;\|\; \vec{0}]$).
    - Ground-truth fraud labels $y$ strictly quarantined and never embedded into feature matrices.
  - *Execution Backends:*
    - Native dispatch supporting `cpu_reference`, `cpu_vectorized` (default), and CUDA runners (`cuda_baseline`, `cuda_coalesced`, `cuda_streams`, `cuda_degree_aware`).
    - CPU reference vs vectorized parity verified ($\text{atol} \le 1.0 \times 10^{-5}$).
- **Automated Verification Suite (`test_feature_enrichment.py`):**
  - 16 dedicated unit and invariant tests in [`backend/tests/test_feature_enrichment.py`](file:///D:/FraudGraph%20Intelligence/backend/tests/test_feature_enrichment.py).
  - Tests pass: **15 passed, 1 skipped** (live CUDA test appropriately skipped locally).
  - Full backend test suite passes: **108 passed, 2 skipped** in 10.35s.
- **Specification Report:** Authored [`docs/FEATURE_ENRICHMENT.md`](file:///D:/FraudGraph%20Intelligence/docs/FEATURE_ENRICHMENT.md).

### 11.6 Phase 7 Gate Status & Next Steps
- **Phase 7.3 Status:** **`PASSED (Feature Enrichment Pipeline Complete & Verified)`**
- **Next Steps:** Proceed to Phase 7.4 (Classifier Baseline Training with Logistic Regression, Random Forest, and LightGBM) to evaluate empirical $\Delta \text{PR-AUC}$ performance gains of enriched representations vs raw tabular features.

