# Changelog — FraudGraph Intelligence (FGI)

All notable changes to the FraudGraph Intelligence platform will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to Semantic Versioning.

---

## [Unreleased]

### Planned for Phase 6: CUDA Optimization Laboratory
- Warp-per-node parallelization (`fgi_aggregator_coalesced_kernel.cu`) for coalesced memory transactions.
- Shared-memory caching and reduction tiles (`__shared__`) for high-degree hub nodes.
- Degree-aware load balancing separating isolated/leaf nodes from dense fraud rings.
- Direct benchmark comparisons against Phase 5 baseline kernel.

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
