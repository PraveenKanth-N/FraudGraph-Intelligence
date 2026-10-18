# Changelog — FraudGraph Intelligence (FGI)

All notable changes to the FraudGraph Intelligence platform will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to Semantic Versioning.

---

## [Unreleased]

### Planned for Phase 4: CPU Reference Implementation
- NumPy and SciPy reference parallel neighbor aggregation engine (`mean`, `sum`, `max`).
- Baseline graph-augmented fraud classification pipeline using scikit-learn.
- End-to-end CPU performance timing harness and numerical verification ground truth.

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
