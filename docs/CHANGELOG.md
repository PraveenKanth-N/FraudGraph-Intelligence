# Changelog — FraudGraph Intelligence (FGI)

All notable changes to the FraudGraph Intelligence platform will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to Semantic Versioning.

---

## [Unreleased]

### Planned for Phase 3: Dataset Pipeline & Graph Representation
- Implementation of CSR graph representation builder (`backend/cpu_engine/csr_graph.py`).
- Synthetic power-law graph generator for CI/CD test fixtures.
- IEEE-CIS / Elliptic data ingestion pipeline with strict temporal splitting and target leakage prevention.

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
