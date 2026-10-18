# Changelog — FraudGraph Intelligence (FGI)

All notable changes to the FraudGraph Intelligence platform will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to Semantic Versioning.

---

## [Unreleased]

### Planned for Phase 2: Literature Review & Research Specification
- Formulation of mathematical research plan for neighbor aggregation on power-law transaction graphs.
- Literature review synthesizing GNN memory bottlenecks, SpMM acceleration, and fraud detection benchmarks.

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
- **Documentation Suite:** Authored foundational technical documentation:
  - `README.md`
  - `docs/PROJECT_CHARTER.md`
  - `docs/RESEARCH_PLAN.md`
  - `docs/RESEARCH_GAP.md`
  - `docs/SYSTEM_ARCHITECTURE.md`
  - `docs/DATA_DICTIONARY.md`
  - `docs/ALGORITHM_SPECIFICATION.md`
  - `docs/API_CONTRACT.md`
  - `docs/DESIGN_SYSTEM.md`
  - `docs/ENVIRONMENT_SETUP.md`
  - `docs/EXPERIMENT_PROTOCOL.md`
  - `docs/TEST_PLAN.md`
  - `docs/SECURITY_AND_PRIVACY.md`
  - `docs/KNOWN_LIMITATIONS.md`
  - `docs/DECISIONS.md`
  - `docs/PHASE_STATUS.md`
  - `docs/CHANGELOG.md`

---

## [0.0.1] - 2026-10-18 — Phase 0: Workspace Audit & Feasibility

### Added
- Comprehensive system audit of Windows 11 host, 116.8 GB free disk on D:, Python 3.14, Node.js v24, Git, and NVIDIA GeForce MX450 GPU (Turing, Compute Capability 7.5).
- Identified local absence of `nvcc` and MSVC build tools; formulated two-tier Google Colab execution strategy for CUDA acceleration.
- Audited Stitch MCP connection and verified access to project `9197530824498398398`.
- Initialized `docs/ENVIRONMENT_SETUP.md` and `docs/PHASE_STATUS.md`.
