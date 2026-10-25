# Phase 9A — Stitch UI and Existing Application Audit

**Project:** FraudGraph Intelligence (FGI)
**Academic Research Title:** *Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation*
**Affiliation:** VIT Vellore — CSI4007 GPU Programming Capstone
**Workspace:** `D:\FraudGraph Intelligence`
**Current HEAD:** `d0ea85a86ffcd5c1d94c9f181903a2ec6543703c`
**Audit Date:** 2026-10-25
**Audit Mode:** Strictly Read-Only Pre-Implementation Evaluation
**Status:** Pre-Implementation Review Gate

---

## 1. Executive Summary & Purpose of the Audit

### 1.1 Purpose
This document provides a comprehensive, evidence-based audit comparing the thirteen (13) Stitch-designed user interface screens located in `stitch_fraudgraph_intelligence_platform` against the actual state of the FraudGraph Intelligence codebase. The primary objective is to establish an unvarnished ground truth before writing any Phase 9 frontend or backend application code, ensuring that all UI representations faithfully reflect the academic research scope, empirical benchmark data, and software-only architecture of the project.

### 1.2 Core Academic Scope & Research Grounding
FraudGraph Intelligence is an academic research software prototype engineered for the VIT Vellore CSI4007 GPU Programming curriculum. The research investigates memory-bandwidth bottlenecks in parallel graph neighbor aggregation (mean, sum, max) across Compressed Sparse Row (CSR) representations of financial transaction networks.

**Academic Reality Safeguards:**
1. **Synthetic Data Only:** All evaluations utilize controlled synthetic transaction graphs (`data/fixtures/synthetic_fraud_10k.npz`, `synthetic_tier_a_10k.npz`, `synthetic_skewed_10k.npz`, and `toy_graph_5nodes.npz`). The system does not connect to live banking networks (FedWire, SWIFT, SEPA, ACH) or process empirical human financial records.
2. **Software-Only Prototype:** Fraud detection operates as a downstream machine learning classification task using frozen feature vectors. The platform does not possess authority or integration to execute automated settlement freezes, transaction intercepts, or financial account quarantines.
3. **Hardware Boundaries:** The local development host runs on Windows 11 with an NVIDIA GeForce MX450 (Compute Capability 7.5, 2,048 MiB VRAM), where native CUDA compilation is unavailable. GPU kernel benchmarks are executed through the remote Google Colab runner (`backend/gpu_engine/colab_runner/`) on an NVIDIA Tesla T4 (40 SMs, 14.9 GB VRAM, Compute Capability 7.5). The repository does not run on 8x NVIDIA H100 SXM5 80GB clusters.
4. **Zero Fabricated Metrics:** The platform must strictly display verified empirical data from Phase 7.4/7.5 evaluation artifacts (`fraud_model_evaluation_20261024_212032.json`) and Phase 8 benchmark sessions (`experiments/results/benchmark_runs/session_20261024_230008_smoke/`). Speculative claims in the Stitch HTML templates (e.g., zk-SNARK cryptographic proofs, 100M+ edges at sub-0.12ms wire speed, FP8 Tensor Core bitonic sorting) are design placeholders and must be clearly replaced or qualified.

### 1.3 Key Audit Findings Summary
- **Frontend Reality:** The existing React frontend (`frontend/src/App.tsx`) is a single-view system health dashboard developed in Phase 1. It displays backend connection status and basic GPU detection. It lacks client-side routing (`react-router-dom`), graph visualizers (Cytoscape.js), charting engines (Recharts/Chart.js/ECharts), and markdown/LaTeX rendering.
- **Backend Reality:** The backend contains high-performance CPU reference/vectorized aggregators (`backend/cpu_engine/`), CUDA kernels (`backend/gpu_engine/`), benchmark orchestrators (`backend/benchmark_engine/`), and machine learning evaluation pipelines (`backend/ml_engine/`). However, the FastAPI layer (`backend/app/`) currently exposes only two endpoints: `GET /` and `GET /api/v1/health`. Every interactive workflow envisioned by the Stitch screens requires dedicated REST endpoints.
- **Design System Cohesion:** The Stitch screens share a striking, cohesive visual identity named **Obsidian Cybernetics** (`stitch_fraudgraph_intelligence_platform/obsidian_cybernetics/DESIGN.md`), featuring dark editorial minimal styling, a strict zero-radius (`0px`) layout, Newsreader serif headings, Geist sans-serif analytical bodies, and high-contrast Electric Cyan (`#00F2FE`) telemetry highlights.
- **Screen Redundancy:** The 13 Stitch screens contain significant overlap. Specifically, three separate screens (`homepage`, `platform_overview`, and `overview_workspace`) represent variations of a landing/overview view. These will be consolidated into a streamlined 8-view application architecture.

---

## 2. Stitch Design Inventory & Taxonomy

The reference directory `stitch_fraudgraph_intelligence_platform` contains exactly fourteen (14) directories: thirteen screen specifications and one design system document.

| Directory Name | Title / Moniker in HTML | Size (`code.html`) | Key UI Elements | Primary Visual Role |
| :--- | :--- | :---: | :--- | :--- |
| `fraudgraph_intelligence_homepage` | FraudGraph Intelligence — Sub-Millisecond Graph Neural Compute | 64,881 B | Hero section, 2D/3D toggle, 4-phase forensic pipeline, FAQ accordion, citation list | Public research overview & platform introduction |
| `fraudgraph_intelligence_platform_overview` | FraudGraph Intelligence — Sub-Millisecond Graph Forensic Telemetry | 46,312 B | Cluster telemetry HUD, 4-metric summary, comparison matrix table, trace export | High-level executive platform summary |
| `fraudgraph_intelligence_overview_workspace` | FraudGraph Intelligence // Forensic Workspace Overview | 59,655 B | Active telemetry KPI bar, live transaction counter, recent runs table, anomaly simulator | Primary operational cockpit / engineer dashboard |
| `fraudgraph_intelligence_graph_explorer_workspace` | FraudGraph Intelligence // Topological Graph Explorer | 51,161 B | Cytoscape/SVG canvas, entity filter sidebar, hop slider, neighborhood inspector | Interactive sub-graph topology visualization |
| `fraudgraph_intelligence_transactions_forensic_workspace` | FraudGraph Intelligence — Transactions Forensic Ledger | 73,664 B | 8-column data grid, multi-faceted filtering bar, pagination, slide-over transaction drawer | Granular tabular transaction audit ledger |
| `fraudgraph_intelligence_cuda_performance_lab` | FRAUDGRAPH // TELEMETRY - CUDA LAB | 52,639 B | Architecture breakdown table, roofline/bandwidth telemetry, kernel comparison cards | Low-level GPU kernel profiling & latency telemetry |
| `fraudgraph_intelligence_experiment_manager` | FraudGraph Intelligence — Experiment Manager | 59,514 B | Experiment configuration form, parameter selectors, historical runs table, side-by-side comparator | Benchmark configuration & reproducibility tracking |
| `fraudgraph_intelligence_evaluation_center` | FraudGraph Intelligence // Dedicated Model Evaluation Center | 61,240 B | Model comparison table, ROC & PR curve viewports, interactive threshold slider ($	au$), confusion matrix | ML model evaluation, precision/recall analysis |
| `fraudgraph_intelligence_dataset_management_ingestion` | FraudGraph Intelligence — Dataset Management & Ingestion | 61,100 B | Schema registry table, ingestion draft dropzone, active dataset ledger, simulation buttons | Graph fixture registry & topology verification |
| `fraudgraph_intelligence_research_reports_results` | FraudGraph Intelligence — Research Reports & Technical Dossiers | 59,437 B | Dossier document viewer, LaTeX/PDF export buttons, empirical metrics table, report selector sidebar | Formal research paper viewer & PDF/LaTeX publisher |
| `fraudgraph_intelligence_system_architecture_formal_methodology` | FraudGraph Intelligence — System Architecture & Formal Methodology | 73,526 B | Academic paper monograph, mathematical formulation, memory layout diagrams, PTX code viewer | Academic monograph & methodology specification |
| `fraudgraph_intelligence_system_health_environment_diagnostics` | System Health and Environment Diagnostics - FraudGraph Intelligence | 49,649 B | Service readiness matrix, local/remote GPU status cards, exception trace log, diagnostic trigger | Hardware/software runtime health & test verification |
| `fraudgraph_intelligence_settings_preferences` | FraudGraph Intelligence — Settings & Workstation Preferences | 65,610 B | Configuration accordion, radio/checkbox controls, table density selector, cache flush actions | Workstation preferences & rendering configuration |
| `obsidian_cybernetics` | *Design System Specification* (`DESIGN.md`) | 8,944 B | Color tokens, typography scales, 12-column grid rules, zero-radius constraint, component patterns | Design language & brand guidelines |

---

## 3. Existing Application State Audit (Ground Truth Baseline)

### 3.1 Frontend Implementation (`frontend/`)
- **Framework & Tooling:** React 18.3.1, TypeScript 5.7.2, Vite 6.1.0.
- **Dependencies Installed:** `clsx` (2.1.1), `lucide-react` (0.475.0), `@types/react` (18.3.18), `@vitejs/plugin-react` (4.3.4).
- **Current State:** A single `App.tsx` file (295 lines) serving as a foundation health dashboard. It renders a header with backend status, three metric cards (Compute Engine, Hardware Accelerator, Research Integrity), and a phase acceptance banner.
- **Missing Infrastructure:**
  - No client-side routing library (`react-router-dom` is absent).
  - No data visualization library (no Recharts, Chart.js, or ECharts for ROC/PR/latency plots).
  - No graph visualization library (no Cytoscape.js or Graphology for topological network rendering).
  - No Markdown or LaTeX rendering engine (no `react-markdown` or `katex` for research dossiers).
  - The subdirectories `src/components/`, `src/hooks/`, `src/pages/`, `src/services/`, and `src/types/` exist on disk but are completely empty.
- **Styling Architecture:** `src/styles/design-tokens.css` is implemented and adheres to Obsidian Cybernetics design tokens, including `--surface-obsidian: #05070A`, `--border-hairline: #21262D`, `--primary: #00F2FE`, and the mandatory universal rule `* { border-radius: 0 !important; }`.

### 3.2 Backend Implementation (`backend/`)
- **FastAPI Layer (`backend/app/`):**
  - Configured with CORS middleware, settings management (`config.py`), and API router mounting.
  - Endpoints implemented:
    1. `GET /`: Basic health check returning project name, academic course, and status.
    2. `GET /api/v1/health`: GPU diagnostics via `nvidia-smi` subprocess querying GPU name, compute capability, VRAM total/free, and local NVCC availability.
  - **Endpoints Missing:** There are no endpoints for graphs, datasets, transactions, benchmarks, experiments, evaluation metrics, or reports.
- **CPU Engine (`backend/cpu_engine/`):**
  - `csr_graph.py`: Fully functional `CSRGraph` class supporting `row_ptr`, `col_idx`, node features, memory profiling, degree computation, neighbor slicing, and validation.
  - `aggregators.py`: Verified CPU reference (Python loop) and vectorized (NumPy) implementations for `mean`, `sum`, and `max` neighbor aggregation.
  - `synthetic_generator.py` & `synthetic_fraud_generator.py`: Deterministic synthetic graph generation with power-law distributions, relational entity structures (`card_id`, `device_id`), and temporal fraud burst patterns.
  - `transaction_graph_builder.py`: Pipeline converting raw transaction tables into CSR graph representations.
- **GPU Engine (`backend/gpu_engine/`):**
  - CUDA source code in `cuda/src/`: Baseline global-memory kernel (`fgi_baseline_kernels.cu`), warp-coalesced kernel (`fgi_coalesced_kernels.cu`), degree-aware kernel (`fgi_degree_aware_kernels.cu`), and multi-stream execution (`main_streams.cu`).
  - `cuda_runner.py`: Python wrapper handling memory allocation and kernel launches.
  - `colab_runner/`: Automated bundle packager, remote execution notebook, and result ingestion scripts for Google Colab Tesla T4 benchmarking.
- **Benchmark Engine (`backend/benchmark_engine/`):**
  - Fully implemented Phase 8 architecture: `benchmark_matrix.py` (workload specifications), `benchmark_telemetry.py` (RAM/VRAM preflight guards and timing boundaries), `benchmark_verifier.py` (dual-tolerance numerical verification), `experiment_manager.py` (session manifests and SHA-256 capture), `benchmark_runner.py` (warmup and statistical execution), and `benchmark_report.py` (reporting).
- **ML Engine (`backend/ml_engine/`):**
  - `feature_enrichment.py`: Aggregates structural neighbor features across temporal partitions.
  - `model_evaluation.py`: Evaluates Logistic Regression and Random Forest classifiers under `causal_history` and `train_only_context` policies, calculating ROC-AUC, PR-AUC, optimal threshold F1, precision, recall, and Brier score.
  - `reporting.py`: Generates canonical publication figures (Figures 1–7) in PNG, SVG, and PDF formats, along with `summary_evaluation_metrics.csv` and `manifest.json`.

### 3.3 Authoritative Data Assets & Cryptographic Provenance
All empirical data displayed in the application must trace back to the following verified repository artifacts:

| Artifact Role | File Path | Format / Details | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Synthetic Fraud Fixture** | `data/fixtures/synthetic_fraud_10k.npz` | 10k nodes, 231,984 edges, 350 frauds (3.5%), dim 32 | `aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430` |
| **Synthetic Tier A Fixture** | `data/fixtures/synthetic_tier_a_10k.npz` | 10k nodes, 94,945 edges, dim 32 | `584b18697688e006ab3cd0a3785783619658e5c03a409ec0f8a2e6705595844e` |
| **Synthetic Skewed Fixture** | `data/fixtures/synthetic_skewed_10k.npz` | 10k nodes, 94,970 edges, power-law degree, dim 32 | `6d4a2671d58c93697a3f4a39885ae99eea62c8bc78a3c4d4fba74092bde8de3c` |
| **Toy Graph Fixture** | `data/fixtures/toy_graph_5nodes.npz` | 5 nodes, 8 edges, dim 4 | `3ea8f702bc1f944d9f164222bad637666307c7be8e4a9767f813e40edc137909` |
| **Canonical ML Evaluation** | `experiments/results/fraud_model_evaluation_20261024_212032.json` | Complete Phase 7.4 cross-validation & test metrics | `155304b7453b15dda1f103817e32471a2ffb2357de9cb298cc7b470d8d0efc90` |
| **Phase 7.5 Metrics CSV** | `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv` | Full-precision evaluation summary table | `86523fd6d6f998602124ba101972eba15dc5b27bc97c852b3dee475dd6f77ba4` |
| **Phase 8 Smoke Session** | `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv` | 36 benchmark runs across CPU and CUDA engines | `2fdbef703e97b80dbfcd67fe17f511404e4b347e4234234ad252e2e946a744e7` |
| **Phase 6 Colab GPU Runs** | `experiments/results/exp_gpu_baseline_tier_a_mean_20261024_142037.json` | Tesla T4 GPU kernel measurements (kernel 0.068 ms) | N/A (JSON record) |

---

## 4. Comprehensive Page-by-Page Audit Matrix

The following matrix audits each of the thirteen (13) Stitch screen designs against actual repository capabilities.

| # | Screen Directory | Page Title | Stitch UI Concept | Existing Frontend | Existing Backend Route | Existing Backend Engine / Data | Missing Backend Endpoint / Logic | Missing Frontend Components | Mock Data vs Canonical Reality | Research Integrity Risk | Grounding Strategy | Classification | Consolidated Nav Target | Priority |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: |
| **1** | `fraudgraph_intelligence_homepage` | Sub-Millisecond Graph Neural Compute | Public landing, hero typography, 2D/3D force visualizer, pipeline overview, citations | None (App.tsx health only) | `GET /` | None | `GET /api/v1/overview/highlights` | Marketing hero, Force graph canvas, FAQ accordion, Citation list | Claims 120M nodes, 0.12ms latency, 99.94% recall, zk-SNARKs | High: Overstated scale and cryptographic claims | Frame as project introduction; ground stats in 10k synthetic benchmarks | `DESIGN_ONLY` | Landing / Overview | P2 |
| **2** | `fraudgraph_intelligence_platform_overview` | Sub-Millisecond Graph Forensic Telemetry | Platform telemetry HUD, 4 metric cards, comparison table, trace JSON export | None | None | None | `GET /api/v1/overview/summary` | Telemetry HUD ribbon, KPI stat cards, Comparison table | Claims 1.42M tx/s, 48.2M V / 312M E, H100 GPU cluster | High: Overstated cluster hardware and enterprise throughput | Consolidate with Workspace Overview | `DESIGN_ONLY` | Merged into Overview | P1 |
| **3** | `fraudgraph_intelligence_overview_workspace` | Forensic Workspace Overview | Operational dashboard, live transaction counter, recent runs table, anomaly simulator | Diagnostic cards in App.tsx | `GET /api/v1/health` | `benchmark_engine`, `evaluation_reporting_7_5` | `GET /api/v1/dashboard/metrics`, `GET /api/v1/runs/recent` | Real-time counter, Interactive anomaly button, Recent runs table | Mock counter increments random integers; mock runs EXP-4105..4108 | Medium: Live streaming ticker implies real bank wire feed | Bind to actual Phase 8 session runs and Phase 7.5 ML summary metrics | `PARTIALLY_IMPLEMENTED` | Forensic Overview (Home) | P0 |
| **4** | `fraudgraph_intelligence_graph_explorer_workspace` | Topological Graph Explorer | Interactive node-link graph visualizer, entity filter, hop slider, sub-graph inspector | None | None | `CSRGraph`, `synthetic_fraud_10k.npz` | `GET /api/v1/graph/topology`, `GET /api/v1/graph/node/{id}` | Cytoscape.js canvas, Graph layout controller, Node inspector drawer | Renders 111 benign, 4 mules, 7 shell corps via static SVG | High: "Forensic Freeze" button implies live banking action | Bind to `synthetic_fraud_10k.npz` subgraphs; disable action buttons | `BACKEND_MISSING` | Graph Explorer | P1 |
| **5** | `fraudgraph_intelligence_transactions_forensic_workspace` | Transactions Forensic Ledger | 8-column data grid, multi-condition filter bar, pagination, slide-over detail drawer | None | None | `synthetic_fraud_10k.npz` (`timestamps`, `features`, `labels`) | `GET /api/v1/transactions`, `GET /api/v1/transactions/{id}` | Virtualized data table, Filter ribbon, Detail slide-over drawer | Displays 148 mock transactions with fake SWIFT/FedWire tags | High: Settlement freeze buttons imply financial intervention | Map directly to 10k synthetic transaction records; remove fake wire tags | `BACKEND_MISSING` | Transaction Ledger | P1 |
| **6** | `fraudgraph_intelligence_cuda_performance_lab` | Telemetry - CUDA Lab | GPU kernel architecture breakdown, roofline analysis, latency comparison, PTX viewer | GPU card in App.tsx | None | `cuda_runner.py`, `colab_runner/`, Phase 6 JSONs | `GET /api/v1/cuda/kernels`, `GET /api/v1/cuda/profile` | Kernel comparison cards, Latency delta table, Roofline chart | Claims Hopper H100 SXM5, FP8 bitonic sort, 182.4x speedup | High: Unqualified claims of H100 hardware and FP8 kernels | Display Tesla T4 Colab results alongside MX450 status; qualify boundaries | `PARTIALLY_IMPLEMENTED` | CUDA Lab | P1 |
| **7** | `fraudgraph_intelligence_experiment_manager` | Experiment Manager | Benchmark configuration form, parameter selectors, historical runs table, run comparator | None | None | `benchmark_runner.py`, `experiment_manager.py`, smoke session | `POST /api/v1/experiments/run`, `GET /api/v1/experiments/history` | Experiment draft form, Historical runs table, Run comparator | Mock experiments EXP-4109..4112 with fake zk-SNARK proofs | Medium: Claims batch cluster dispatch on 8x H100 | Bind to `backend/benchmark_engine/` orchestrator and smoke session JSONs | `PARTIALLY_IMPLEMENTED` | Experiment Manager | P0 |
| **8** | `fraudgraph_intelligence_evaluation_center` | Dedicated Model Evaluation Center | Model comparison table, ROC & PR curves, threshold slider ($	au$), confusion matrix | None | None | `model_evaluation.py`, `reporting.py`, Phase 7.5 artifacts | `GET /api/v1/evaluation/models`, `GET /api/v1/evaluation/curves` | Interactive threshold slider, Confusion matrix card, ROC/PR chart | Displays continuous PR curves (omitted in Phase 7.5 per ADR-007) | Critical: Misrepresents evaluation protocol and PR curve status | Bind directly to Phase 7.4 JSON; display ADR-007 PR omission notice | `BACKEND_MISSING` | Evaluation Center | P0 |
| **9** | `fraudgraph_intelligence_dataset_management_ingestion` | Dataset Management & Ingestion | Schema registry table, drag-and-drop ingestion, dataset ledger, air-gap toggle | None | None | `data/fixtures/*.npz`, `csr_graph.py` | `GET /api/v1/datasets`, `POST /api/v1/datasets/validate` | Schema viewer table, Dropzone component, Simulation presets | Lists FedWire-2025-Q1 (12.4M rows), FIPS 140-3 Level 4 claims | High: Fake enterprise encryption and air-gap enclaves | Present actual 4 NPZ fixtures; validate CSR schema against `CSRGraph` | `BACKEND_MISSING` | Dataset Management | P2 |
| **10** | `fraudgraph_intelligence_research_reports_results` | Research Reports & Technical Dossiers | Research paper viewer, LaTeX/PDF export, empirical metrics table, report index | None | None | `docs/PHASE_7_5_EVALUATION_REPORT.md`, `BENCHMARK_AUTOMATION.md` | `GET /api/v1/reports`, `GET /api/v1/reports/{id}/export` | Markdown reader, Asset download buttons, PDF viewer modal | Cites fictional report TR-2025-04 by Dr. Vance on Hopper H100 | Medium: Fictional authors and inflated performance metrics | Render Phase 7.5 and Phase 8 research reports; serve canonical figures | `BACKEND_MISSING` | Research Reports | P1 |
| **11** | `fraudgraph_intelligence_system_architecture_formal_methodology` | System Architecture & Formal Methodology | Interactive monograph, mathematical equations, CSR diagrams, PTX code viewer | None | None | `docs/SYSTEM_ARCHITECTURE.md`, `ALGORITHM_SPECIFICATION.md` | `GET /api/v1/docs/monograph` | LaTeX formula renderer, Code block with syntax highlighting | Hopper sm_90 register diagrams and direct NIC-to-HBM3e DMA | Medium: Over-engineered micro-architecture claims | Serve existing architecture specifications and CSR memory layout | `DESIGN_ONLY` | System Architecture | P2 |
| **12** | `fraudgraph_intelligence_system_health_environment_diagnostics` | System Health & Environment Diagnostics | Service readiness grid, CUDA environment cards, error logs, diagnostic test runner | Basic status in App.tsx | `GET /api/v1/health` | `health.py`, `test_*.py` test suite | `GET /api/v1/diagnostics/full`, `POST /api/v1/diagnostics/run` | Health status matrix, Exception log viewer, Test suite runner | References cuGraph v24.04 and DGL v2.1.0 (neither is installed) | Medium: Claims dependencies that do not exist in requirements.txt | Wire to actual packages (`scikit-learn`, `numpy`, `scipy`, `matplotlib`) | `PARTIALLY_IMPLEMENTED` | System Health | P0 |
| **13** | `fraudgraph_intelligence_settings_preferences` | Settings & Workstation Preferences | Configuration tabs, theme/density selectors, cache purge buttons, factory reset modal | None | None | Local storage / frontend state | `GET /api/v1/settings`, `POST /api/v1/settings` | Form controls (switches, radios, density selectors), Modal dialog | Includes NVMe buffer purge and enterprise FIPS key rotation | Low: Enterprise settings irrelevant to prototype | Persist UI preferences (density, theme, default dataset) in LocalStorage | `DESIGN_ONLY` | Settings Modal/Page | P3 |

---

## 5. Deep-Dive Workflow Audits

### 5.1 Workflow A: Overview & Operational Cockpit
- **Stitch References:** `fraudgraph_intelligence_overview_workspace`, `fraudgraph_intelligence_homepage`, `fraudgraph_intelligence_platform_overview`.
- **UI Architecture:** KPI metrics ribbon (`42.8k tx/s`, `12.4M records`), live transaction ticker simulator (increments by random offset every 1,000ms), recent benchmark run table, and interactive anomaly simulation button.
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Claims real-time wire tap throughput across 12.4M transactions with sub-20ms clearing threshold intercept points.
  - *Empirical Baseline:* The repository operates on a 10,000-node synthetic transaction graph. There is no streaming WebSocket or live socket feed.
  - *Feasibility Plan:* Consolidate into a single **Forensic Workspace Overview**. Bind the KPI ribbon to actual dataset properties (10,000 nodes, 231,984 edges, 3.5% fraud rate) and display real execution statistics from the Phase 8 benchmark smoke session. Replace the random ticker with a deterministic playback loop over `timestamps` in `synthetic_fraud_10k.npz`.

### 5.2 Workflow B: Topological Graph Explorer
- **Stitch References:** `fraudgraph_intelligence_graph_explorer_workspace`.
- **UI Architecture:** Split-screen layout (65% graph canvas, 35% analytical inspector), node type filters (Mule, Shell Corp, Escrow, Merchant, Benign), degree threshold slider, hop distance slider, and action buttons ("Run Isomorphism", "Forensic Freeze", "Export GEXF/JSON").
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Hardcoded inline SVG nodes rendering an artificial "Synthetic Ring #8849" with 111 benign nodes, 4 mules, and 7 shell corporations. Clicking "Forensic Freeze" displays an alert claiming an interbank settlement freeze was triggered.
  - *Empirical Baseline:* The `CSRGraph` class provides complete topological access via `get_neighbors(node_id)` and `degree(node_id)`. `synthetic_fraud_10k.npz` contains 10,000 nodes, 231,984 directed edges, and 350 labeled fraud nodes.
  - *Feasibility Plan:* Implement an interactive graph visualizer using **Cytoscape.js**. Construct a backend endpoint `GET /api/v1/graph/subgraph` that accepts a `center_node_id`, `max_hops` (1–3), and `max_nodes` (e.g., 100), traversing the CSR arrays to return JSON-formatted node and edge lists. Replace "Forensic Freeze" with an academic "Flag Node for Review" action.

### 5.3 Workflow C: Transaction Forensics Ledger
- **Stitch References:** `fraudgraph_intelligence_transactions_forensic_workspace`.
- **UI Architecture:** Dense, 8-column data grid with column sorting, search input, risk score range slider, classification filter chips ("High-Risk Cycle", "Rapid Fan-Out", "Layered Structuring", "Benign Clear"), and a slide-over transaction detail drawer.
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Displays 148 mock transaction rows with fictitious wire details (e.g., `0x7f4b8849...0192`, `FedWire ISO pacs.008`, `ShellCorp_33 Mule Origin`, `$850,000.00 USD`).
  - *Empirical Baseline:* In `synthetic_fraud_10k.npz`, transactions are indexed by integer IDs ($0 \dots 9,999$), with timestamps spanning 30 days (0 to 2,592,000 seconds), continuous 32-dimensional feature vectors, binary fraud labels (0 or 1), and relational groupings based on `card_id` and `device_id`.
  - *Feasibility Plan:* Expose `GET /api/v1/transactions` with query parameters `page`, `page_size`, `min_risk`, `max_risk`, and `label`. Transform the raw feature vectors into human-interpretable synthetic attributes (synthetic amount derived from feature column 0, synthetic sender/receiver IDs derived from node indices). The slide-over drawer will render the node's local CSR neighborhood and model prediction score.

### 5.4 Workflow D: CUDA Performance Lab
- **Stitch References:** `fraudgraph_intelligence_cuda_performance_lab`.
- **UI Architecture:** Execution breakdown table, memory hierarchy analysis (CSR fetch, shared memory tiling, warp bitonic sort), arithmetic intensity metrics, and benchmark run comparison cards.
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Asserts an active 8x NVIDIA H100 SXM5 cluster executing FP8 Tensor Core bitonic sorting with 182.4x speedup, 4.62ms latency, and zk-SNARK attestation hashes.
  - *Empirical Baseline:* The repository implements three CUDA kernels (`baseline`, `coalesced`, `degree_aware`) and a multi-stream harness. Local hardware is an NVIDIA GeForce MX450 (uncompiled locally due to missing NVCC). Verified GPU execution occurred on an **NVIDIA Tesla T4** in Google Colab (Phase 6), yielding 0.068ms kernel latency for Tier A mean aggregation. Phase 8 smoke benchmarks record CPU execution (`cpu_reference` 114.86ms, `cpu_vectorized` 2.58ms, speedup 44.6x).
  - *Feasibility Plan:* Clearly bifurcate the UI into **Local Host Environment** (showing MX450 status and `SKIPPED_CUDA_UNAVAILABLE`) and **Remote GPU Benchmark Archive** (showing Tesla T4 Colab measurements). Accurately display documented timing boundaries (`cpu_host_timer` vs `gpu_cuda_event`) and variability labels (`STABLE` vs `HIGH_VARIABILITY`).

### 5.5 Workflow E: Experiment Manager
- **Stitch References:** `fraudgraph_intelligence_experiment_manager`.
- **UI Architecture:** Experiment dispatch form (dataset selector, implementation target selector, warmup/repetition inputs), side-by-side run comparator (EXP-4112 vs EXP-4110), and historical experiment ledger.
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Hardcoded mock run entries with arbitrary hashes, status badges ("RUNNING 74%"), and fake cancel/duplicate actions.
  - *Empirical Baseline:* `backend/benchmark_engine/experiment_manager.py` manages reproducible experiment sessions, recording `session_manifest.json`, `summary_table.csv`, configuration hashes, and individual run JSONs in `experiments/results/benchmark_runs/`.
  - *Feasibility Plan:* Expose `GET /api/v1/experiments/sessions` to list completed sessions (such as `session_20261024_230008_smoke`) and `GET /api/v1/experiments/sessions/{id}` to retrieve run details. Provide a functional dispatch button that executes a bounded benchmark run via `backend/benchmark_engine/benchmark_runner.py` in a background thread, streaming progress to the UI.

### 5.6 Workflow F: Model Evaluation Center
- **Stitch References:** `fraudgraph_intelligence_evaluation_center`.
- **UI Architecture:** Architecture comparison table (Modality 1 to 4), interactive decision threshold slider ($	au$), continuous ROC-AUC and PR-AUC curve charts, confusion matrix breakdown, and regulatory model dossier export button ("SR 11-7").
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Displays continuous ROC and PR curves with claimed AUC of 0.994, F1-score of 0.963, and inference latency of 4.6ms on Hopper H100.
  - *Empirical Baseline:* Phase 7.4 evaluation (`fraud_model_evaluation_20261024_212032.json`) evaluated Logistic Regression and Random Forest models across five graph representations (`raw`, `mean`, `max`, `multi_operator`, `sum`). Under `causal_history`, the top test PR-AUC achieved was **0.3315** (Logistic Regression `max`) and top test ROC-AUC was **0.7865**. In accordance with **ADR-007**, continuous PR curves were explicitly omitted due to discrete validation thresholding, and an official omission notice figure was generated (`fig3_pr_curve_omission_notice.png`).
  - *Feasibility Plan:* Replace all mock figures with the verified Phase 7.5 artifacts (`fig1` through `fig7`). Display the ADR-007 omission notice prominently in the PR curve pane. Allow the threshold slider to dynamically update the confusion matrix based on the validation threshold grid ($	au \in [0.01, 0.99]$) recorded in the canonical evaluation JSON.

### 5.7 Workflow G: Dataset Management & Ingestion
- **Stitch References:** `fraudgraph_intelligence_dataset_management_ingestion`.
- **UI Architecture:** Schema registry preview table, drag-and-drop file upload zone, active enclave dataset ledger, air-gap security toggle, and error simulation presets ("Unsupported File", "Oversized File", "Missing Columns").
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Lists fictitious datasets such as `FedWire-Synthetic-2025-Q1` (12.4M rows, 1.84 GB) and cites compliance claims like "FIPS 140-3 LEVEL 4 Enclave Safeguards".
  - *Empirical Baseline:* The repository includes four canonical `.npz` fixtures in `data/fixtures/` validated by `backend/cpu_engine/generate_fixtures.py`.
  - *Feasibility Plan:* Present the four actual dataset fixtures in the registry with their exact node counts, edge counts, feature dimensions, memory footprints, and SHA-256 checksums. Implement a functional file inspection endpoint that validates uploaded `.npz` or `.csv` files against the `CSRGraph` specification, returning topology statistics. Eliminate claims of FIPS enclaves and military-grade air-gaps.

### 5.8 Workflow H: Research Reports & Technical Dossiers
- **Stitch References:** `fraudgraph_intelligence_research_reports_results`.
- **UI Architecture:** Two-column research dossier reader, interactive report selector sidebar, export buttons (.PDF, .TAR.GZ, .JSON, .CSV), LaTeX/BibTeX citation generator, and embedded benchmark metrics table.
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Features fictional report `TR-2025-04` by "Dr. A. Vance and K. Valenti" claiming 182.4x speedup on 10M-node graphs via Hopper FP8 Tensor Cores.
  - *Empirical Baseline:* The repository contains authentic, publication-grade markdown reports: `docs/PHASE_7_5_EVALUATION_REPORT.md`, `docs/BENCHMARK_AUTOMATION.md`, and `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv`.
  - *Feasibility Plan:* Ingest and render the actual project reports in the document viewer. Provide real export links pointing directly to the 21 generated figure files (`.png`, `.svg`, `.pdf`) and summary CSV in `experiments/results/evaluation_reporting_7_5/`. Format BibTeX citations to reference the actual academic capstone project (VIT Vellore CSI4007).

### 5.9 Workflow I: System Health & Environment Diagnostics
- **Stitch References:** `fraudgraph_intelligence_system_health_environment_diagnostics`.
- **UI Architecture:** Service readiness table, CUDA environment status card, kernel exception trace log, diagnostic trigger button ("Run Full Diagnostic"), and remediation playbook links.
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Reports `cuGraph v24.04.00`, `PyTorch v2.3.0+cu124`, and `DGL v2.1.0-cuda` as "AVAILABLE" and linked.
  - *Empirical Baseline:* Neither cuGraph, PyTorch, nor DGL is installed in `backend/requirements.txt`. The backend relies strictly on NumPy, SciPy, scikit-learn, and Matplotlib. Local GPU is an MX450 without local NVCC. The pytest suite comprises 144 tests (142 passed, 2 skipped).
  - *Feasibility Plan:* Update the service readiness table to reflect the true software runtime (Python 3.14, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2, FastAPI 0.115). Connect the diagnostic trigger to a backend service that reports live RAM usage, GPU detection from `nvidia-smi`, and the latest test suite execution results.

### 5.10 Workflow J: Architecture Monograph & Preferences
- **Stitch References:** `fraudgraph_intelligence_system_architecture_formal_methodology`, `fraudgraph_intelligence_settings_preferences`.
- **UI Architecture:** Multi-section academic monograph with mathematical formulations and PTX snippets; comprehensive settings panel with display density toggles, theme selectors, and cache management.
- **Backend Grounding & Discrepancies:**
  - *Stitch Mock:* Methodology paper describes Hopper sm_90 micro-architecture, 228 KB shared memory bypass, and FP8 tensor contractions. Settings include enterprise NVMe buffer purges.
  - *Empirical Baseline:* The formal methodology is documented in `docs/SYSTEM_ARCHITECTURE.md`, `docs/ALGORITHM_SPECIFICATION.md`, and `docs/RESEARCH_SPECIFICATION.md`, focusing on CSR memory layouts and parallel neighbor reduction algorithms.
  - *Feasibility Plan:* Render the true academic methodology documentation using Markdown and KaTeX. Implement the settings view as a functional client-side preference manager (storing table density, active dataset selection, and display luminance in `localStorage`).

---

## 6. Obsidian Cybernetics Design System Specification

The user interface must strictly conform to the **Obsidian Cybernetics** design language defined in `stitch_fraudgraph_intelligence_platform/obsidian_cybernetics/DESIGN.md`.

### 6.1 Color Palette Architecture
The palette is built upon a high-contrast dark canvas that eliminates ambient light pollution to emphasize computational vectors and telemetry.

```css
:root {
  /* Surface Depths */
  --surface-obsidian: #05070A;       /* Ground plane, graph canvas, root background */
  --surface-graphite: #0D1117;       /* Workspace panels, data tables, containers */
  --surface-raised: #161B22;         /* Dropdowns, inspectors, active code modals */
  --surface-container-high: #272a2f; /* Highlighted cards and elevated headers */

  /* Hairline Borders */
  --border-hairline: #21262D;        /* Structural 1px division grid */
  --border-active: #00F2FE;          /* Active focus and high-priority borders */

  /* Luminescent Accents */
  --primary: #00F2FE;                /* Electric Cyan: active compute, traversal paths */
  --primary-hover: #0AE8B7;          /* Mint hover state */
  --on-primary: #05070A;             /* High contrast dark text on cyan buttons */

  --secondary: #0AE8B7;              /* Phosphor Mint: verified throughput, healthy status */
  --tertiary: #EFECE6;               /* Warm Alabaster: editorial headlines and monograph text */

  --alert-critical: #FF3B30;         /* Spectral Red: fraud anomalies and fatal errors */
  --metric-dim: #484F58;             /* Inactive labels and muted telemetry */
  --text-primary: #E1E2E8;           /* Primary analytical body text */
  --text-muted: #8B949E;             /* Secondary descriptions and metadata */
}
```

### 6.2 Strict Zero-Radius Geometric Rule
An absolute architectural constraint across the platform is the **uncompromising zero-radius rule**:
```css
* {
  border-radius: 0 !important;
}
```
Every container, button, modal, tooltip, chip, tag, and input field must terminate in crisp 90-degree right angles. Selection controls adhere to this geometry:
- **Checkboxes:** Absolute 14x14px squares with 1px border (`#21262D`), filled with `#00F2FE` when checked.
- **Radio Buttons:** Concentric nested squares (an inner 6x6px solid square centered inside the outer 14x14px border) rather than circles.

### 6.3 Typographic Hierarchy
The typographic architecture pairs human editorial investigation with machine computation:
1. **Editorial Headlines (`Newsreader`):** Classical serif typeface (`400`/`500` weight) applied to dossier titles, research questions, and major panel headlines. Evokes institutional authority.
2. **Analytical Engine (`Geist`):** Cold, geometric sans-serif for UI chrome, metric readouts, dense tables, and form controls. Legible down to 11px.
3. **Telemetry & Code (`Geist Mono` / `label-code`):** Monospaced uppercase with expanded letter-spacing (`0.08em`) for memory states, execution hashes, latencies, and coordinates.

### 6.4 Component Inventory for Phase 9B Scaffolding
To ensure code reusability across all views, the following zero-radius atomic components will be created in `frontend/src/components/common/`:
- `CyberButton`: Primary (Cyan fill), Secondary (Outline Hairline), and Terminal Micro-Action.
- `Panel`: Graphite container with 1px hairline border and partitioned header ribbon.
- `TelemetryCard`: Compact metric block displaying uppercase label, large readout, and trend indicator.
- `DataTable`: Monochromatic virtualized grid with 1px row dividers and hover highlight.
- `StatusBadge`: Rectangular status chip with semantic color schemes (Critical Red, Verified Mint, Inactive Dim).
- `FilterBar`: Input search field paired with categorical filter toggle buttons.
- `RangeSlider`: Custom styled range control for thresholds and hop distances.
- `Modal`: Floating graphite overlay with 12px backdrop blur and top-edge highlight.
- `Tabs`: Segmented navigation control with active bottom-border indicator.

---

## 7. Catalog of Speculative / Unsupported Claims & Research Grounding

The Stitch design screens contain marketing copy, hypothetical features, and enterprise claims that exceed the actual scope of this academic prototype. The table below catalogs each unsupported claim, identifies its source screen, explains why it must be corrected, and specifies the grounded research replacement.

| # | Stitch Screen | Speculative / Unsupported Claim Text | Empirical Ground Truth | Mandatory Grounding Strategy |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `cuda_performance_lab`, `experiment_manager` | "H100 SXM5 80GB (NODE-0)", "FP8 Tensor Core Bitonic Sorter", "182.4x Speedup" | Host runs on Windows with NVIDIA GeForce MX450 (2GB VRAM). Remote GPU runs executed on NVIDIA Tesla T4 in Google Colab (Phase 6). No Hopper H100 hardware or FP8 kernels exist in the repository. | Display actual local hardware (GeForce MX450) and label remote benchmark results with their true platform: "Google Colab NVIDIA Tesla T4 (Phase 6 Archive)". Display measured speedups (e.g., 44.6x CPU vectorized vs CPU reference). |
| **2** | `evaluation_center` | Continuous PR curve plot with claimed PR-AUC of 0.994 and F1 of 0.963 | Downstream evaluation on synthetic fraud dataset achieved max PR-AUC of 0.3315. Continuous PR curve was explicitly omitted per ADR-007 due to discrete threshold evaluation protocol. | Replace mock PR chart with `fig3_pr_curve_omission_notice.png`. Display true canonical test metrics from `summary_evaluation_metrics.csv` (PR-AUC 0.12–0.33, ROC-AUC 0.72–0.79). |
| **3** | `transactions_forensic_workspace`, `homepage` | "Trigger Immediate Settlement Intercept", "AUTO-FROZEN (PENDING REVIEW)", "FedWire ISO pacs.008", "SWIFT MT103" | The project is a software-only academic prototype. It has no integration with central banks, FedWire, SWIFT, or payment clearinghouses. Data is synthetic. | Remove interbank wire terminology and settlement action buttons. Label records as "Synthetic Transaction Records" with analytical actions: "Inspect Neighborhood", "Filter Subgraph", "Flag for Academic Review". |
| **4** | `dataset_management_ingestion` | "FedWire-Synthetic-2025-Q1 (12.4M records)", "FIPS 140-3 LEVEL 4 Enclave Safeguards", "Air-Gap Enforce" | The evaluation dataset is `synthetic_fraud_10k.npz` (10,000 nodes, 231,984 edges). No multi-million record enterprise datasets or hardware security enclaves exist. | Present the 4 authentic fixtures (`synthetic_fraud_10k`, `synthetic_tier_a_10k`, `synthetic_skewed_10k`, `toy_graph_5nodes`). Remove references to FIPS and military-grade enclaves. |
| **5** | `cuda_performance_lab`, `experiment_manager` | "BIT-EXACT [zk-SNARK: 0x8a91]", "ECDSA-P256 ATTESTED", "Export Encrypted zk-Proof" | No zero-knowledge proofs or cryptographic attestation circuits are implemented. Verification is handled via numerical comparison (`atol=1e-5`, `rtol=1e-5`) and SHA-256 manifests. | Replace "zk-SNARK" with "SHA-256 Checksum" and "Dual-Tolerance Numerical Verification (rtol=1e-5, atol=1e-5)". Show real verification statuses: `VERIFIED_PASS` or `MEASURED`. |
| **6** | `overview_workspace` | Incremental real-time transaction ticker incrementing by 38–82 tx/s continuously | No streaming data source or WebSocket exists; all fixtures are static `.npz` files stored locally. | Replace random counter with static dataset size indicator (10,000 transactions) or an opt-in simulated timeline playback bounded by `timestamps` in the fixture. |
| **7** | `system_health_environment_diagnostics` | "cuGraph v24.04.00 AVAILABLE", "PyTorch v2.3.0+cu124 AVAILABLE", "DGL v2.1.0 AVAILABLE" | cuGraph, PyTorch, and DGL are not present in `backend/requirements.txt`. The system uses standard scientific Python (`numpy`, `scipy`, `scikit-learn`, `matplotlib`). | Replace with verified package inventory: NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2, and Python 3.14.3. |

---

## 8. Screen Consolidation & Unified Navigation Architecture

### 8.1 Analysis of Redundancies
The Stitch inventory includes 13 screens with notable functional overlap:
- **Overview Overlap:** `homepage`, `platform_overview`, and `overview_workspace` all provide executive-level dashboards. Having three separate overview pages creates user confusion.
- **Dossier & Methodology Overlap:** `research_reports_results` and `system_architecture_formal_methodology` both present academic documentation.

### 8.2 Consolidated 8-View Application Sitemap
To deliver a coherent, professional workstation, the platform will be structured into eight (8) primary views managed by a top navigation bar, with secondary diagnostic and settings panels accessible as modals or flyouts:

```
[FRAUDGRAPH INTELLIGENCE PLATFORM]
  ├── Top Navigation Bar (Always Visible)
  │     ├── [1] Overview (Forensic Workspace Cockpit)
  │     ├── [2] Graph Explorer (Cytoscape Topological Subgraph Inspection)
  │     ├── [3] Transaction Ledger (Tabular Forensic Audit Grid)
  │     ├── [4] CUDA Performance Lab (Kernel Profiling & Hardware Telemetry)
  │     ├── [5] Experiment Manager (Benchmark Matrix & Reproducibility)
  │     ├── [6] Evaluation Center (ML Classifier Metrics & ROC/PR Analysis)
  │     ├── [7] Dataset Registry (Fixture Metadata & Schema Verification)
  │     └── [8] Research Reports (Technical Monograph & Publication Artifacts)
  │
  ├── Utility Controls (Top Right Cluster)
  │     ├── System Health Drawer (Live Hardware/Runtime Diagnostics)
  │     ├── Settings Modal (Table Density, Theme, Local Storage Prefs)
  │     └── Project Charter / Academic Disclaimers
```

### 8.3 Consolidation Mapping
1. **Overview View (`/` or `/overview`):** Merges `overview_workspace` with the explanatory highlights of `platform_overview`. Serves as the operational landing page.
2. **Graph Explorer (`/explorer`):** Direct implementation of `graph_explorer_workspace` backed by Cytoscape.js.
3. **Transaction Ledger (`/transactions`):** Direct implementation of `transactions_forensic_workspace` backed by `synthetic_fraud_10k.npz`.
4. **CUDA Performance Lab (`/cuda-lab`):** Direct implementation of `cuda_performance_lab` backed by Colab Tesla T4 archive and local MX450 telemetry.
5. **Experiment Manager (`/experiments`):** Direct implementation of `experiment_manager` backed by `backend/benchmark_engine/`.
6. **Evaluation Center (`/evaluation`):** Direct implementation of `evaluation_center` backed by Phase 7.5 figures and Phase 7.4 evaluation JSON.
7. **Dataset Registry (`/datasets`):** Direct implementation of `dataset_management_ingestion` backed by `data/fixtures/`.
8. **Research Reports (`/reports`):** Direct implementation of `research_reports_results` merged with the monograph viewer from `system_architecture_formal_methodology`.
9. **System Health Modal (`/diagnostics`):** Derived from `system_health_environment_diagnostics`.
10. **Settings Modal (`/settings`):** Derived from `settings_preferences`.

---

## 9. Phased Implementation Roadmap (Phases 9B – 9E)

Implementation will proceed across four structured sub-phases, adhering strictly to pre-commit review gates.

### Phase 9B: Core Foundation, Routing, Shared Components & Backend API Scaffolding (Priority P0)
- **Frontend:**
  - Install `react-router-dom` in `frontend/package.json`.
  - Implement top navigation header, view router, and layout shell.
  - Build reusable zero-radius Obsidian Cybernetics components in `src/components/common/` (`CyberButton`, `Panel`, `TelemetryCard`, `DataTable`, `Badge`, `Tabs`, `Slider`).
- **Backend:**
  - Create FastAPI routers in `backend/app/api/v1/endpoints/`:
    - `dashboard.py`: Endpoint for system overview KPI telemetry.
    - `health.py`: Enhance existing health check with detailed package and environment telemetry.
- **Verification Gate:** Frontend compiles with zero TypeScript errors (`npm run build`); backend tests pass.

### Phase 9C: Core Operational & Research Workspaces (Priority P1)
- **Frontend & Backend Workflows:**
  - **Evaluation Center:** Implement `endpoints/evaluation.py` to serve Phase 7.4 metrics and Phase 7.5 figure assets. Build React view with interactive threshold slider and ADR-007 PR omission notice.
  - **CUDA Lab & Experiment Manager:** Implement `endpoints/experiments.py` to list benchmark sessions and trigger smoke runs. Build CUDA telemetry dashboard displaying Colab Tesla T4 results and local MX450 status.
  - **Research Reports:** Implement `endpoints/reports.py` to serve markdown reports and publication assets. Build document viewer with PDF/LaTeX export links.
- **Verification Gate:** Verification against canonical JSON hashes; zero hardcoded mock metrics.

### Phase 9D: Graph Explorer & Transaction Forensic Ledger (Priority P2)
- **Frontend & Backend Workflows:**
  - Install `cytoscape` (and type definitions) in `frontend/package.json`.
  - Implement `endpoints/graph.py` to extract sub-graphs around high-risk nodes from `synthetic_fraud_10k.npz`.
  - Implement `endpoints/transactions.py` with pagination, filtering, and sorting over the 10,000 synthetic records.
  - Build the interactive Cytoscape canvas with node selection, hop traversal, and slide-over inspector drawer.
- **Verification Gate:** Graph rendering smoothly handles up to 250 nodes; transaction table virtualizes cleanly.

### Phase 9E: Public Monograph, Diagnostics & Settings Refinement (Priority P3)
- **Frontend & Backend Workflows:**
  - Implement Dataset Management view connected to `data/fixtures/`.
  - Implement System Health flyout drawer connected to live diagnostic probes and test-suite status.
  - Implement Settings modal storing preferences in `localStorage`.
  - Embed the formal academic monograph with KaTeX formula rendering.
- **Verification Gate:** End-to-end integration test of all 8 views; full backend test suite pass (144 tests); clean production build.

---

## 10. Required FastAPI Endpoints & Data Contracts

To support the 8-view architecture, the following FastAPI endpoints will be introduced in `backend/app/api/v1/endpoints/`:

```
backend/app/api/v1/endpoints/
  ├── health.py            [EXISTS] GET /api/v1/health (System runtime & GPU detection)
  ├── dashboard.py         [NEW]    GET /api/v1/dashboard/summary (Overview KPIs & active dataset stats)
  ├── graph.py             [NEW]    GET /api/v1/graph/subgraph (CSR sub-graph extraction for Cytoscape)
  │                                 GET /api/v1/graph/node/{id} (Individual node degree and feature summary)
  ├── transactions.py      [NEW]    GET /api/v1/transactions (Paginated synthetic transaction ledger)
  │                                 GET /api/v1/transactions/{id} (Detailed transaction attribute dossier)
  ├── cuda_lab.py          [NEW]    GET /api/v1/cuda/kernels (Kernel inventory, memory hierarchy specs)
  │                                 GET /api/v1/cuda/benchmarks/colab (Ingested Phase 6 Tesla T4 measurements)
  ├── experiments.py       [NEW]    GET /api/v1/experiments/sessions (List historical benchmark sessions)
  │                                 GET /api/v1/experiments/sessions/{id} (Retrieve session manifest & runs)
  │                                 POST /api/v1/experiments/run (Dispatch a bounded smoke benchmark run)
  ├── evaluation.py        [NEW]    GET /api/v1/evaluation/summary (Phase 7.4/7.5 canonical metric tables)
  │                                 GET /api/v1/evaluation/thresholds (Validation threshold grid & confusion matrix)
  │                                 GET /api/v1/evaluation/figures/{id} (Serve figure PNG/SVG/PDF assets)
  ├── datasets.py          [NEW]    GET /api/v1/datasets (List NPZ fixtures with sizes, nodes, edges, hashes)
  │                                 POST /api/v1/datasets/validate (Validate uploaded NPZ/CSV structure)
  └── reports.py           [NEW]    GET /api/v1/reports (List available research documents)
                                    GET /api/v1/reports/{id} (Retrieve rendered Markdown report)
```

### 10.1 Key Pydantic Data Contracts (Schemas)

```python
# Evaluation Summary Schema
class ModelEvaluationMetrics(BaseModel):
    model_name: str
    representation: str
    feature_dim: int
    val_pr_auc: float
    val_roc_auc: float
    test_pr_auc: float
    test_roc_auc: float
    val_optimal_threshold: float
    test_f1: float
    test_precision: float
    test_recall: float
    test_true_negative: int
    test_false_positive: int
    test_false_negative: int
    test_true_positive: int

# Cytoscape Graph Subgraph Schema
class GraphNode(BaseModel):
    id: str
    label: str
    is_fraud: bool
    degree: int
    features_preview: list[float]

class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float

class SubgraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    center_node_id: int
    hop_count: int
    total_nodes_in_subgraph: int
```

---

## 11. Acceptance Gates, Verification Plan & Safety Rules

Before any code modification in subsequent phases:
1. **Mandatory Safety Rule:** Maintain clean git working tree; never modify or delete reference files in `stitch_fraudgraph_intelligence_platform/`.
2. **Type Safety & Build Cleanliness:** The React application must compile with zero errors under `npm run typecheck` (`tsc -b`) and `npm run build`.
3. **Backend Test Suite Integrity:** The existing 144 backend tests must remain green (142 passed, 2 skipped). New tests must be added in `backend/tests/` for all new FastAPI endpoints.
4. **Data Integrity Assurance:**
   - No mock data hardcoded in production components.
   - All ML numbers must match `summary_evaluation_metrics.csv` to 4 decimal places.
   - All benchmark latencies must match `summary_table.csv`.
   - The ADR-007 PR curve omission notice must be displayed instead of a fabricated continuous PR curve.
5. **No Speculative Feature Drift:** Disable or remove buttons claiming to trigger financial settlement blocks, zk-SNARK generation, or 8x H100 cluster deployments.

---

## 12. Audit Sign-off & Pre-Implementation Gate Checklist

- [x] All thirteen (13) Stitch screen HTML files fully inspected.
- [x] Design system specification (`obsidian_cybernetics/DESIGN.md`) audited and tokenized.
- [x] Existing frontend (`App.tsx`, `package.json`, `styles/`) audited.
- [x] Existing backend engines (`cpu_engine`, `gpu_engine`, `benchmark_engine`, `ml_engine`) audited.
- [x] Data fixtures and canonical result archives cryptographically verified via SHA-256.
- [x] 15-column master implementation matrix constructed.
- [x] 10 user-facing workflows deeply analyzed with grounding strategies.
- [x] Catalog of unsupported claims and academic corrections documented.
- [x] Sitemap consolidated from 13 screens to 8 cohesive views.
- [x] Phased roadmap (9B through 9E) and API contracts specified.
- [x] Repository state verified clean (`git status`, HEAD `d0ea85a86ffcd5c1d94c9f181903a2ec6543703c`).

**Conclusion:** Phase 9A audit is complete and verified. The repository is ready to proceed to Phase 9B upon user authorization.
