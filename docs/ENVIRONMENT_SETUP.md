# Environment Setup and System Audit — FraudGraph Intelligence (FGI)

## 1. Overview and Workspace Identity

- **Project Name:** FraudGraph Intelligence (FGI)
- **Academic Course:** VIT Vellore — CSI4007 GPU Programming
- **Final Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation
- **Workspace Root:** `D:\FraudGraph Intelligence`
- **Audit Timestamp:** 2026-10-18T21:20:00+05:30
- **Auditor / Agent:** Antigravity (Google DeepMind)
- **Version:** 0.1.0 (Phase 1 Baseline)

---

## 2. Hardware and Operating System Diagnostics

| Component | Target / Required Specification | Detected Local System Specification | Status |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 10/11 64-bit | Microsoft Windows 11 (NT build 10.0.26200.0) | **PASS** |
| **Drive D: Capacity** | Dedicated workspace partition | Total: 122.07 GB, Free: 116.83 GB, Used: 5.24 GB | **PASS** (116.8 GB available) |
| **Shell Environment** | PowerShell 5.1+ / 7+ | PowerShell 5.1.26200.5074 | **PASS** |
| **Dedicated GPU** | NVIDIA CUDA-capable GPU | NVIDIA GeForce MX450 (Device 0000:01:00.0) | **DETECTED** |
| **GPU Architecture** | Compute Capability >= 6.0 | Turing Architecture, Compute Capability 7.5 (`sm_75`) | **COMPATIBLE** |
| **VRAM Capacity** | >= 2 GB dedicated memory | 2048 MiB Total, 1907 MiB Free | **PASS** (Sufficient for subgraphs) |
| **NVIDIA Driver** | Recent NVIDIA Display Driver | Version 581.83 (supports CUDA Runtime up to 13.0) | **PASS** |

---

## 3. Toolchain and Language Runtimes

| Tool / Runtime | Target Version | Detected System Version & Path | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Git** | >= 2.30 | `git version 2.53.0.windows.2` (`C:\Program Files\Git`) | **PASS** |
| **Python** | >= 3.10 | `Python 3.14.3` (`C:\Program Files\Python314\python.exe`) | **PASS** |
| **Local Virtual Env** | Project-specific `.venv` | `D:\FraudGraph Intelligence\.venv` (Python 3.14.3) | **INSTALLED** |
| **Python Packages** | Core scientific & API stack | NumPy 2.5.4, SciPy 1.18.1, Pandas 3.0.6, Scikit-Learn 1.9.1, FastAPI 0.144.0, Uvicorn 0.54.0, Pydantic 2.14.1, Pytest 9.1.1, HTTPX 0.28.1 | **VERIFIED (PASS)** |
| **Node.js** | >= 18.0 | `v24.14.0` (`C:\Program Files\nodejs\node.exe`) | **PASS** |
| **npm** | >= 9.0 | `11.9.0` | **PASS** |
| **Frontend Packages**| React + Vite + TypeScript | React 18.3.1, TypeScript 5.7.2, Vite 6.4.4, Lucide React (75 packages audited, 0 vulnerabilities) | **VERIFIED (PASS)** |
| **CUDA Compiler (`nvcc`)**| CUDA Toolkit (12.x or 13.x) | **Not installed / Not found on PATH** | **LOCAL BLOCKER** (See Fallback Plan) |
| **C++ Host Compiler (`cl.exe`)**| MSVC (VS 2022 / Build Tools) | **Not installed / Not found** | **LOCAL BLOCKER** |
| **Stitch MCP Server** | Google Stitch Integration | Connected & operational (accessed project `9197530824498398398`) | **PASS** |
| **Stitch Export Asset**| `stitch_fraudgraph_intelligence_platform` | Exported & present in workspace (`obsidian_cybernetics/DESIGN.md` + 13 screen designs) | **PASS** |

---

## 4. CUDA Execution Feasibility Analysis

### Local Environment Findings
1. The hardware contains an NVIDIA GeForce MX450 GPU with Compute Capability 7.5 (`sm_75`) and Driver 581.83.
2. The NVIDIA CUDA Toolkit (compiler `nvcc`, headers, and libraries) and Visual Studio C++ build tools (`cl.exe`) are not installed on the local system.
3. Because `nvcc` and MSVC are absent, compiling and executing native `.cu` CUDA kernels directly on the local Windows OS is currently blocked.

### Two-Tier Execution Strategy

```
+-----------------------------------------------------------------------------------+
|                            FRAUDGRAPH INTELLIGENCE PLATFORM                       |
+-----------------------------------------+-----------------------------------------+
|                TIER 1                   |                 TIER 2                  |
|          LOCAL WORKSPACE                |           CUDA ACCELERATION             |
|       (Windows / D: Drive)              |           (Colab / Cloud GPU)           |
+-----------------------------------------+-----------------------------------------+
| - FastAPI REST Backend (/api/v1)        | - Google Colab GPU runtime (T4 / A100)  |
| - React + Vite + TS Frontend            | - Native NVCC compilation (sm_75/sm_70) |
| - CPU Reference Engine (CSR Aggregation)| - Baseline CUDA C++ kernels             |
| - Dataset ETL & Graph Construction      | - Memory-coalesced & shared-memory opt  |
| - Evaluation Metrics (F1, PR-AUC, etc.) | - CUDA Event micro-benchmarking         |
| - Graph Explorer (Cytoscape.js)         | - Deterministic test output export      |
| - Experiment Persistence & Export       | - Verified benchmark JSON/CSV outputs   |
+-----------------------------------------+-----------------------------------------+
                                          |
                      Packaging & Verification Flow:
     [Local Graph CSR Data] ---> [Colab Package (.tar.gz)] ---> [Execute Kernels]
                                                                        |
     [Verified Benchmarks & Embeddings JSON] <-------------------------+
```

1. **Local Windows Execution (Core Platform & CPU Reference):**
   - The entire frontend, backend, dataset pipeline, graph construction, CPU reference neighbor aggregation, fraud evaluation, and experiment dashboard run natively on `D:\FraudGraph Intelligence`.
2. **Google Colab Execution Workflow (CUDA Kernels & Benchmarking):**
   - A specialized, reproducible packaging utility (`backend/gpu_engine/colab_runner/`) packages:
     - CSR adjacency data (`row_ptr`, `col_idx`, `node_features`).
     - CUDA C++ source code (`fgi_aggregator_baseline.cu`, `fgi_aggregator_coalesced.cu`, `fgi_aggregator_shared.cu`).
     - Automated harness script executing kernels with CUDA events, verifying floating-point numerical equivalence against CPU fixtures, and dumping structured benchmark logs.
   - Verified benchmark results and execution logs are synced back to `D:\FraudGraph Intelligence\experiments\results\`.
   - In accordance with the Master Engineering Directive, GPU metrics will **never** be fabricated or faked. If a run has not yet been executed on the GPU, it will display an explicit `STATUS: AWAITING_GPU_RUN` indicator.

---

## 5. UI Design & Reference Alignment Plan

### 5.1 Google Stitch Design System Asset
- **Connected Project:** `projects/9197530824498398398`
- **Workspace Export:** `D:\FraudGraph Intelligence\stitch_fraudgraph_intelligence_platform`
- **Design System Identity:** *Obsidian Cybernetics* (`obsidian_cybernetics/DESIGN.md`)
- **Base Canvas:** Pure Jet Black (`#000000`) and Deep Obsidian (`#05070A`)
- **Card Containers:** Deep Graphite (`#0D1117`) and Raised Surface (`#161B22`) with 1px hairline stroke (`#21262D`)
- **Status Accents:** Electric Cyan (`#00F2FE`), Phosphor Mint (`#0AE8B7`), Spectral Red (`#FF3B30`), Alabaster Cream (`#EFECE6`)
- **Typography:** Newsreader (Editorial serif monograph headlines) paired with Geist (Analytical engine, tabular data, labels)

### 5.2 Recorded Reference Assets
- **Video Reference:** `C:\Users\nivet\Videos\Screen Recordings\Screen Recording 2026-10-10 232034.mp4` (Minimalist SaaS / Squarespace editorial aesthetic, clean navbar, topic header pill, clutter-free controls).
- **Secondary Video:** `C:\Users\nivet\Videos\Screen Recordings\Screen Recording 2026-10-16 192814.mp4`
- **Reference URL:** `https://teamcomputers.com/managed-services/` (Enterprise information hierarchy, domain cards, executive situational awareness).
