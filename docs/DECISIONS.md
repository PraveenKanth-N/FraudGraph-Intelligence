# Architectural Decision Records (ADRs) — FraudGraph Intelligence (FGI)

## ADR-001: Separation of Presentation, Orchestration, and Compute Engines

- **Status:** Accepted
- **Context:** The system requires real-time graph visualization (Cytoscape.js), interactive forensic investigation, high-throughput GPU kernel benchmarking, and automated fraud evaluation. Bundling all logic into a monolithic script or relying on Streamlit would degrade GPU timing precision, lock the UI thread during multi-hop graph traversals, and violate the Master Engineering Directive.
- **Decision:** Establish three strictly decoupled tiers:
  1. *Presentation Tier:* React 18 + TypeScript + Vite with the Obsidian Cybernetics design system.
  2. *Orchestration Tier:* Python 3.14 + FastAPI REST API with non-blocking async routes under `/api/v1`.
  3. *Compute Engines:* Isolated CPU reference engine (NumPy/SciPy) and CUDA C++ engine with dual-tier local/Colab execution adapter.
- **Consequences:** Clean separation of concerns, independent testability, and zero UI stuttering during heavy tensor compute.

---

## ADR-002: Compressed Sparse Row (CSR) as Canonical Graph Adjacency Format

- **Status:** Accepted
- **Context:** Graph representation in GPU memory must minimize memory footprint while enabling fast contiguous access to neighbor lists during parallel reduction. Coordinate list (COO) requires sorting and duplicate key handling, while adjacency matrices consume $O(N^2)$ memory, which would immediately exceed the 2 GB VRAM ceiling.
- **Decision:** Utilize Compressed Sparse Row (CSR) format with 0-indexed contiguous `row_ptr` and `col_idx` arrays.
- **Consequences:** Minimal memory overhead ($O(N + M)$), constant-time degree lookup ($d_i = \text{row\_ptr}[i+1] - \text{row\_ptr}[i]$), and direct compatibility with CUDA warp-synchronous memory coalescing.

---

## ADR-003: Two-Tier GPU Execution Strategy (Local Detection + Colab Workflow)

- **Status:** Accepted
- **Context:** The local host environment features an NVIDIA GeForce MX450 GPU, but lacks the local NVIDIA CUDA Toolkit (`nvcc`) and MSVC build tools (`cl.exe`). Attempting to silently download multi-gigabyte external compilers without user approval violates the Master Directive.
- **Decision:** Implement a dual-tier execution architecture:
  1. Detect and report local GPU hardware telemetry transparently.
  2. Author and version-control all native `.cu` CUDA kernels in `backend/gpu_engine/cuda/src/`.
  3. Implement an automated packaging utility (`backend/gpu_engine/colab_runner/`) to bundle CSR graph inputs and CUDA sources for execution on Google Colab GPU runtimes.
  4. Ingest verified benchmark logs and tensors back into `experiments/results/`.
  5. Strictly prohibit any fabricated GPU metrics; display `AWAITING_GPU_RUN` for unexecuted configurations.
- **Consequences:** Transparent, 100% reproducible GPU benchmarks without violating system integrity or inventing fake performance numbers.

---

## ADR-004: Adoption of Obsidian Cybernetics Design System

- **Status:** Accepted
- **Context:** The application requires a cohesive, research-grade fintech forensic aesthetic. Stitch MCP export `stitch_fraudgraph_intelligence_platform` provides the authoritative design specification (`Obsidian Cybernetics`).
- **Decision:** Implement the strict zero-radius (`0px`), dark obsidian minimal visual language (`#05070A`, `#0D1117`, `#161B22`, `#00F2FE`, `#0AE8B7`, `#FF3B30`) pairing Newsreader serif typography for monograph headings and Geist sans-serif for dense analytical telemetry.
- **Consequences:** High-fidelity visual consistency directly aligning with the approved Stitch source of truth.
