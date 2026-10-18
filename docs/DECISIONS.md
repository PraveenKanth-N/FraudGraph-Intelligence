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

---

## ADR-005: Dual-Dataset Strategy for Verification and Downstream Evaluation

- **Status:** Accepted (Phase 2)
- **Context:** Evaluating GPU acceleration requires precisely controlled graph scales ($N, M$), degree distributions, and feature dimensions ($D$) to validate micro-architectural hypotheses without dependency on multi-gigabyte raw downloads. Simultaneously, demonstrating downstream fraud utility requires real tabular transactions with actual fraud imbalance.
- **Decision:** Adopt a dual-dataset strategy:
  1. *Controlled Synthetic Fixtures:* Deterministic Barabási–Albert and R-MAT graph generators used for CI/CD correctness tests, scale stress testing, and micro-architectural kernel comparisons.
  2. *Empirical Financial Benchmark:* Managed, leak-free temporal subset of IEEE-CIS Fraud Detection (or Elliptic benchmark) for evaluating downstream classification gains (PR-AUC, F1-score).
- **Consequences:** Zero external network dependency for kernel verification tests; high statistical defensibility for fraud-scoring evaluations.

---

## ADR-006: Mathematical Aggregation Operator Scope & Zero-Degree Sentinel

- **Status:** Accepted (Phase 2)
- **Context:** Graph aggregations must handle isolated nodes ($d_i = 0$) deterministically without producing NaNs, division-by-zero, or $-\infty$ values that break downstream neural networks.
- **Decision:** 
  1. Designate **Mean Aggregation** ($\frac{1}{d_i} \sum X_j$) as the primary benchmark operator, guarded by an explicit zero check ($d_i = 0 \implies \vec{0}$).
  2. Support **Sum Aggregation** ($\sum X_j$) as a linear additive baseline.
  3. Specify **Max Aggregation** with a strict zero-sentinel policy for empty neighborhoods ($\max(\emptyset) \triangleq \vec{0}$) rather than $-\infty$.
- **Consequences:** Complete numerical stability and strict compatibility with single-precision floating-point tolerances ($atol \le 10^{-5}$).

---

## ADR-007: Decoupled GPU Benchmarking & Explicit Measurement States

- **Status:** Accepted (Phase 2)
- **Context:** Reporting GPU speedups without isolating PCIe transfer times from pure kernel runtimes leads to invalid research conclusions. Additionally, placeholder or mock values must never masquerade as verified measurements.
- **Decision:**
  1. Measure kernel execution time exclusively using hardware CUDA events (`cudaEventRecord`).
  2. Measure Host-to-Device and Device-to-Host transfer latencies as separate metrics.
  3. Enforce the tri-state measurement standard: every reported datapoint must be explicitly marked as `MEASURED`, `SAMPLE`, or `NOT_RUN`.
- **Consequences:** Unimpeachable research integrity, reproducible benchmarks, and transparent performance accounting.
