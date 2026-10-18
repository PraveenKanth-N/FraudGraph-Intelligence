# Quality Assurance & Test Plan — FraudGraph Intelligence (FGI)

## 1. Testing Philosophy & Multi-Layer Coverage

The testing framework guarantees correctness, numerical precision, and stability across every architectural layer:

```
+-------------------------------------------------------------------------------+
|                               TESTING PYRAMID                                 |
+-------------------------------------------------------------------------------+
|   Layer 4: End-to-End System Tests (FastAPI Client to UI, Benchmark Runs)     |
+-------------------------------------------------------------------------------+
|   Layer 3: Numerical Equivalence Tests (CPU vs CUDA Outputs, atol <= 1e-5)    |
+-------------------------------------------------------------------------------+
|   Layer 2: API Contract & Graph Integration Tests (FastAPI + CSR Builder)     |
+-------------------------------------------------------------------------------+
|   Layer 1: Unit Tests (Health Endpoint, CSR Array Shapes, Edge Boundary Cases)|
+-------------------------------------------------------------------------------+
```

---

## 2. Test Suites by Module

### 2.1 Backend Health & Diagnostics (`backend/tests/test_health.py`)
- Verified `/` root response and `/api/v1/health` JSON payload.
- Confirmed reporting of local GPU detection (`NVIDIA GeForce MX450`, compute capability 7.5) and local CUDA compiler status (`UNAVAILABLE_LOCAL_COLAB_READY`).

### 2.2 CSR Graph Representation Tests (`backend/tests/test_csr_graph.py`) — Phase 3 (20 Tests Passed)
- Validates monotonicity of `row_ptr`: $\text{row\_ptr}[i+1] \ge \text{row\_ptr}[i]$.
- Validates start offset $\text{row\_ptr}[0] == 0$ and termination $\text{row\_ptr}[N] == M$.
- Validates index boundaries: $0 \le \text{col\_idx}[k] < N$.
- Validates isolated node handling: nodes with degree 0 supported without exception.
- Validates duplicate edge policies and self-loop retention/pruning.
- Validates directed vs. undirected symmetry rules.
- Validates `.npz` round-trip serialization and manifest restoration.
- Validates temporal transaction builder causality ($t_u \le t_v$), sentinel rejection, and label quarantine.

### 2.3 CPU Reference Engine Tests (`backend/tests/test_aggregators.py`) — Phase 4 (22 Tests Passed)
- Hand-calculated verification of Sum, Mean, and Max aggregators on 5-node toy fixture.
- Strict empty-neighborhood zero-vector return ($d_i = 0 \implies \vec{0}$) across all operators.
- Entirely empty graph ($N=0$) and single-node graphs (with and without self-loops).
- Directed asymmetric vs. undirected symmetric adjacency behavior.
- High-degree hub node stress testing (500-leaf star graph).
- Multiple feature dimensions: $D \in \{1, 4, 16, 32, 128\}$.
- Negative and zero-valued feature handling (confirming max does not erroneously return zero for negative inputs).
- Deterministic output across repeated calls and input immutability.
- Strict defensive input validation (`TypeError` and `ValueError`).
- Cross-verification between reference loop and vectorized SpMM ($\text{atol} \le 10^{-5}$).

### 2.4 CUDA Correctness & Colab Workflow Tests (`backend/tests/test_cuda_baseline.py`) — Phase 5 (11 Passed, 1 Skipped)
- Verifies existence and completeness of native CUDA source and header files (`fgi_cuda_common.cuh`, `fgi_baseline_kernels.cuh`, `fgi_baseline_kernels.cu`, `main_baseline.cu`, `Makefile`).
- Validates automated packaging of self-contained Colab archive (`colab_bundle_fgi_phase5.zip`, 3.55 MB).
- Validates Colab workflow notebook syntax and required execution commands (`fgi_colab_workflow.ipynb`).
- Enforces strict JSON schema validation and rejects unmeasured (`SAMPLE`/`NOT_RUN`) or incomplete benchmark outputs.
- Validates numerical comparison logic against Phase 4 CPU reference with $\text{atol} \le 1.0 \times 10^{-5}$.
- Defensive error testing for shape mismatches, NaNs, Infs, and tolerance violations.
- Transparent offline error reporting when local compiler (`nvcc`) is absent; live GPU tests explicitly marked as `SKIPPED`.

### 2.5 Frontend Unit & Type Tests (Phase 9 Gate)
- TypeScript strict compilation: `tsc -b`.
- Production bundle verification: `vite build`.
- Zero-radius layout adherence check.
