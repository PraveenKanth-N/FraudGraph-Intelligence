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

### 2.4 CUDA Correctness & Numerical Equivalence (Phase 5 & 6 Gates)
- Floating-point tolerance check: $|Y_{\text{GPU}} - Y_{\text{CPU}}| \le 10^{-5}$.
- Edge cases:
  - Empty graph ($N = 0, M = 0$).
  - Fully disconnected graph (all degree 0).
  - Star graph (1 hub node connected to $N-1$ leaves).
  - Large random power-law graph ($N = 100,000$).

### 2.5 Frontend Unit & Type Tests (Phase 9 Gate)
- TypeScript strict compilation: `tsc -b`.
- Production bundle verification: `vite build`.
- Zero-radius layout adherence check.
