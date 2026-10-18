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

### 2.2 CSR Graph Representation Tests (Phase 3 Gate)
- Validates monotonicity of `row_ptr`: $\text{row\_ptr}[i+1] \ge \text{row\_ptr}[i]$.
- Validates index boundaries: $0 \le \text{col\_idx}[k] < N$.
- Validates isolated node handling: nodes with degree 0 produce a zero row in the aggregated matrix without dividing by zero.

### 2.3 CPU Reference Engine Tests (Phase 4 Gate)
- Verifies deterministic output of Mean, Sum, and Max aggregation across small synthetic toy graphs.
- Checks edge-weight multiplication accuracy.

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
