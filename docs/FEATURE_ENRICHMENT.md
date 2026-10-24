# Graph Feature Enrichment Pipeline Specification

**Module:** FGI-MOD-07 (Fraud Scoring & Classifier Baseline)
**Academic Course:** VIT Vellore — CSI4007 GPU Programming
**Authoritative Source:** [`backend/ml_engine/feature_enrichment.py`](file:///D:/FraudGraph%20Intelligence/backend/ml_engine/feature_enrichment.py)
**Status:** Phase 7.3 Complete & Empirically Verified

---

## 1. Overview & Research Scope

The **Feature Enrichment Pipeline** bridges graph aggregation engines (CPU reference and CUDA accelerated kernels) with downstream fraud classification models.

It constructs concatenated multidimensional feature tensors by binding raw transaction features $X \in \mathbb{R}^{N \times D}$ with relational neighborhood embeddings $Y \in \mathbb{R}^{N \times D}$ computed over causal transaction graphs.

> [!IMPORTANT]
> **Research Boundary & Non-Presumption of Improvement:**
> In accordance with Phase 7 research guidelines, this module provides the *infrastructure* for feature enrichment. **No claim is made that graph-enriched representations improve fraud classification accuracy** until Phase 7.4 empirical evaluation rigorously establishes test ROC-AUC and PR-AUC deltas ($\Delta \text{PR-AUC}$) across baseline models.

---

## 2. Supported Feature Representations

All representations enforce strict column layout, float32 precision, and deterministic ordering:

| Representation Name | Symbol / Layout | Dimension ($D=32$) | Description & Concatenation Order |
| :--- | :--- | :--- | :--- |
| **`raw`** | $X$ | $32$ ($D$) | Base transaction features without topological context. |
| **`mean`** | $[X \,\|\, Y_{\text{mean}}]$ | $64$ ($2D$) | Raw features $[0:D]$ concatenated with neighbor mean aggregation $[D:2D]$. |
| **`max`** | $[X \,\|\, Y_{\text{max}}]$ | $64$ ($2D$) | Raw features $[0:D]$ concatenated with neighbor max aggregation $[D:2D]$. |
| **`sum`** | $[X \,\|\, Y_{\text{sum}}]$ | $64$ ($2D$) | Raw features $[0:D]$ concatenated with neighbor sum aggregation $[D:2D]$. |
| **`multi_operator`**<br>(alias: **`combined`**) | $[X \,\|\, Y_{\text{mean}} \,\|\, Y_{\text{max}}]$ | $96$ ($3D$) | Multi-operator fusion: raw features $[0:D]$, mean embeddings $[D:2D]$, and max embeddings $[2D:3D]$. |

### Mathematical Column Ordering Contract:
For any enriched matrix $E \in \mathbb{R}^{N \times kD}$:
$$E_{i, :} = \begin{cases}
X_{i, :} & \text{if rep} = \text{RAW} \\
\big[X_{i, :} \;\Vert\; Y_{\text{mean}, i, :}\big] & \text{if rep} = \text{MEAN} \\
\big[X_{i, :} \;\Vert\; Y_{\text{max}, i, :}\big] & \text{if rep} = \text{MAX} \\
\big[X_{i, :} \;\Vert\; Y_{\text{sum}, i, :}\big] & \text{if rep} = \text{SUM} \\
\big[X_{i, :} \;\Vert\; Y_{\text{mean}, i, :} \;\Vert\; Y_{\text{max}, i, :}\big] & \text{if rep} \in \{\text{MULTI\_OPERATOR}, \text{COMBINED}\}
\end{cases}$$

---

## 3. Graph Neighborhood Context Policies

To enable rigorous evaluation across both online streaming and strict inductive validation regimes, two explicit context policies are implemented:

```
1. CAUSAL_HISTORY Policy (Online Streaming):
   Node i aggregates from all historical neighbors: {j : j < i, t_j < t_i, (t_i - t_j) <= 7d}
   [Train Nodes] -------------> aggregates from historical Train
   [Val Nodes]   -------------> aggregates from historical Train + past Val
   [Test Nodes]  -------------> aggregates from historical Train + Val + past Test

2. TRAIN_ONLY_CONTEXT Policy (Strict Inductive Historical Feature Store):
   Node i aggregates ONLY from historical training nodes: {j : j < train_end, t_j < t_i}
   [Train Nodes] -------------> aggregates from historical Train
   [Val Nodes]   -------------> aggregates strictly from historical Train (Val-to-Val edges pruned)
   [Test Nodes]  -------------> aggregates strictly from historical Train (Test lookback pruned)
```

### 3.1 Policy 1: `causal_history` (Default)
- **Semantics:** Each node aggregates from all historically available preceding neighbors within the sliding lookback window ($t_j < t_i$, $j < i$).
- **Streaming Parity:** Models a production fraud screening engine where incoming transactions query recent activity over the previous 7 days (including unconfirmed recent transactions).
- **Training Isolation:** Because $100\%$ of CSR edges satisfy $j < i$, training nodes $i < \text{train\_end}$ have **zero edges** into validation or test partitions. Training node representations are strictly isolated.

### 3.2 Policy 2: `train_only_context`
- **Semantics:** Restricts neighbor lookback exclusively to historical *training* transactions ($j < \text{train\_end}$).
- **Implementation:** Vectorized CSR pruning masks all edges where `col_idx >= train_end`.
- **Properties:**
  - Training nodes ($i < \text{train\_end}$) remain bit-for-bit identical to `causal_history`.
  - Validation and test nodes discard context from unconfirmed recent transactions, aggregating strictly from confirmed historical training entities.
  - Required if label propagation or target encoding is ever introduced.

---

## 4. Execution Backends & Numerical Invariants

### 4.1 Supported Aggregation Backends
1. **`cpu_vectorized` (Default):** High-performance SciPy sparse matrix multiplication (SpMM) for sum/mean and block-vectorized reductions for max.
2. **`cpu_reference`:** Python loop reference implementation used for golden-truth numerical verification.
3. **`cuda_baseline`:** Native CUDA baseline kernel (1 thread per node).
4. **`cuda_coalesced`:** Native warp-coalesced CUDA kernel (32 threads per node, Phase 6.1).
5. **`cuda_streams`:** Pinned memory asynchronous stream CUDA kernel (Phase 6.2).
6. **`cuda_degree_aware`:** Topological hybrid dispatch CUDA kernel (Phase 6.3).

### 4.2 Numerical Contracts & Sentinels
- **Dtype:** Strictly `np.float32`.
- **Finiteness:** 100% finite guarantee. Any NaN or Inf in inputs or outputs triggers immediate validation failure.
- **Zero-Degree Sentinel:** For isolated nodes ($d_i = 0$), aggregated feature columns are guaranteed exact zero vectors ($\vec{0}$), preserving the FGI zero-sentinel contract:
  $$d_i = 0 \implies Y_{i, :} = \vec{0} \implies E_{i, :} = [X_{i, :} \;\|\; \vec{0}]$$
- **CPU/GPU Parity:** Where native CUDA runners are executed, outputs must match CPU reference within $\text{atol} \le 1.0 \times 10^{-5}$.

---

## 5. Python API Reference

```python
from backend.ml_engine.feature_enrichment import (
    enrich_features,
    enrich_features_with_split,
    apply_context_policy,
    RepresentationType,
    AggregationBackend,
    ContextPolicy,
    get_feature_dimension,
)

# 1. Enrich full graph with combined multi-operator features:
X_enriched = enrich_features(
    graph=graph,
    representation=RepresentationType.COMBINED,  # or "raw", "mean", "max"
    backend=AggregationBackend.CPU_VECTORIZED,
    context_policy=ContextPolicy.CAUSAL_HISTORY,
)  # Shape: (10000, 96), dtype: float32

# 2. Enrich chronological dataset splits under train-only context:
X_train, X_val, X_test = enrich_features_with_split(
    graph=graph,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    representation=RepresentationType.MEAN,
    context_policy=ContextPolicy.TRAIN_ONLY_CONTEXT,
)  # Shapes: (7000, 64), (1500, 64), (1500, 64)
```

---

## 6. Research Limitations

1. **Feature-Only Aggregation:** The enrichment pipeline aggregates only numerical transaction features $X$. Ground-truth labels $y$ are quarantined and never passed into aggregation kernels.
2. **First-Order (1-Hop) Context:** Current pipeline evaluates 1-hop relational neighborhoods. Multi-hop cascades (2-hop aggregations) are deferred to future extensions.
3. **Local GPU Availability:** Local execution falls back cleanly to verified CPU backends when MSVC/nvcc is absent; GPU validation is conducted via the turnkey Google Colab Tesla T4 benchmark bundle.
