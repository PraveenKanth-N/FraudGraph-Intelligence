"""
Graph Feature Enrichment Pipeline
Module: FGI-MOD-07 (Fraud Scoring & Classifier Baseline)
Course: VIT Vellore — CSI4007 GPU Programming

Constructs enriched feature representations by combining raw transaction features with
relational neighborhood embeddings generated via CPU reference aggregators and native
CUDA acceleration kernels.

Supported Representations:
- raw: X, shape (N, D)
- mean: [X || Y_mean], shape (N, 2D)
- max: [X || Y_max], shape (N, 2D)
- sum: [X || Y_sum], shape (N, 2D)
- multi_operator (alias combined): [X || Y_mean || Y_max], shape (N, 3D)

Context Policies:
- causal_history: each node aggregates from all historically prior neighbors (t_j < t_i, j < i).
- train_only_context: strictly restricts neighbor aggregation to historical training transactions
  (j < train_end), pruning edges pointing to validation or test transactions.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import sys
from typing import Any, Literal, Sequence, Union

# Ensure project root is in sys.path
_root_dir = str(Path(__file__).resolve().parents[2])
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import numpy as np

from backend.cpu_engine.aggregators import AggregationOp, aggregate_neighbors
from backend.cpu_engine.csr_graph import CSRGraph
from backend.gpu_engine.cuda_runner import execute_cuda_aggregation, is_nvcc_available, get_cuda_binary_path


class RepresentationType(str, Enum):
    """Supported enriched feature representation formats."""
    RAW = "raw"
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    MULTI_OPERATOR = "multi_operator"
    COMBINED = "combined"  # Canonical alias for multi_operator [X || Y_mean || Y_max]


class AggregationBackend(str, Enum):
    """Supported execution engines for neighbor feature aggregation."""
    CPU_REFERENCE = "cpu_reference"
    CPU_VECTORIZED = "cpu_vectorized"
    CUDA_BASELINE = "cuda_baseline"
    CUDA_COALESCED = "cuda_coalesced"
    CUDA_STREAMS = "cuda_streams"
    CUDA_DEGREE_AWARE = "cuda_degree_aware"


class ContextPolicy(str, Enum):
    """Graph neighborhood context policy governing temporal train/test isolation."""
    CAUSAL_HISTORY = "causal_history"
    TRAIN_ONLY_CONTEXT = "train_only_context"


def _parse_representation(rep: str | RepresentationType) -> RepresentationType:
    """Normalize and validate representation type."""
    if isinstance(rep, RepresentationType):
        return rep
    if isinstance(rep, str):
        cleaned = rep.strip().lower()
        for member in RepresentationType:
            if member.value == cleaned:
                return member
        valid_reps = [m.value for m in RepresentationType]
        raise ValueError(f"Unknown representation '{rep}'. Supported representations: {valid_reps}")
    raise TypeError(f"Expected representation to be str or RepresentationType, got {type(rep).__name__}")


def _parse_backend(backend: str | AggregationBackend) -> AggregationBackend:
    """Normalize and validate aggregation backend."""
    if isinstance(backend, AggregationBackend):
        return backend
    if isinstance(backend, str):
        cleaned = backend.strip().lower()
        for member in AggregationBackend:
            if member.value == cleaned:
                return member
        valid_backends = [m.value for m in AggregationBackend]
        raise ValueError(f"Unknown aggregation backend '{backend}'. Supported backends: {valid_backends}")
    raise TypeError(f"Expected backend to be str or AggregationBackend, got {type(backend).__name__}")


def _parse_context_policy(policy: str | ContextPolicy) -> ContextPolicy:
    """Normalize and validate graph context policy."""
    if isinstance(policy, ContextPolicy):
        return policy
    if isinstance(policy, str):
        cleaned = policy.strip().lower()
        for member in ContextPolicy:
            if member.value == cleaned:
                return member
        valid_policies = [m.value for m in ContextPolicy]
        raise ValueError(f"Unknown context policy '{policy}'. Supported policies: {valid_policies}")
    raise TypeError(f"Expected context_policy to be str or ContextPolicy, got {type(policy).__name__}")


def get_feature_dimension(base_dim: int, representation: str | RepresentationType) -> int:
    """
    Compute output feature dimension for a given base node dimension and representation.

    Args:
        base_dim: Dimensionality of raw node features (D).
        representation: Selected representation type.

    Returns:
        Total output feature dimension (D, 2D, or 3D).
    """
    if base_dim < 0:
        raise ValueError(f"base_dim must be non-negative, got {base_dim}")
    rep_enum = _parse_representation(representation)
    if rep_enum == RepresentationType.RAW:
        return base_dim
    elif rep_enum in (RepresentationType.MEAN, RepresentationType.MAX, RepresentationType.SUM):
        return base_dim * 2
    elif rep_enum in (RepresentationType.MULTI_OPERATOR, RepresentationType.COMBINED):
        return base_dim * 3
    raise ValueError(f"Unhandled representation: {rep_enum}")


def apply_context_policy(
    graph: CSRGraph,
    policy: str | ContextPolicy = ContextPolicy.CAUSAL_HISTORY,
    train_end: int | None = None,
) -> CSRGraph:
    """
    Apply neighborhood context policy to a CSRGraph.

    Policies:
    - CAUSAL_HISTORY: Preserves the existing graph unchanged, where each node aggregates
      from all historically preceding neighbors (t_j < t_i, j < i).
    - TRAIN_ONLY_CONTEXT: Strictly limits neighbor aggregation to historical training transactions
      (j < train_end). For validation/test nodes (i >= train_end), any edges pointing to
      other validation or test transactions are pruned.

    Args:
        graph: Input CSRGraph instance.
        policy: Context policy to enforce.
        train_end: Index cutoff separating training nodes [0, train_end) from validation/test.
                   Required when policy is TRAIN_ONLY_CONTEXT.

    Returns:
        CSRGraph meeting all invariants with the specified context policy enforced.
    """
    if not isinstance(graph, CSRGraph):
        raise TypeError(f"Expected graph to be an instance of CSRGraph, got {type(graph).__name__}")

    policy_enum = _parse_context_policy(policy)

    if policy_enum == ContextPolicy.CAUSAL_HISTORY:
        return graph

    if policy_enum == ContextPolicy.TRAIN_ONLY_CONTEXT:
        if train_end is None:
            raise ValueError("train_end must be explicitly provided when using TRAIN_ONLY_CONTEXT policy")
        if not (0 <= train_end <= graph.num_nodes):
            raise ValueError(f"train_end must be in [0, {graph.num_nodes}], got {train_end}")

        if graph.num_nodes == 0 or graph.num_edges == 0:
            return graph

        # Vectorized edge pruning: keep only edges where target j < train_end
        row_ptr = graph.row_ptr
        col_idx = graph.col_idx

        degrees = np.diff(row_ptr)
        row_indices = np.repeat(np.arange(graph.num_nodes, dtype=np.int32), degrees)
        keep_mask = col_idx < train_end

        filtered_col_idx = col_idx[keep_mask]
        filtered_row_indices = row_indices[keep_mask]

        filtered_degrees = np.bincount(filtered_row_indices, minlength=graph.num_nodes).astype(np.int32)
        new_row_ptr = np.zeros(graph.num_nodes + 1, dtype=np.int32)
        np.cumsum(filtered_degrees, out=new_row_ptr[1:])

        meta = dict(graph.metadata)
        meta["context_policy"] = "train_only_context"
        meta["train_end"] = train_end
        meta["pruned_edges"] = int(graph.num_edges - len(filtered_col_idx))

        return CSRGraph(
            row_ptr=new_row_ptr,
            col_idx=filtered_col_idx,
            features=graph.features,
            is_directed=graph.is_directed,
            node_id_to_idx=graph.node_id_to_idx,
            idx_to_node_id=graph.idx_to_node_id,
            metadata=meta,
            validate=True,
        )

    raise ValueError(f"Unhandled context policy: {policy_enum}")


def compute_neighbor_aggregation(
    graph: CSRGraph,
    operator: str | AggregationOp,
    backend: str | AggregationBackend = AggregationBackend.CPU_VECTORIZED,
) -> np.ndarray:
    """
    Compute neighborhood feature aggregation over a CSRGraph using the specified backend.

    Args:
        graph: Validated CSRGraph instance.
        operator: Aggregation operator ("mean", "max", or "sum").
        backend: Execution engine to use.

    Returns:
        np.ndarray of shape (num_nodes, feature_dim) with dtype float32.
    """
    if not isinstance(graph, CSRGraph):
        raise TypeError(f"Expected graph to be an instance of CSRGraph, got {type(graph).__name__}")

    backend_enum = _parse_backend(backend)

    # Normalize operator string
    if isinstance(operator, AggregationOp):
        op_str = operator.value
    elif isinstance(operator, str):
        op_str = operator.strip().lower()
        if op_str not in ("mean", "max", "sum"):
            raise ValueError(f"Unsupported operator '{operator}'. Supported: ['mean', 'max', 'sum']")
    else:
        raise TypeError(f"Expected operator to be str or AggregationOp, got {type(operator).__name__}")

    if graph.num_nodes == 0 or graph.feature_dim == 0:
        return np.zeros((graph.num_nodes, graph.feature_dim), dtype=np.float32)

    # Dispatch to CPU or GPU runner
    if backend_enum == AggregationBackend.CPU_REFERENCE:
        return aggregate_neighbors(graph, op=op_str, method="reference")

    elif backend_enum == AggregationBackend.CPU_VECTORIZED:
        return aggregate_neighbors(graph, op=op_str, method="vectorized")

    elif backend_enum == AggregationBackend.CUDA_BASELINE:
        output_arr, _ = execute_cuda_aggregation(graph, operator=op_str, kernel="baseline")
        return output_arr

    elif backend_enum == AggregationBackend.CUDA_COALESCED:
        output_arr, _ = execute_cuda_aggregation(graph, operator=op_str, kernel="coalesced")
        return output_arr

    elif backend_enum == AggregationBackend.CUDA_STREAMS:
        output_arr, _ = execute_cuda_aggregation(graph, operator=op_str, kernel="streams")
        return output_arr

    elif backend_enum == AggregationBackend.CUDA_DEGREE_AWARE:
        output_arr, _ = execute_cuda_aggregation(graph, operator=op_str, kernel="degree_aware")
        return output_arr

    raise ValueError(f"Unhandled backend: {backend_enum}")


def enrich_features(
    graph: CSRGraph,
    representation: str | RepresentationType = RepresentationType.MEAN,
    backend: str | AggregationBackend = AggregationBackend.CPU_VECTORIZED,
    context_policy: str | ContextPolicy = ContextPolicy.CAUSAL_HISTORY,
    train_end: int | None = None,
) -> np.ndarray:
    """
    Construct enriched feature matrix for all nodes in graph.

    Representations:
    - RAW: X, shape (N, D)
    - MEAN: [X || Y_mean], shape (N, 2D)
    - MAX: [X || Y_max], shape (N, 2D)
    - SUM: [X || Y_sum], shape (N, 2D)
    - MULTI_OPERATOR / COMBINED: [X || Y_mean || Y_max], shape (N, 3D)

    Concatenation Order Contract:
    - Always starts with raw features X at columns [0:D].
    - For MEAN / MAX / SUM: aggregated features at columns [D:2D].
    - For MULTI_OPERATOR / COMBINED: Y_mean at [D:2D], Y_max at [2D:3D].

    Zero-Degree Sentinel Contract:
    - Nodes with d_i = 0 produce exact zeros for aggregated columns.

    Args:
        graph: Validated CSRGraph instance containing base node features.
        representation: Output representation format.
        backend: Execution engine for neighbor aggregation.
        context_policy: Temporal isolation policy ("causal_history" or "train_only_context").
        train_end: Index cutoff for TRAIN_ONLY_CONTEXT policy.

    Returns:
        np.ndarray of shape (N, D_out) with dtype float32, guaranteed 100% finite.
    """
    if not isinstance(graph, CSRGraph):
        raise TypeError(f"Expected graph to be an instance of CSRGraph, got {type(graph).__name__}")

    rep_enum = _parse_representation(representation)
    backend_enum = _parse_backend(backend)
    policy_enum = _parse_context_policy(context_policy)

    num_nodes = graph.num_nodes
    feature_dim = graph.feature_dim
    raw_features = graph.features

    # Boundary handling for empty inputs
    if num_nodes == 0 or feature_dim == 0:
        expected_dim = get_feature_dimension(feature_dim, rep_enum)
        return np.zeros((num_nodes, expected_dim), dtype=np.float32)

    # 1. Raw representation: return copy of raw features
    if rep_enum == RepresentationType.RAW:
        return raw_features.copy().astype(np.float32, copy=False)

    # Apply context policy to graph if needed
    context_graph = apply_context_policy(graph, policy=policy_enum, train_end=train_end)

    # 2. Single-operator representations
    if rep_enum == RepresentationType.MEAN:
        y_mean = compute_neighbor_aggregation(context_graph, operator="mean", backend=backend_enum)
        enriched = np.hstack([raw_features, y_mean]).astype(np.float32, copy=False)

    elif rep_enum == RepresentationType.MAX:
        y_max = compute_neighbor_aggregation(context_graph, operator="max", backend=backend_enum)
        enriched = np.hstack([raw_features, y_max]).astype(np.float32, copy=False)

    elif rep_enum == RepresentationType.SUM:
        y_sum = compute_neighbor_aggregation(context_graph, operator="sum", backend=backend_enum)
        enriched = np.hstack([raw_features, y_sum]).astype(np.float32, copy=False)

    # 3. Multi-operator representation: [X || Y_mean || Y_max]
    elif rep_enum in (RepresentationType.MULTI_OPERATOR, RepresentationType.COMBINED):
        y_mean = compute_neighbor_aggregation(context_graph, operator="mean", backend=backend_enum)
        y_max = compute_neighbor_aggregation(context_graph, operator="max", backend=backend_enum)
        enriched = np.hstack([raw_features, y_mean, y_max]).astype(np.float32, copy=False)

    else:
        raise ValueError(f"Unhandled representation type: {rep_enum}")

    # Validate output invariants
    expected_dim = get_feature_dimension(feature_dim, rep_enum)
    if enriched.shape != (num_nodes, expected_dim):
        raise ValueError(
            f"Enriched feature shape mismatch: expected ({num_nodes}, {expected_dim}), got {enriched.shape}"
        )
    if enriched.dtype != np.float32:
        enriched = enriched.astype(np.float32, copy=False)
    if not np.all(np.isfinite(enriched)):
        raise ValueError("Enriched feature matrix contains non-finite values (NaN or Inf)")

    return enriched


def enrich_features_with_split(
    graph: CSRGraph,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    representation: str | RepresentationType = RepresentationType.MEAN,
    backend: str | AggregationBackend = AggregationBackend.CPU_VECTORIZED,
    context_policy: str | ContextPolicy = ContextPolicy.CAUSAL_HISTORY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enrich features for chronological dataset partitions (train, val, test).

    If context_policy is TRAIN_ONLY_CONTEXT, train_end is automatically set to len(train_idx).

    Args:
        graph: Validated CSRGraph instance.
        train_idx: 1D integer array of training node indices.
        val_idx: 1D integer array of validation node indices.
        test_idx: 1D integer array of test node indices.
        representation: Output representation type.
        backend: Execution engine for neighbor aggregation.
        context_policy: Graph context policy ("causal_history" or "train_only_context").

    Returns:
        Tuple of (X_train_enriched, X_val_enriched, X_test_enriched).
    """
    policy_enum = _parse_context_policy(context_policy)
    train_end = len(train_idx) if policy_enum == ContextPolicy.TRAIN_ONLY_CONTEXT else None

    # Compute full enriched feature matrix under the specified context policy
    full_enriched = enrich_features(
        graph=graph,
        representation=representation,
        backend=backend,
        context_policy=policy_enum,
        train_end=train_end,
    )

    x_train = full_enriched[train_idx]
    x_val = full_enriched[val_idx]
    x_test = full_enriched[test_idx]

    return x_train, x_val, x_test
