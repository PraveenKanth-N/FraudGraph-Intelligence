"""
CPU Reference Aggregation Engine
Implements deterministic, mathematically sound neighbor aggregation for CSR graphs.

Supported operators:
- sum:  Y_{i, :} = sum_{j in N(i)} X_{j, :}
- mean: Y_{i, :} = (1 / |N(i)|) * sum_{j in N(i)} X_{j, :}  (0 if |N(i)| == 0)
- max:  Y_{i, :} = max_{j in N(i)} X_{j, :}                  (0 if |N(i)| == 0)

All operations guarantee float32 precision, input immutability, explicit zero-degree handling,
and compliance with the FGI Graph Data Contract.
"""

from enum import Enum
from typing import Literal, Union
import numpy as np
from scipy import sparse

from backend.cpu_engine.csr_graph import CSRGraph


class AggregationOp(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"


def _validate_inputs(graph: CSRGraph, op: Union[str, AggregationOp]) -> tuple[CSRGraph, AggregationOp]:
    """Validates the input graph and aggregation operator."""
    if not isinstance(graph, CSRGraph):
        raise TypeError(f"Expected graph to be an instance of CSRGraph, got {type(graph).__name__}")

    if not isinstance(graph.features, np.ndarray) or graph.features.ndim != 2:
        raise ValueError("Graph features must be a 2D NumPy array (num_nodes, feature_dim)")

    if not np.all(np.isfinite(graph.features)):
        raise ValueError("Graph features contain NaN or Infinite values, which violate CSR invariants")

    if isinstance(op, str):
        op_clean = op.strip().lower()
        try:
            op_enum = AggregationOp(op_clean)
        except ValueError:
            valid_ops = [o.value for o in AggregationOp]
            raise ValueError(f"Unknown aggregation operator '{op}'. Supported operators: {valid_ops}")
    elif isinstance(op, AggregationOp):
        op_enum = op
    else:
        raise TypeError(f"Expected op to be str or AggregationOp, got {type(op).__name__}")

    return graph, op_enum


def aggregate_sum_reference(graph: CSRGraph) -> np.ndarray:
    """
    Reference sum aggregation over neighbor features.
    
    Y_{i, k} = sum_{j in N(i)} X_{j, k}
    For empty neighborhoods (|N(i)| == 0), returns a zero vector.
    """
    num_nodes = graph.num_nodes
    feature_dim = graph.feature_dim
    features = graph.features
    row_ptr = graph.row_ptr
    col_idx = graph.col_idx

    output = np.zeros((num_nodes, feature_dim), dtype=np.float32)

    for i in range(num_nodes):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        if end > start:
            neighbors = col_idx[start:end]
            output[i] = np.sum(features[neighbors], axis=0, dtype=np.float32)

    return output


def aggregate_mean_reference(graph: CSRGraph) -> np.ndarray:
    """
    Reference mean aggregation over neighbor features.
    
    Y_{i, k} = (1 / |N(i)|) * sum_{j in N(i)} X_{j, k} if |N(i)| > 0 else 0
    Guaranteed zero vector for isolated nodes.
    """
    num_nodes = graph.num_nodes
    feature_dim = graph.feature_dim
    features = graph.features
    row_ptr = graph.row_ptr
    col_idx = graph.col_idx

    output = np.zeros((num_nodes, feature_dim), dtype=np.float32)

    for i in range(num_nodes):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        deg = end - start
        if deg > 0:
            neighbors = col_idx[start:end]
            # Accumulate in float32 and divide by degree
            neighbor_sum = np.sum(features[neighbors], axis=0, dtype=np.float32)
            output[i] = neighbor_sum / np.float32(deg)

    return output


def aggregate_max_reference(graph: CSRGraph) -> np.ndarray:
    """
    Reference max aggregation over neighbor features.
    
    Y_{i, k} = max_{j in N(i)} X_{j, k} if |N(i)| > 0 else 0
    Guaranteed zero vector for isolated nodes (explicitly guarded).
    """
    num_nodes = graph.num_nodes
    feature_dim = graph.feature_dim
    features = graph.features
    row_ptr = graph.row_ptr
    col_idx = graph.col_idx

    output = np.zeros((num_nodes, feature_dim), dtype=np.float32)

    for i in range(num_nodes):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        if end > start:
            neighbors = col_idx[start:end]
            output[i] = np.max(features[neighbors], axis=0)

    return output


def aggregate_vectorized(graph: CSRGraph, op: Union[str, AggregationOp] = AggregationOp.MEAN) -> np.ndarray:
    """
    Vectorized aggregation using SciPy sparse matrix multiplication (SpMM) where applicable.
    
    For 'sum':  Y = A * X
    For 'mean': Y = D^{-1} * A * X
    For 'max':  Iterative block-vectorized reduction
    """
    graph, op_enum = _validate_inputs(graph, op)
    num_nodes = graph.num_nodes
    feature_dim = graph.feature_dim

    if num_nodes == 0 or feature_dim == 0:
        return np.zeros((num_nodes, feature_dim), dtype=np.float32)

    row_ptr = graph.row_ptr
    col_idx = graph.col_idx
    num_edges = graph.num_edges
    features = graph.features

    if num_edges == 0:
        return np.zeros((num_nodes, feature_dim), dtype=np.float32)

    if op_enum == AggregationOp.SUM:
        data = np.ones(num_edges, dtype=np.float32)
        adj_matrix = sparse.csr_matrix((data, col_idx, row_ptr), shape=(num_nodes, num_nodes))
        # SpMM product: (N, N) * (N, D) -> (N, D)
        return (adj_matrix @ features).astype(np.float32, copy=False)

    elif op_enum == AggregationOp.MEAN:
        # Calculate degree for each row: d_i = row_ptr[i+1] - row_ptr[i]
        degrees = row_ptr[1:] - row_ptr[:-1]
        inv_degrees = np.zeros(num_nodes, dtype=np.float32)
        nonzero_mask = degrees > 0
        inv_degrees[nonzero_mask] = 1.0 / degrees[nonzero_mask].astype(np.float32)

        # Expand inv_degrees to each edge in col_idx
        edge_degrees = np.repeat(inv_degrees, degrees)
        norm_adj = sparse.csr_matrix((edge_degrees, col_idx, row_ptr), shape=(num_nodes, num_nodes))
        return (norm_adj @ features).astype(np.float32, copy=False)

    elif op_enum == AggregationOp.MAX:
        # SciPy SpMM does not support semiring max-plus directly; use optimized reference
        return aggregate_max_reference(graph)

    raise ValueError(f"Unsupported operator: {op_enum}")


def aggregate_neighbors(
    graph: CSRGraph,
    op: Union[str, AggregationOp] = AggregationOp.MEAN,
    method: Literal["reference", "vectorized"] = "reference",
) -> np.ndarray:
    """
    Public entry point for CPU graph neighbor aggregation.
    
    Parameters:
        graph: Validated CSRGraph instance.
        op: Aggregation operator ("sum", "mean", "max" or AggregationOp).
        method: "reference" (straightforward loop) or "vectorized" (SciPy SpMM).
        
    Returns:
        np.ndarray: Aggregated feature matrix of shape (num_nodes, feature_dim) with dtype float32.
    """
    graph, op_enum = _validate_inputs(graph, op)

    if method == "vectorized":
        return aggregate_vectorized(graph, op=op_enum)
    elif method == "reference":
        if op_enum == AggregationOp.SUM:
            return aggregate_sum_reference(graph)
        elif op_enum == AggregationOp.MEAN:
            return aggregate_mean_reference(graph)
        elif op_enum == AggregationOp.MAX:
            return aggregate_max_reference(graph)
        else:
            raise ValueError(f"Unsupported operator: {op_enum}")
    else:
        raise ValueError(f"Unknown aggregation method '{method}'. Supported: 'reference', 'vectorized'")
