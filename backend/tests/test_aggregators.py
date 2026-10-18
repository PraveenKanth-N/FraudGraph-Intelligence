"""
Unit and Correctness Tests for CPU Reference Aggregation Engine
Covers all 18 verification requirements from Phase 4 specification:
1. Sum aggregation on hand-calculated graph
2. Mean aggregation on hand-calculated graph
3. Max aggregation on hand-calculated graph
4. Node with no neighbors (isolated node)
5. Entirely empty graph (N=0)
6. Single-node graph (with and without self-loop)
7. Directed graphs with asymmetric adjacency
8. Undirected graphs with symmetric adjacency
9. Graphs containing a hub with high degree
10. Duplicate edges and self-loops policies
11. Feature dimensions: 1, 4, 16, 32, 128
12. Negative and zero-valued features
13. Deterministic output across repeated calls
14. Input immutability
15. Invalid graph or aggregation configuration handling
16. Correct output shape and float32 dtype
17. Non-finite output / input detection
18. Cross-verification between reference and vectorized implementations
"""

import numpy as np
import pytest

from backend.cpu_engine.aggregators import (
    AggregationOp,
    aggregate_max_reference,
    aggregate_mean_reference,
    aggregate_neighbors,
    aggregate_sum_reference,
    aggregate_vectorized,
)
from backend.cpu_engine.csr_graph import CSRGraph
from backend.cpu_engine.synthetic_generator import generate_synthetic_graph


@pytest.fixture
def toy_5node_graph():
    """
    Known 5-node graph with 8 directed CSR entries (4 undirected pairs).
    Nodes: 0 (deg 2 -> 1, 2), 1 (deg 3 -> 0, 2, 3), 2 (deg 2 -> 0, 1), 3 (deg 1 -> 1), 4 (deg 0, isolated).
    """
    features = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],          # Node 0
            [10.0, 20.0, 30.0, 40.0],      # Node 1
            [5.0, 5.0, 5.0, 5.0],          # Node 2
            [-2.0, 0.0, 2.0, 4.0],         # Node 3
            [100.0, 200.0, 300.0, 400.0],  # Node 4 (isolated)
        ],
        dtype=np.float32,
    )
    edge_list = [(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (3, 1)]
    return CSRGraph.from_edge_list(num_nodes=5, edge_list=edge_list, features=features, feature_dim=4)


# 1. Sum aggregation on a hand-calculated graph
def test_hand_calculated_sum_aggregation(toy_5node_graph):
    out = aggregate_neighbors(toy_5node_graph, op="sum", method="reference")

    expected = np.array(
        [
            [15.0, 25.0, 35.0, 45.0],       # Node 0: neighbors 1 + 2
            [4.0, 7.0, 10.0, 13.0],         # Node 1: neighbors 0 + 2 + 3
            [11.0, 22.0, 33.0, 44.0],       # Node 2: neighbors 0 + 1
            [10.0, 20.0, 30.0, 40.0],       # Node 3: neighbor 1
            [0.0, 0.0, 0.0, 0.0],           # Node 4: isolated -> zero vector
        ],
        dtype=np.float32,
    )
    assert np.allclose(out, expected, atol=1e-6)
    assert out.dtype == np.float32


# 2. Mean aggregation on a hand-calculated graph
def test_hand_calculated_mean_aggregation(toy_5node_graph):
    out = aggregate_neighbors(toy_5node_graph, op="mean", method="reference")

    expected = np.array(
        [
            [7.5, 12.5, 17.5, 22.5],                        # Node 0: (15, 25, 35, 45) / 2
            [4.0 / 3.0, 7.0 / 3.0, 10.0 / 3.0, 13.0 / 3.0], # Node 1: (4, 7, 10, 13) / 3
            [5.5, 11.0, 16.5, 22.0],                        # Node 2: (11, 22, 33, 44) / 2
            [10.0, 20.0, 30.0, 40.0],                       # Node 3: 1 neighbor
            [0.0, 0.0, 0.0, 0.0],                           # Node 4: isolated -> 0
        ],
        dtype=np.float32,
    )
    assert np.allclose(out, expected, atol=1e-6)
    assert out.dtype == np.float32


# 3. Max aggregation on a hand-calculated graph
def test_hand_calculated_max_aggregation(toy_5node_graph):
    out = aggregate_neighbors(toy_5node_graph, op="max", method="reference")

    expected = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],       # Node 0: max(N1, N2)
            [5.0, 5.0, 5.0, 5.0],           # Node 1: max(N0, N2, N3)
            [10.0, 20.0, 30.0, 40.0],       # Node 2: max(N0, N1)
            [10.0, 20.0, 30.0, 40.0],       # Node 3: max(N1)
            [0.0, 0.0, 0.0, 0.0],           # Node 4: isolated -> zero vector
        ],
        dtype=np.float32,
    )
    assert np.allclose(out, expected, atol=1e-6)
    assert out.dtype == np.float32


# 4. A node with no neighbors
def test_node_with_no_neighbors_returns_zero_vector(toy_5node_graph):
    for op in ["sum", "mean", "max"]:
        out = aggregate_neighbors(toy_5node_graph, op=op)
        # Node 4 is isolated
        assert np.array_equal(out[4], np.zeros(4, dtype=np.float32))
        # Isolated node 4's feature [100, 200, 300, 400] must not appear in any other node's output
        assert not np.any(out == 100.0)


# 5. An entirely empty graph
def test_entirely_empty_graph():
    empty_graph = CSRGraph(
        row_ptr=np.array([0], dtype=np.int32),
        col_idx=np.array([], dtype=np.int32),
        features=np.zeros((0, 8), dtype=np.float32),
        is_directed=True,
    )
    for op in ["sum", "mean", "max"]:
        out_ref = aggregate_neighbors(empty_graph, op=op, method="reference")
        out_vec = aggregate_neighbors(empty_graph, op=op, method="vectorized")
        assert out_ref.shape == (0, 8)
        assert out_vec.shape == (0, 8)
        assert out_ref.dtype == np.float32
        assert out_vec.dtype == np.float32


# 6. A single-node graph
def test_single_node_graph_isolated_and_self_loop():
    # 6a. Single node without self-loop
    single_isolated = CSRGraph(
        row_ptr=np.array([0, 0], dtype=np.int32),
        col_idx=np.array([], dtype=np.int32),
        features=np.array([[42.0, -7.0]], dtype=np.float32),
        is_directed=True,
    )
    for op in ["sum", "mean", "max"]:
        out = aggregate_neighbors(single_isolated, op=op)
        assert np.array_equal(out[0], np.zeros(2, dtype=np.float32))

    # 6b. Single node with self-loop
    single_loop = CSRGraph(
        row_ptr=np.array([0, 1], dtype=np.int32),
        col_idx=np.array([0], dtype=np.int32),
        features=np.array([[42.0, -7.0]], dtype=np.float32),
        is_directed=True,
    )
    for op in ["sum", "mean", "max"]:
        out = aggregate_neighbors(single_loop, op=op)
        assert np.allclose(out[0], np.array([42.0, -7.0], dtype=np.float32))


# 7. Directed graphs with asymmetric adjacency
def test_directed_graph_asymmetric_adjacency():
    # Directed edge 0 -> 1 only.
    # Node 0 has neighbor 1. Node 1 has no neighbors.
    features = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    graph = CSRGraph.from_edge_list(
        num_nodes=2,
        edge_list=[(0, 1)],
        features=features,
        feature_dim=2,
        is_directed=True,
    )
    for op in ["sum", "mean", "max"]:
        out = aggregate_neighbors(graph, op=op)
        # Node 0 aggregates Node 1's features
        assert np.allclose(out[0], np.array([30.0, 40.0], dtype=np.float32))
        # Node 1 has no neighbors -> zero vector
        assert np.array_equal(out[1], np.zeros(2, dtype=np.float32))


# 8. Undirected graphs with symmetric adjacency
def test_undirected_graph_symmetric_adjacency():
    features = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    graph = CSRGraph.from_edge_list(
        num_nodes=2,
        edge_list=[(0, 1)],
        features=features,
        feature_dim=2,
        is_directed=False,  # creates (0, 1) and (1, 0)
    )
    for op in ["sum", "mean", "max"]:
        out = aggregate_neighbors(graph, op=op)
        # Node 0 aggregates Node 1
        assert np.allclose(out[0], np.array([30.0, 40.0], dtype=np.float32))
        # Node 1 aggregates Node 0
        assert np.allclose(out[1], np.array([10.0, 20.0], dtype=np.float32))


# 9. Graphs containing a hub with high degree
def test_hub_high_degree_node():
    # Hub node 0 connected to 500 peripheral nodes
    num_peripherals = 500
    num_nodes = num_peripherals + 1
    edge_list = [(0, i) for i in range(1, num_nodes)]
    # All peripherals have feature = 1.0, hub has feature = 999.0
    features = np.ones((num_nodes, 4), dtype=np.float32)
    features[0] = 999.0

    hub_graph = CSRGraph.from_edge_list(
        num_nodes=num_nodes,
        edge_list=edge_list,
        features=features,
        feature_dim=4,
        is_directed=True,
    )

    # Hub sum: 500 * 1.0 = 500.0
    sum_out = aggregate_neighbors(hub_graph, op="sum")
    assert np.allclose(sum_out[0], np.full(4, 500.0, dtype=np.float32))

    # Hub mean: 500.0 / 500 = 1.0
    mean_out = aggregate_neighbors(hub_graph, op="mean")
    assert np.allclose(mean_out[0], np.full(4, 1.0, dtype=np.float32))

    # Hub max: max(1.0, ..., 1.0) = 1.0
    max_out = aggregate_neighbors(hub_graph, op="max")
    assert np.allclose(max_out[0], np.full(4, 1.0, dtype=np.float32))

    # Peripherals have out-degree 0 -> all zeros
    for i in range(1, num_nodes):
        assert np.array_equal(sum_out[i], np.zeros(4, dtype=np.float32))


# 10. Duplicate edges and self-loops policies
def test_duplicate_edges_and_self_loops():
    features = np.array([[2.0], [5.0]], dtype=np.float32)
    # Edge list has duplicate (0, 1) and self-loop (0, 0)
    edge_list = [(0, 1), (0, 1), (0, 0)]
    graph = CSRGraph.from_edge_list(
        num_nodes=2,
        edge_list=edge_list,
        features=features,
        feature_dim=1,
        deduplicate=True,
        allow_self_loops=True,
    )
    # Deduplicated edges for node 0 should be: (0, 0) and (0, 1)
    sum_out = aggregate_neighbors(graph, op="sum")
    # Node 0 has neighbors {0, 1} -> 2.0 + 5.0 = 7.0
    assert np.isclose(sum_out[0, 0], 7.0)

    # If self-loops not allowed:
    graph_no_loop = CSRGraph.from_edge_list(
        num_nodes=2,
        edge_list=edge_list,
        features=features,
        feature_dim=1,
        deduplicate=True,
        allow_self_loops=False,
    )
    sum_out_no_loop = aggregate_neighbors(graph_no_loop, op="sum")
    # Only neighbor 1 -> 5.0
    assert np.isclose(sum_out_no_loop[0, 0], 5.0)


# 11. Feature dimensions: 1, 4, 16, 32, 128
@pytest.mark.parametrize("dim", [1, 4, 16, 32, 128])
def test_multiple_feature_dimensions(dim):
    graph = generate_synthetic_graph(
        num_nodes=50,
        avg_degree=4.0,
        feature_dim=dim,
        topology="uniform",
        seed=100 + dim,
    )
    for op in ["sum", "mean", "max"]:
        out = aggregate_neighbors(graph, op=op)
        assert out.shape == (50, dim)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))


# 12. Negative and zero-valued features
def test_negative_and_zero_valued_features():
    features = np.array([[-10.0, -50.0], [-20.0, -10.0], [0.0, 0.0]], dtype=np.float32)
    # Node 0 -> Node 1 (has neighbor with negative features)
    # Node 1 -> Node 2 (has neighbor with zero features)
    # Node 2 -> isolated
    graph = CSRGraph.from_edge_list(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        features=features,
        feature_dim=2,
        is_directed=True,
    )
    max_out = aggregate_neighbors(graph, op="max")
    # Node 0 max of neighbor 1: [-20.0, -10.0] (must be negative, not 0!)
    assert np.allclose(max_out[0], np.array([-20.0, -10.0], dtype=np.float32))
    # Node 1 max of neighbor 2: [0.0, 0.0]
    assert np.allclose(max_out[1], np.array([0.0, 0.0], dtype=np.float32))
    # Node 2 has no neighbors -> explicitly [0.0, 0.0] sentinel
    assert np.array_equal(max_out[2], np.zeros(2, dtype=np.float32))


# 13. Deterministic output across repeated calls
def test_deterministic_output(toy_5node_graph):
    for op in ["sum", "mean", "max"]:
        run1 = aggregate_neighbors(toy_5node_graph, op=op)
        run2 = aggregate_neighbors(toy_5node_graph, op=op)
        assert np.array_equal(run1, run2)


# 14. Input immutability
def test_input_immutability(toy_5node_graph):
    orig_features = toy_5node_graph.features.copy()
    orig_row_ptr = toy_5node_graph.row_ptr.copy()
    orig_col_idx = toy_5node_graph.col_idx.copy()

    for op in ["sum", "mean", "max"]:
        aggregate_neighbors(toy_5node_graph, op=op, method="reference")
        aggregate_neighbors(toy_5node_graph, op=op, method="vectorized")

    assert np.array_equal(toy_5node_graph.features, orig_features)
    assert np.array_equal(toy_5node_graph.row_ptr, orig_row_ptr)
    assert np.array_equal(toy_5node_graph.col_idx, orig_col_idx)


# 15. Invalid graph or aggregation configuration handling
def test_invalid_graph_and_config_handling(toy_5node_graph):
    with pytest.raises(TypeError, match="Expected graph to be an instance of CSRGraph"):
        aggregate_neighbors("not_a_graph")  # type: ignore

    with pytest.raises(ValueError, match="Unknown aggregation operator 'median'"):
        aggregate_neighbors(toy_5node_graph, op="median")

    with pytest.raises(ValueError, match="Unknown aggregation method 'unsupported'"):
        aggregate_neighbors(toy_5node_graph, op="mean", method="unsupported")  # type: ignore


# 16. Correct output shape and dtype
def test_output_shape_and_dtype(toy_5node_graph):
    for op in [AggregationOp.SUM, AggregationOp.MEAN, AggregationOp.MAX]:
        out = aggregate_neighbors(toy_5node_graph, op=op)
        assert out.shape == (5, 4)
        assert out.dtype == np.float32


# 17. Non-finite output / input detection
def test_non_finite_input_rejection():
    # Construct invalid graph manually with NaN feature to test aggregator defense
    nan_features = np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32)
    invalid_graph = object.__new__(CSRGraph)
    invalid_graph.features = nan_features
    invalid_graph.row_ptr = np.array([0, 1, 2], dtype=np.int32)
    invalid_graph.col_idx = np.array([1, 0], dtype=np.int32)
    invalid_graph.num_nodes = 2
    invalid_graph.num_edges = 2
    invalid_graph.feature_dim = 2

    with pytest.raises(ValueError, match="Graph features contain NaN or Infinite values"):
        aggregate_neighbors(invalid_graph, op="mean")


# 18. Cross-verification between reference and vectorized implementations
def test_cross_verification_reference_vs_vectorized():
    graph = generate_synthetic_graph(
        num_nodes=200,
        avg_degree=6.0,
        feature_dim=16,
        topology="power_law",
        seed=42,
    )
    for op in ["sum", "mean", "max"]:
        ref = aggregate_neighbors(graph, op=op, method="reference")
        vec = aggregate_neighbors(graph, op=op, method="vectorized")
        assert np.allclose(ref, vec, atol=1e-5), f"Mismatch between reference and vectorized for op={op}"
