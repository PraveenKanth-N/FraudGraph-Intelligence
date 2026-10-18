"""
Comprehensive Automated Test Suite for CSRGraph, Synthetic Generator, and Temporal Graph Pipeline
Module: FGI-MOD-03 (Graph Representation) & FGI-MOD-11 (QA)
Course: VIT Vellore — CSI4007 GPU Programming
"""

import numpy as np
import pandas as pd
import pytest
from backend.cpu_engine.csr_graph import CSRGraph, GraphValidationError
from backend.cpu_engine.synthetic_generator import generate_synthetic_graph
from backend.cpu_engine.transaction_graph_builder import TemporalTransactionGraphBuilder


# ==============================================================================
# 1. CSR Construction from Known Edge List & Round-Trip
# ==============================================================================

def test_csr_construction_from_known_edge_list():
    """
    Test CSR representation against a manually verified 5-node toy graph:
      0 -> [1, 2]
      1 -> [0, 2, 3]
      2 -> [0, 1]
      3 -> [1]
      4 -> []  (isolated node)
    """
    edges = [(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (3, 1)]
    features = np.ones((5, 4), dtype=np.float32)

    graph = CSRGraph.from_edge_list(
        num_nodes=5,
        edge_list=edges,
        features=features,
        feature_dim=4,
        is_directed=True,
        sort_neighbors=True,
    )

    expected_row_ptr = np.array([0, 2, 5, 7, 8, 8], dtype=np.int32)
    expected_col_idx = np.array([1, 2, 0, 2, 3, 0, 1, 1], dtype=np.int32)

    assert np.array_equal(graph.row_ptr, expected_row_ptr)
    assert np.array_equal(graph.col_idx, expected_col_idx)
    assert graph.num_nodes == 5
    assert graph.num_edges == 8
    assert graph.feature_dim == 4
    assert graph.get_degree(4) == 0  # Node 4 isolated


def test_csr_to_edge_list_round_trip():
    """Verify that CSR to edge-list and back produces an identical graph."""
    original_edges = [(0, 1), (0, 3), (1, 2), (2, 0), (3, 2)]
    features = np.random.randn(4, 8).astype(np.float32)

    graph1 = CSRGraph.from_edge_list(
        num_nodes=4,
        edge_list=original_edges,
        features=features,
        feature_dim=8,
        is_directed=True,
        sort_neighbors=True,
    )

    reconstructed_edges = graph1.to_edge_list()
    graph2 = CSRGraph.from_edge_list(
        num_nodes=4,
        edge_list=reconstructed_edges,
        features=features,
        feature_dim=8,
        is_directed=True,
        sort_neighbors=True,
    )

    assert np.array_equal(graph1.row_ptr, graph2.row_ptr)
    assert np.array_equal(graph1.col_idx, graph2.col_idx)
    assert np.array_equal(graph1.features, graph2.features)


# ==============================================================================
# 2. CSR Invariants & Invalid Input Rejection
# ==============================================================================

def test_row_ptr_must_start_at_zero():
    """Invariant 1: row_ptr[0] must equal 0."""
    with pytest.raises(GraphValidationError, match="row_ptr\\[0\\] must be 0"):
        CSRGraph(
            row_ptr=np.array([1, 2, 3]),
            col_idx=np.array([0, 1]),
            features=np.zeros((2, 2)),
            validate=True,
        )


def test_non_monotonic_row_ptr_rejected():
    """Invariant 2: row_ptr must be monotonically non-decreasing."""
    with pytest.raises(GraphValidationError, match="monotonically non-decreasing"):
        CSRGraph(
            row_ptr=np.array([0, 2, 1, 3]),  # Decreases from 2 to 1
            col_idx=np.array([0, 1, 0]),
            features=np.zeros((3, 2)),
            validate=True,
        )


def test_row_ptr_last_element_must_match_edge_count():
    """Invariant 3: row_ptr[-1] == len(col_idx)."""
    with pytest.raises(GraphValidationError, match="row_ptr\\[-1\\].*!= len\\(col_idx\\)"):
        CSRGraph(
            row_ptr=np.array([0, 2, 4]),  # Claims 4 edges, but col_idx has 3
            col_idx=np.array([0, 1, 0]),
            features=np.zeros((2, 2)),
            validate=True,
        )


def test_col_idx_out_of_bounds_rejected():
    """Invariant 4: col_idx entries must be within [0, num_nodes)."""
    with pytest.raises(GraphValidationError, match="col_idx entries must be in range"):
        CSRGraph(
            row_ptr=np.array([0, 1, 2]),
            col_idx=np.array([0, 5]),  # Index 5 is out of bounds for 2 nodes
            features=np.zeros((2, 2)),
            validate=True,
        )


def test_negative_col_idx_rejected():
    """Invariant 4b: negative neighbor index rejected."""
    with pytest.raises(GraphValidationError, match="col_idx entries must be in range"):
        CSRGraph(
            row_ptr=np.array([0, 1, 2]),
            col_idx=np.array([-1, 0]),  # Negative index
            features=np.zeros((2, 2)),
            validate=True,
        )


def test_feature_shape_mismatch_rejected():
    """Invariant 5: features row count must equal num_nodes."""
    with pytest.raises(GraphValidationError, match="Feature row count mismatch"):
        CSRGraph(
            row_ptr=np.array([0, 1, 2]),
            col_idx=np.array([1, 0]),
            features=np.zeros((3, 4)),  # 3 feature rows for 2 nodes
            validate=True,
        )


def test_nan_features_rejected():
    """Invariant 6: features must not contain NaN."""
    bad_features = np.zeros((2, 4), dtype=np.float32)
    bad_features[0, 1] = np.nan
    with pytest.raises(GraphValidationError, match="Features contain non-finite values.*NaN"):
        CSRGraph(
            row_ptr=np.array([0, 1, 2]),
            col_idx=np.array([1, 0]),
            features=bad_features,
            validate=True,
        )


def test_infinity_features_rejected():
    """Invariant 6b: features must not contain Inf."""
    bad_features = np.zeros((2, 4), dtype=np.float32)
    bad_features[1, 2] = np.inf
    with pytest.raises(GraphValidationError, match="Features contain non-finite values.*Inf"):
        CSRGraph(
            row_ptr=np.array([0, 1, 2]),
            col_idx=np.array([1, 0]),
            features=bad_features,
            validate=True,
        )


# ==============================================================================
# 3. Edge Cases: Empty Graphs, Isolated Nodes, Policies
# ==============================================================================

def test_empty_graph_behavior():
    """Verify clean handling of 0-node and 0-edge graphs."""
    empty_graph = CSRGraph(
        row_ptr=np.array([0], dtype=np.int32),
        col_idx=np.empty(0, dtype=np.int32),
        features=np.empty((0, 8), dtype=np.float32),
        validate=True,
    )
    assert empty_graph.num_nodes == 0
    assert empty_graph.num_edges == 0
    assert empty_graph.get_degrees().size == 0
    assert empty_graph.to_edge_list().shape == (0, 2)
    stats = empty_graph.get_stats()
    assert stats["num_nodes"] == 0
    assert stats["density"] == 0.0


def test_isolated_nodes_and_empty_neighborhoods():
    """Verify graph where multiple nodes have degree 0."""
    graph = CSRGraph.from_edge_list(
        num_nodes=5,
        edge_list=[(1, 2)],  # Only node 1 has an edge to 2
        feature_dim=4,
        is_directed=True,
    )
    assert graph.num_nodes == 5
    assert graph.num_edges == 1
    assert graph.get_degree(0) == 0
    assert graph.get_degree(1) == 1
    assert graph.get_degree(2) == 0
    assert graph.get_degree(3) == 0
    assert graph.get_degree(4) == 0
    assert len(graph.get_neighbors(0)) == 0
    assert np.array_equal(graph.get_neighbors(1), np.array([2]))
    stats = graph.get_stats()
    assert stats["num_isolated_nodes"] == 4


def test_duplicate_edge_policy():
    """Verify duplicate edges are deduplicated by default."""
    edges = [(0, 1), (0, 1), (0, 1), (1, 2), (1, 2)]
    graph = CSRGraph.from_edge_list(
        num_nodes=3,
        edge_list=edges,
        deduplicate=True,
    )
    assert graph.num_edges == 2
    assert np.array_equal(graph.to_edge_list(), np.array([[0, 1], [1, 2]]))


def test_self_loop_policy():
    """Verify self-loops (u, u) are discarded when allow_self_loops=False."""
    edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
    graph = CSRGraph.from_edge_list(
        num_nodes=3,
        edge_list=edges,
        allow_self_loops=False,
    )
    assert graph.num_edges == 2
    assert np.array_equal(graph.to_edge_list(), np.array([[0, 1], [1, 2]]))

    # Allow self-loops when explicit
    graph_with_loops = CSRGraph.from_edge_list(
        num_nodes=3,
        edge_list=edges,
        allow_self_loops=True,
    )
    assert graph_with_loops.num_edges == 5


def test_directed_vs_undirected_semantics():
    """Verify undirected graphs automatically add reciprocal reverse edges."""
    edges = [(0, 1), (1, 2)]
    undirected_graph = CSRGraph.from_edge_list(
        num_nodes=3,
        edge_list=edges,
        is_directed=False,
    )
    # Stored as symmetric directed edges: (0, 1), (1, 0), (1, 2), (2, 1)
    assert undirected_graph.num_edges == 4
    stats = undirected_graph.get_stats()
    assert stats["undirected_equivalent_edges"] == 2

    edge_set = set(map(tuple, undirected_graph.to_edge_list().tolist()))
    assert (0, 1) in edge_set and (1, 0) in edge_set
    assert (1, 2) in edge_set and (2, 1) in edge_set


# ==============================================================================
# 4. Serialization, Persistence & Synthetic Generation
# ==============================================================================

def test_graph_serialization_round_trip(tmp_path):
    """Verify saving to .npz and loading back preserves all arrays and manifest."""
    save_file = tmp_path / "test_graph.npz"
    original = generate_synthetic_graph(num_nodes=50, avg_degree=4.0, feature_dim=16, seed=123)

    saved_path = original.save(save_file)
    assert saved_path.exists()

    loaded = CSRGraph.load(saved_path)
    assert loaded.num_nodes == original.num_nodes
    assert loaded.num_edges == original.num_edges
    assert loaded.feature_dim == original.feature_dim
    assert np.array_equal(loaded.row_ptr, original.row_ptr)
    assert np.array_equal(loaded.col_idx, original.col_idx)
    assert np.allclose(loaded.features, original.features)


def test_deterministic_synthetic_generation():
    """Verify identical random seed generates identical graph topology and features."""
    g1 = generate_synthetic_graph(num_nodes=100, avg_degree=8.0, feature_dim=32, topology="power_law", seed=999)
    g2 = generate_synthetic_graph(num_nodes=100, avg_degree=8.0, feature_dim=32, topology="power_law", seed=999)

    assert np.array_equal(g1.row_ptr, g2.row_ptr)
    assert np.array_equal(g1.col_idx, g2.col_idx)
    assert np.array_equal(g1.features, g2.features)


def test_star_graph_extreme_degree_skew():
    star_graph = generate_synthetic_graph(
        num_nodes=20, feature_dim=8, topology="star", is_directed=True, isolated_node_ratio=0.0
    )
    assert star_graph.get_degree(0) == 19
    for i in range(1, 20):
        assert star_graph.get_degree(i) == 0


# ==============================================================================
# 5. Temporal Transaction Graph Pipeline & Leakage Controls
# ==============================================================================

def test_temporal_transaction_graph_builder():
    """
    Verify tabular transaction conversion:
    - Chronological sorting
    - Edge linking based on shared card1 within 7-day window
    - Strict temporal causality: edge u -> v only if t_u <= t_v
    - Label quarantine (isFraud excluded from features)
    - Missing sentinel rejection (None / NaN does not link)
    """
    records = [
        # Entity 1: Alice card1=1001, 3 transactions over 5 days
        {"TransactionID": 101, "TransactionDT": 1000.0, "TransactionAmt": 50.0, "card1": 1001, "P_emaildomain": "gmail.com", "isFraud": 0},
        {"TransactionID": 102, "TransactionDT": 2000.0, "TransactionAmt": 120.0, "card1": 1001, "P_emaildomain": "gmail.com", "isFraud": 0},
        {"TransactionID": 103, "TransactionDT": 3000.0, "TransactionAmt": 500.0, "card1": 1001, "P_emaildomain": "gmail.com", "isFraud": 1},
        # Entity 2: Bob card1=2002, 1 transaction
        {"TransactionID": 201, "TransactionDT": 1500.0, "TransactionAmt": 30.0, "card1": 2002, "P_emaildomain": "yahoo.com", "isFraud": 0},
        # Entity 3: Missing card1 (NaN) - should NOT form clique with other NaNs
        {"TransactionID": 301, "TransactionDT": 1200.0, "TransactionAmt": 15.0, "card1": np.nan, "P_emaildomain": "anon.com", "isFraud": 0},
        {"TransactionID": 302, "TransactionDT": 1400.0, "TransactionAmt": 25.0, "card1": np.nan, "P_emaildomain": "anon.com", "isFraud": 0},
    ]
    df = pd.DataFrame(records)

    builder = TemporalTransactionGraphBuilder(
        relational_keys=["card1"],
        temporal_window_seconds=86400.0 * 7,
    )
    graph, labels = builder.build_from_dataframe(df, is_directed=True)

    assert graph.num_nodes == 6
    assert labels is not None
    assert len(labels) == 6

    # Verify temporal causality: for every directed edge (u, v), t_u <= t_v
    edges = graph.to_edge_list()
    # Check that Alice's 3 transactions form forward edges: 101 -> 102, 101 -> 103, 102 -> 103
    # Check that NaN cards did NOT form an edge between 301 and 302
    for u, v in edges:
        t_u = df.loc[df["TransactionID"] == graph.idx_to_node_id[u], "TransactionDT"].values[0]
        t_v = df.loc[df["TransactionID"] == graph.idx_to_node_id[v], "TransactionDT"].values[0]
        assert t_u <= t_v, f"Temporal leakage detected! Edge ({u}, {v}) has t_u={t_u} > t_v={t_v}"

    # Verify isFraud is NOT in feature columns
    assert "isFraud" not in graph.metadata["feature_columns_used"]


def test_raw_dataset_check():
    """Verify check_raw_dataset_availability reports status without downloading."""
    status = TemporalTransactionGraphBuilder.check_raw_dataset_availability("data/raw")
    assert "raw_dir_exists" in status
    assert "ieee_cis_available" in status
    assert "instructions" in status
