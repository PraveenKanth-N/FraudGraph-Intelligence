"""
Unit and Invariant Tests for Graph Feature Enrichment Pipeline
Module: FGI-MOD-07 (Fraud Scoring & Classifier Baseline)
Course: VIT Vellore — CSI4007 GPU Programming
"""

from pathlib import Path
import numpy as np
import pytest

from backend.cpu_engine.aggregators import aggregate_neighbors
from backend.cpu_engine.csr_graph import CSRGraph
from backend.cpu_engine.synthetic_fraud_generator import load_labeled_graph, temporal_train_test_split
from backend.gpu_engine.cuda_runner import get_cuda_binary_path, is_nvcc_available
from backend.ml_engine.feature_enrichment import (
    AggregationBackend,
    ContextPolicy,
    RepresentationType,
    apply_context_policy,
    compute_neighbor_aggregation,
    enrich_features,
    enrich_features_with_split,
    get_feature_dimension,
)


@pytest.fixture(scope="module")
def canonical_fraud_data():
    """Load canonical synthetic fraud dataset fixture."""
    fixture_path = Path("data/fixtures/synthetic_fraud_10k.npz")
    assert fixture_path.exists(), f"Fixture missing at: {fixture_path}"
    graph, labels, timestamps = load_labeled_graph(fixture_path)
    return graph, labels, timestamps


@pytest.fixture(scope="module")
def toy_fixture_graph():
    """Load toy 5-node fixture containing isolated node 4."""
    fixture_path = Path("data/fixtures/toy_graph_5nodes.npz")
    assert fixture_path.exists(), f"Fixture missing at: {fixture_path}"
    return CSRGraph.load(fixture_path)


def test_raw_representation_dimension_and_identity(canonical_fraud_data):
    """Verify raw representation returns exact copy of X with shape (N, D) and float32 dtype."""
    graph, _, _ = canonical_fraud_data
    x_raw = enrich_features(graph, representation=RepresentationType.RAW)

    assert x_raw.shape == (graph.num_nodes, graph.feature_dim)
    assert x_raw.dtype == np.float32
    np.testing.assert_array_equal(x_raw, graph.features)

    # Immutability check
    x_raw[0, 0] += 99.0
    assert graph.features[0, 0] != x_raw[0, 0]


def test_mean_representation_dimensions_and_concatenation(canonical_fraud_data):
    """Verify mean representation returns shape (N, 2D) with correct concatenation order."""
    graph, _, _ = canonical_fraud_data
    x_mean = enrich_features(graph, representation=RepresentationType.MEAN, backend="cpu_vectorized")

    assert x_mean.shape == (graph.num_nodes, graph.feature_dim * 2)
    assert x_mean.dtype == np.float32

    # Columns 0:D must be raw features X
    np.testing.assert_array_equal(x_mean[:, :graph.feature_dim], graph.features)

    # Columns D:2D must match neighbor mean aggregation
    y_mean = aggregate_neighbors(graph, op="mean", method="vectorized")
    np.testing.assert_allclose(x_mean[:, graph.feature_dim:], y_mean, atol=1e-5)


def test_max_representation_dimensions_and_concatenation(canonical_fraud_data):
    """Verify max representation returns shape (N, 2D) with correct concatenation order."""
    graph, _, _ = canonical_fraud_data
    x_max = enrich_features(graph, representation=RepresentationType.MAX, backend="cpu_vectorized")

    assert x_max.shape == (graph.num_nodes, graph.feature_dim * 2)
    assert x_max.dtype == np.float32

    # Columns 0:D must be raw features X
    np.testing.assert_array_equal(x_max[:, :graph.feature_dim], graph.features)

    # Columns D:2D must match neighbor max aggregation
    y_max = aggregate_neighbors(graph, op="max", method="vectorized")
    np.testing.assert_allclose(x_max[:, graph.feature_dim:], y_max, atol=1e-5)


def test_sum_representation_dimensions_and_concatenation(canonical_fraud_data):
    """Verify sum representation returns shape (N, 2D) with correct concatenation order."""
    graph, _, _ = canonical_fraud_data
    x_sum = enrich_features(graph, representation=RepresentationType.SUM, backend="cpu_vectorized")

    assert x_sum.shape == (graph.num_nodes, graph.feature_dim * 2)
    assert x_sum.dtype == np.float32

    # Columns 0:D must be raw features X
    np.testing.assert_array_equal(x_sum[:, :graph.feature_dim], graph.features)

    # Columns D:2D must match neighbor sum aggregation
    y_sum = aggregate_neighbors(graph, op="sum", method="vectorized")
    np.testing.assert_allclose(x_sum[:, graph.feature_dim:], y_sum, atol=1e-5)


def test_multi_operator_combined_representation(canonical_fraud_data):
    """Verify multi_operator representation returns shape (N, 3D) [X || Y_mean || Y_max]."""
    graph, _, _ = canonical_fraud_data
    d = graph.feature_dim

    x_multi = enrich_features(graph, representation=RepresentationType.MULTI_OPERATOR, backend="cpu_vectorized")
    x_comb = enrich_features(graph, representation="combined", backend="cpu_vectorized")

    assert x_multi.shape == (graph.num_nodes, d * 3)
    assert x_comb.shape == (graph.num_nodes, d * 3)
    np.testing.assert_array_equal(x_multi, x_comb)

    # Order verification: [X || Y_mean || Y_max]
    y_mean = aggregate_neighbors(graph, op="mean", method="vectorized")
    y_max = aggregate_neighbors(graph, op="max", method="vectorized")

    np.testing.assert_array_equal(x_multi[:, :d], graph.features)
    np.testing.assert_allclose(x_multi[:, d:2*d], y_mean, atol=1e-5)
    np.testing.assert_allclose(x_multi[:, 2*d:3*d], y_max, atol=1e-5)


def test_cpu_reference_vs_vectorized_parity(toy_fixture_graph):
    """Verify bitwise/numerical parity between cpu_reference and cpu_vectorized backends."""
    for rep in [RepresentationType.RAW, RepresentationType.MEAN, RepresentationType.MAX, RepresentationType.SUM, RepresentationType.MULTI_OPERATOR]:
        out_ref = enrich_features(toy_fixture_graph, representation=rep, backend=AggregationBackend.CPU_REFERENCE)
        out_vec = enrich_features(toy_fixture_graph, representation=rep, backend=AggregationBackend.CPU_VECTORIZED)

        assert out_ref.shape == out_vec.shape
        np.testing.assert_allclose(out_ref, out_vec, atol=1e-5)


def test_zero_degree_nodes_produce_zero_sentinels(toy_fixture_graph):
    """
    Verify that zero-degree nodes produce exact zeros for aggregated columns,
    preserving the FGI zero-sentinel contract (d_i = 0 => 0).
    """
    # In toy_fixture_graph, node 4 is isolated (degree = 0)
    degrees = toy_fixture_graph.get_degrees()
    assert degrees[4] == 0

    d = toy_fixture_graph.feature_dim
    x_mean = enrich_features(toy_fixture_graph, representation=RepresentationType.MEAN)
    x_max = enrich_features(toy_fixture_graph, representation=RepresentationType.MAX)
    x_comb = enrich_features(toy_fixture_graph, representation=RepresentationType.COMBINED)

    # Node 4: raw features are preserved
    np.testing.assert_array_equal(x_mean[4, :d], toy_fixture_graph.features[4])

    # Node 4: neighbor features must be exactly 0.0
    np.testing.assert_array_equal(x_mean[4, d:], np.zeros(d, dtype=np.float32))
    np.testing.assert_array_equal(x_max[4, d:], np.zeros(d, dtype=np.float32))
    np.testing.assert_array_equal(x_comb[4, d:], np.zeros(2 * d, dtype=np.float32))


def test_get_feature_dimension_utility():
    """Verify dimension lookup utility for all representation types."""
    base_d = 32
    assert get_feature_dimension(base_d, RepresentationType.RAW) == 32
    assert get_feature_dimension(base_d, RepresentationType.MEAN) == 64
    assert get_feature_dimension(base_d, RepresentationType.MAX) == 64
    assert get_feature_dimension(base_d, RepresentationType.SUM) == 64
    assert get_feature_dimension(base_d, RepresentationType.MULTI_OPERATOR) == 96
    assert get_feature_dimension(base_d, "combined") == 96

    with pytest.raises(ValueError):
        get_feature_dimension(base_d, "invalid_rep")
    with pytest.raises(ValueError):
        get_feature_dimension(-5, RepresentationType.RAW)


def test_empty_and_zero_feature_graphs():
    """Verify graceful handling of boundary conditions: N=0 or D=0 graphs."""
    empty_graph = CSRGraph(
        row_ptr=np.array([0], dtype=np.int32),
        col_idx=np.empty(0, dtype=np.int32),
        features=np.empty((0, 16), dtype=np.float32),
        is_directed=True,
    )
    res_raw = enrich_features(empty_graph, representation=RepresentationType.RAW)
    assert res_raw.shape == (0, 16)

    res_mean = enrich_features(empty_graph, representation=RepresentationType.MEAN)
    assert res_mean.shape == (0, 32)

    res_comb = enrich_features(empty_graph, representation=RepresentationType.COMBINED)
    assert res_comb.shape == (0, 48)


def test_invalid_arguments_and_error_handling(toy_fixture_graph):
    """Verify that invalid arguments raise informative ValueErrors and TypeErrors."""
    # Invalid graph type
    with pytest.raises(TypeError, match="Expected graph to be an instance of CSRGraph"):
        enrich_features("not_a_graph")

    # Invalid representation
    with pytest.raises(ValueError, match="Unknown representation"):
        enrich_features(toy_fixture_graph, representation="quantum_rep")

    # Invalid backend
    with pytest.raises(ValueError, match="Unknown aggregation backend"):
        enrich_features(toy_fixture_graph, backend="tpu_cloud")

    # Invalid context policy
    with pytest.raises(ValueError, match="Unknown context policy"):
        enrich_features(toy_fixture_graph, context_policy="future_peeking")

    # TRAIN_ONLY_CONTEXT without train_end
    with pytest.raises(ValueError, match="train_end must be explicitly provided"):
        enrich_features(toy_fixture_graph, context_policy=ContextPolicy.TRAIN_ONLY_CONTEXT)

    # TRAIN_ONLY_CONTEXT with out-of-range train_end
    with pytest.raises(ValueError, match="train_end must be in"):
        enrich_features(toy_fixture_graph, context_policy=ContextPolicy.TRAIN_ONLY_CONTEXT, train_end=999)


def test_finite_outputs_and_float32_dtype(canonical_fraud_data):
    """Verify that all enriched matrices are 100% finite and strictly float32."""
    graph, _, _ = canonical_fraud_data
    for rep in [RepresentationType.RAW, RepresentationType.MEAN, RepresentationType.MAX, RepresentationType.COMBINED]:
        res = enrich_features(graph, representation=rep, backend="cpu_vectorized")
        assert res.dtype == np.float32
        assert np.all(np.isfinite(res)), f"Non-finite values found in {rep}"


def test_temporal_causality_and_context_policies(canonical_fraud_data):
    """
    Verify that in TRAIN_ONLY_CONTEXT policy:
    1. Training nodes (i < train_end) have identical features to CAUSAL_HISTORY (zero difference).
    2. Test nodes (i >= train_end) only aggregate from training nodes (j < train_end).
    """
    graph, _, _ = canonical_fraud_data
    train_end = 7000

    x_causal = enrich_features(graph, representation=RepresentationType.MEAN, context_policy=ContextPolicy.CAUSAL_HISTORY)
    x_train_only = enrich_features(
        graph,
        representation=RepresentationType.MEAN,
        context_policy=ContextPolicy.TRAIN_ONLY_CONTEXT,
        train_end=train_end,
    )

    # Training nodes [0 : train_end] must be bitwise identical!
    np.testing.assert_array_equal(x_causal[:train_end], x_train_only[:train_end])

    # Validation and test nodes [train_end :] differ because edges into non-train nodes are pruned
    assert not np.array_equal(x_causal[train_end:], x_train_only[train_end:])

    # Confirm that under TRAIN_ONLY_CONTEXT, the masked graph has zero edges pointing to j >= train_end
    masked_graph = apply_context_policy(graph, policy=ContextPolicy.TRAIN_ONLY_CONTEXT, train_end=train_end)
    assert np.all(masked_graph.col_idx < train_end), "Found edge pointing to non-training node under TRAIN_ONLY_CONTEXT"


def test_train_val_test_partition_enrichment(canonical_fraud_data):
    """Verify enrich_features_with_split produces non-overlapping arrays of correct shapes."""
    graph, _, _ = canonical_fraud_data
    train_idx, val_idx, test_idx = temporal_train_test_split(graph.num_nodes, 0.70, 0.15, 0.15)

    x_train, x_val, x_test = enrich_features_with_split(
        graph,
        train_idx,
        val_idx,
        test_idx,
        representation=RepresentationType.MEAN,
        context_policy=ContextPolicy.TRAIN_ONLY_CONTEXT,
    )

    assert x_train.shape == (7000, 64)
    assert x_val.shape == (1500, 64)
    assert x_test.shape == (1500, 64)
    assert x_train.dtype == np.float32
    assert x_val.dtype == np.float32
    assert x_test.dtype == np.float32


def test_labels_strictly_quarantined_from_enriched_features(canonical_fraud_data):
    """
    Verify that ground-truth labels y are never embedded into enriched matrices,
    never added as a column, and no single column exhibits trivial correlation.
    """
    graph, labels, _ = canonical_fraud_data

    for rep in [RepresentationType.RAW, RepresentationType.MEAN, RepresentationType.MAX, RepresentationType.COMBINED]:
        enriched = enrich_features(graph, representation=rep)

        # 1. No column is identical to labels
        for k in range(enriched.shape[1]):
            col = enriched[:, k]
            assert not np.array_equal(col, labels)
            # Correlation bounded strictly below 0.35
            corr = float(np.corrcoef(col, labels)[0, 1])
            assert abs(corr) < 0.35, f"Column {k} in {rep} has excessive target correlation: {corr:.4f}"


def test_existing_fixtures_and_reference_implementations_unmodified():
    """Ensure Phase 3, 4, 5, and 6 fixtures and engines are preserved without modification."""
    toy_path = Path("data/fixtures/toy_graph_5nodes.npz")
    tier_a_path = Path("data/fixtures/synthetic_tier_a_10k.npz")
    skewed_path = Path("data/fixtures/synthetic_skewed_10k.npz")

    assert toy_path.exists()
    assert tier_a_path.exists()
    assert skewed_path.exists()

    toy = CSRGraph.load(toy_path)
    assert toy.num_nodes == 5
    assert toy.num_edges == 8

    tier_a = CSRGraph.load(tier_a_path)
    assert tier_a.num_nodes == 10000
    assert tier_a.num_edges == 94945

    skewed = CSRGraph.load(skewed_path)
    assert skewed.num_nodes == 10000
    assert skewed.num_edges == 94970


@pytest.mark.skipif(
    not is_nvcc_available() or not get_cuda_binary_path("streams").exists(),
    reason="Local CUDA binary not available (run Google Colab workflow on Tesla T4)",
)
def test_live_cuda_gpu_parity_if_available(canonical_fraud_data):
    """
    When GPU runner is compiled and available, verify GPU representation agrees
    with CPU reference representation within tolerance (atol <= 1e-5).
    """
    graph, _, _ = canonical_fraud_data

    x_cpu = enrich_features(graph, representation=RepresentationType.MEAN, backend=AggregationBackend.CPU_REFERENCE)
    x_gpu = enrich_features(graph, representation=RepresentationType.MEAN, backend=AggregationBackend.CUDA_STREAMS)

    np.testing.assert_allclose(x_gpu, x_cpu, atol=1e-5)
