"""
Unit and Invariant Tests for Deterministic Synthetic Fraud Generator
Module: FGI-MOD-07 (Fraud Scoring & Classifier Baseline)
Course: VIT Vellore — CSI4007 GPU Programming
"""

import json
from pathlib import Path

import numpy as np
import pytest

from backend.cpu_engine.csr_graph import CSRGraph
from backend.cpu_engine.synthetic_fraud_generator import (
    GENERATOR_VERSION,
    generate_synthetic_fraud_graph,
    load_labeled_graph,
    save_labeled_graph,
    temporal_train_test_split,
)


@pytest.fixture(scope="module")
def default_fraud_graph():
    """Generates the canonical 10k synthetic fraud graph once for read-only invariant assertions."""
    graph, labels, timestamps = generate_synthetic_fraud_graph(
        num_nodes=10000,
        feature_dim=32,
        target_fraud_rate=0.035,
        seed=42,
    )
    return graph, labels, timestamps


def test_deterministic_generation_same_seed():
    """Verify that generating with the same seed produces bitwise identical arrays."""
    g1, y1, t1 = generate_synthetic_fraud_graph(num_nodes=1000, feature_dim=16, seed=42)
    g2, y2, t2 = generate_synthetic_fraud_graph(num_nodes=1000, feature_dim=16, seed=42)

    np.testing.assert_array_equal(g1.row_ptr, g2.row_ptr)
    np.testing.assert_array_equal(g1.col_idx, g2.col_idx)
    np.testing.assert_array_equal(g1.features, g2.features)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(t1, t2)


def test_different_seeds_produce_different_data():
    """Verify that distinct seeds produce statistically and structurally distinct datasets."""
    g1, y1, _ = generate_synthetic_fraud_graph(num_nodes=1000, feature_dim=16, seed=42)
    g2, y2, _ = generate_synthetic_fraud_graph(num_nodes=1000, feature_dim=16, seed=999)

    assert not np.array_equal(g1.features, g2.features)
    assert not np.array_equal(y1, y2)


def test_exact_positive_fraud_count(default_fraud_graph):
    """Verify that exactly 350 positive fraud labels (3.50%) are generated for N=10,000."""
    _, labels, _ = default_fraud_graph

    assert len(labels) == 10000
    total_frauds = int(np.sum(labels))
    assert total_frauds == 350, f"Expected exactly 350 frauds, got {total_frauds}"

    unique_vals = set(np.unique(labels))
    assert unique_vals == {0, 1}, f"Labels must be strictly binary, found {unique_vals}"


def test_feature_matrix_shape_and_finiteness(default_fraud_graph):
    """Verify feature matrix dimensions and ensure zero NaNs or Infs."""
    graph, _, _ = default_fraud_graph

    assert graph.features.shape == (10000, 32)
    assert graph.features.dtype == np.float32
    assert np.all(np.isfinite(graph.features)), "Features contain non-finite elements (NaN or Inf)"


def test_no_label_leakage_into_feature_matrix(default_fraud_graph):
    """
    Ensure the 32-dimensional feature matrix contains no direct target column,
    exact copy, or trivially predictive linear shortcut to the label.
    """
    graph, labels, _ = default_fraud_graph
    features = graph.features

    for k in range(graph.feature_dim):
        col = features[:, k]
        # 1. Column cannot be binary {0, 1}
        assert set(np.unique(col)) != {0, 1}, f"Feature column {k} contains raw binary values"
        # 2. Check correlation with labels: must be < 0.35 to prevent trivial separation
        corr = float(np.corrcoef(col, labels)[0, 1])
        assert abs(corr) < 0.35, f"Feature column {k} has excessive correlation with target: {corr:.4f}"


def test_csr_invariants_and_node_indices(default_fraud_graph):
    """Validate all standard CSR invariants via CSRGraph validator."""
    graph, _, _ = default_fraud_graph

    # CSRGraph.validate() raises GraphValidationError if invariants are violated
    graph.validate()
    assert graph.num_nodes == 10000
    assert graph.num_edges == len(graph.col_idx)
    assert graph.row_ptr[0] == 0
    assert graph.row_ptr[-1] == graph.num_edges
    assert np.all(graph.col_idx >= 0)
    assert np.all(graph.col_idx < graph.num_nodes)


def test_valid_chronological_timestamps(default_fraud_graph):
    """Verify timestamps are 1D float64, non-negative, and strictly monotonically non-decreasing."""
    _, _, timestamps = default_fraud_graph

    assert timestamps.shape == (10000,)
    assert timestamps.dtype == np.float64
    assert timestamps[0] == 0.0
    assert np.all(timestamps >= 0.0)
    # Check monotonicity
    assert np.all(timestamps[1:] >= timestamps[:-1]), "Timestamps are not chronologically sorted"


def test_temporal_causality_zero_future_leakage(default_fraud_graph):
    """
    Verify that in CSR adjacency, every neighbor j stored for query node i
    satisfies t_j <= t_i, ensuring zero future-to-past information leakage.
    """
    graph, _, timestamps = default_fraud_graph
    row_ptr = graph.row_ptr
    col_idx = graph.col_idx

    future_leakage_count = 0
    for i in range(graph.num_nodes):
        t_i = timestamps[i]
        for p in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[p]
            if timestamps[j] > t_i:
                future_leakage_count += 1

    assert future_leakage_count == 0, f"Found {future_leakage_count} future-directed leakage edges"


def test_save_and_load_round_trip(tmp_path):
    """Verify that saving and loading labeled graphs preserves all arrays and manifest structures."""
    g_orig, y_orig, t_orig = generate_synthetic_fraud_graph(num_nodes=500, feature_dim=8, seed=123)

    save_path = tmp_path / "test_fraud_graph.npz"
    saved_file = save_labeled_graph(save_path, g_orig, y_orig, t_orig)
    assert saved_file.exists()

    g_loaded, y_loaded, t_loaded = load_labeled_graph(saved_file)

    np.testing.assert_array_equal(g_orig.row_ptr, g_loaded.row_ptr)
    np.testing.assert_array_equal(g_orig.col_idx, g_loaded.col_idx)
    np.testing.assert_array_equal(g_orig.features, g_loaded.features)
    np.testing.assert_array_equal(y_orig, y_loaded)
    np.testing.assert_array_equal(t_orig, t_loaded)

    # Verify standard CSRGraph.load also succeeds
    g_csr = CSRGraph.load(saved_file)
    assert g_csr.num_nodes == g_orig.num_nodes
    assert g_csr.num_edges == g_orig.num_edges


def test_temporal_train_test_split_utility():
    """Verify chronological split partitions indices disjointly and completely without overlap."""
    n = 1000
    train_idx, val_idx, test_idx = temporal_train_test_split(n, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    assert len(train_idx) == 700
    assert len(val_idx) == 150
    assert len(test_idx) == 150

    # Disjointness
    all_indices = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_indices) == 1000
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(val_idx).isdisjoint(set(test_idx))
    assert set(train_idx).isdisjoint(set(test_idx))

    # Chronological order
    assert np.max(train_idx) < np.min(val_idx)
    assert np.max(val_idx) < np.min(test_idx)


def test_existing_graph_fixtures_remain_unchanged():
    """Confirm that Phase 3 and Phase 6 unlabeled fixtures are not modified or corrupted."""
    fixtures_dir = Path("data/fixtures")

    toy_path = fixtures_dir / "toy_graph_5nodes.npz"
    tier_a_path = fixtures_dir / "synthetic_tier_a_10k.npz"
    skewed_path = fixtures_dir / "synthetic_skewed_10k.npz"

    assert toy_path.exists(), "toy_graph_5nodes.npz is missing"
    assert tier_a_path.exists(), "synthetic_tier_a_10k.npz is missing"
    assert skewed_path.exists(), "synthetic_skewed_10k.npz is missing"

    # Verify node counts
    toy = CSRGraph.load(toy_path)
    assert toy.num_nodes == 5
    assert toy.num_edges == 8

    tier_a = CSRGraph.load(tier_a_path)
    assert tier_a.num_nodes == 10000
    assert tier_a.num_edges == 94945

    skewed = CSRGraph.load(skewed_path)
    assert skewed.num_nodes == 10000
    assert skewed.num_edges == 94970


def test_manifest_contains_generator_version(tmp_path):
    """Verify newly saved graph manifests contain generator_version '1.0.0'."""
    g, y, t = generate_synthetic_fraud_graph(num_nodes=100, feature_dim=8, seed=42)
    save_path = tmp_path / "version_check.npz"
    saved_file = save_labeled_graph(save_path, g, y, t)

    with np.load(saved_file, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        assert "generator_version" in manifest
        assert manifest["generator_version"] == GENERATOR_VERSION
        assert manifest["generator_version"] == "1.0.0"

    g_loaded, _, _ = load_labeled_graph(saved_file)
    assert g_loaded.metadata.get("generator_version") == "1.0.0"


def test_fixed_seed_preserves_canonical_array_equality():
    """Verify that generating with seed=42 reproduces 100% identical arrays to canonical fixture."""
    canonical_path = Path("data/fixtures/synthetic_fraud_10k.npz")
    assert canonical_path.exists(), "Canonical fixture synthetic_fraud_10k.npz missing"

    g_canon, y_canon, t_canon = load_labeled_graph(canonical_path)
    g_fresh, y_fresh, t_fresh = generate_synthetic_fraud_graph(
        num_nodes=10000,
        feature_dim=32,
        target_fraud_rate=0.035,
        seed=42,
    )

    np.testing.assert_array_equal(g_fresh.row_ptr, g_canon.row_ptr)
    np.testing.assert_array_equal(g_fresh.col_idx, g_canon.col_idx)
    np.testing.assert_array_equal(g_fresh.features, g_canon.features)
    np.testing.assert_array_equal(y_fresh, y_canon)
    np.testing.assert_array_equal(t_fresh, t_canon)


def test_historical_fixture_without_generator_version_readable(tmp_path):
    """Verify that historical fixtures lacking generator_version remain fully readable."""
    canonical_path = Path("data/fixtures/synthetic_fraud_10k.npz")
    g_canon, y_canon, t_canon = load_labeled_graph(canonical_path)

    assert g_canon.num_nodes == 10000
    assert g_canon.num_edges == 231984
    assert int(np.sum(y_canon)) == 350
    assert len(t_canon) == 10000

    # Also test an explicit legacy archive created without generator_version in manifest
    legacy_manifest = {
        "num_nodes": 50,
        "num_edges": 80,
        "feature_dim": 4,
        "is_directed": True,
        "stats": {},
        "metadata": {"legacy_seed": 7},
        "fraud_stats": {"fraud_count": 2, "licit_count": 48, "fraud_percentage": 4.0},
        "temporal_stats": {"min_timestamp_sec": 0.0, "max_timestamp_sec": 100.0},
        "array_schema": {},
        "idx_to_node_id": None,
    }
    legacy_file = tmp_path / "legacy_graph.npz"
    row_ptr = np.zeros(51, dtype=np.int32)
    row_ptr[-1] = 80
    col_idx = np.zeros(80, dtype=np.int32)
    features = np.zeros((50, 4), dtype=np.float32)
    labels = np.zeros(50, dtype=np.int32)
    labels[0] = 1
    labels[1] = 1
    timestamps = np.linspace(0.0, 100.0, 50, dtype=np.float64)

    np.savez_compressed(
        legacy_file,
        row_ptr=row_ptr,
        col_idx=col_idx,
        features=features,
        labels=labels,
        timestamps=timestamps,
        manifest=json.dumps(legacy_manifest),
    )

    g_leg, y_leg, t_leg = load_labeled_graph(legacy_file)
    assert g_leg.num_nodes == 50
    assert g_leg.metadata.get("legacy_seed") == 7
    assert g_leg.metadata.get("generator_version") is None
    assert int(np.sum(y_leg)) == 2
