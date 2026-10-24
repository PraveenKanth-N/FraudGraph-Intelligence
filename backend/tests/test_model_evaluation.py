"""
Unit and Invariant Tests for Controlled Fraud Model Evaluation Engine
Module: FGI-MOD-07 (Fraud Scoring & Classifier Baseline)
Course: VIT Vellore — CSI4007 GPU Programming
"""

from pathlib import Path
import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler

from backend.cpu_engine.synthetic_fraud_generator import load_labeled_graph, temporal_train_test_split
from backend.ml_engine.feature_enrichment import ContextPolicy, RepresentationType
from backend.ml_engine.model_evaluation import (
    compute_metrics,
    evaluate_single_model,
    select_optimal_f1_threshold,
    select_recall_target_threshold,
)


@pytest.fixture(scope="module")
def canonical_fraud_fixture():
    """Load canonical synthetic fraud dataset fixture."""
    fixture_path = Path("data/fixtures/synthetic_fraud_10k.npz")
    assert fixture_path.exists(), f"Fixture missing at: {fixture_path}"
    graph, labels, timestamps = load_labeled_graph(fixture_path)
    return graph, labels, timestamps


def test_chronological_split_boundaries_and_label_separation(canonical_fraud_fixture):
    """
    Verify chronological split partitions [0:7000), [7000:8500), [8500:10000)
    contain exactly 239, 58, and 53 positive fraud labels with strict temporal bounds.
    """
    graph, labels, timestamps = canonical_fraud_fixture
    train_idx, val_idx, test_idx = temporal_train_test_split(graph.num_nodes, 0.70, 0.15, 0.15)

    assert len(train_idx) == 7000
    assert len(val_idx) == 1500
    assert len(test_idx) == 1500

    # Strict chronological partition
    assert timestamps[train_idx[-1]] <= timestamps[val_idx[0]]
    assert timestamps[val_idx[-1]] <= timestamps[test_idx[0]]

    # Exact fraud label counts
    assert int(np.sum(labels[train_idx])) == 239
    assert int(np.sum(labels[val_idx])) == 58
    assert int(np.sum(labels[test_idx])) == 53
    assert int(np.sum(labels)) == 350

    # Ensure labels are quarantined from features
    assert graph.features.shape == (10000, 32)
    for k in range(32):
        col = graph.features[:, k]
        assert not np.array_equal(col, labels)
        corr = float(np.corrcoef(col, labels)[0, 1])
        assert abs(corr) < 0.35


def test_train_only_preprocessing_scaler_isolation():
    """
    Verify that StandardScaler is fitted strictly on X_train.
    Extreme outliers in X_val and X_test must NOT alter X_train transformation statistics.
    """
    np.random.seed(42)
    x_train = np.random.normal(loc=10.0, scale=2.0, size=(100, 4)).astype(np.float32)
    x_val = np.random.normal(loc=10.0, scale=2.0, size=(20, 4)).astype(np.float32)
    x_test = np.random.normal(loc=10.0, scale=2.0, size=(20, 4)).astype(np.float32)

    # Infiltrate extreme outlier in test set
    x_test_corrupted = x_test.copy()
    x_test_corrupted[0, 0] = 1_000_000.0

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # Test scaler transform
    x_test_scaled_clean = scaler.transform(x_test)
    x_test_scaled_corrupted = scaler.transform(x_test_corrupted)

    # Training parameters must be uninfluenced
    np.testing.assert_allclose(scaler.mean_, np.mean(x_train, axis=0), atol=1e-5)
    np.testing.assert_allclose(scaler.var_, np.var(x_train, axis=0), atol=1e-5)

    # Corrupted test sample affects only its own scaled output
    np.testing.assert_allclose(x_test_scaled_clean[1:], x_test_scaled_corrupted[1:], atol=1e-5)


def test_validation_only_threshold_selection():
    """
    Verify threshold selection functions select optimal F1 and target-recall thresholds
    strictly from validation predictions.
    """
    y_val = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.70, 0.75, 0.80, 0.85, 0.90], dtype=np.float64)

    best_th, best_f1 = select_optimal_f1_threshold(y_val, y_prob)
    # Perfectly separable: any threshold between 0.25 and 0.70 yields F1 = 1.0
    assert 0.25 < best_th <= 0.70
    assert best_f1 == pytest.approx(1.0)

    # Target recall 0.60 (needs 3 out of 5 positives detected: probs >= 0.80)
    rec_th = select_recall_target_threshold(y_val, y_prob, target_recall=0.60)
    assert rec_th >= 0.70


def test_metric_calculations_and_confusion_matrix_consistency():
    """
    Verify compute_metrics generates mathematically consistent confusion matrices,
    support, and precision/recall values.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    # Predicted probabilities
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9], dtype=np.float64)

    # Using threshold 0.5:
    # y_pred = [0, 0, 1, 1, 0, 1]
    # Licit (0): two 0s (TN=2), two 1s (FP=2)
    # Fraud (1): one 0 (FN=1), one 1 (TP=1)
    metrics = compute_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        recall_target_threshold=0.5,
        target_recall_spec=0.5,
    )

    cm = metrics["optimal_f1_evaluation"]["confusion_matrix"]
    assert cm["true_negative"] == 2
    assert cm["false_positive"] == 2
    assert cm["false_negative"] == 1
    assert cm["true_positive"] == 1

    # Precision: TP / (TP + FP) = 1 / (1 + 2) = 1/3
    assert metrics["optimal_f1_evaluation"]["precision"] == pytest.approx(1.0 / 3.0, abs=1e-5)
    # Recall: TP / (TP + FN) = 1 / (1 + 1) = 0.5
    assert metrics["optimal_f1_evaluation"]["recall"] == pytest.approx(0.5, abs=1e-5)
    # F1: 2 * (1/3 * 1/2) / (1/3 + 1/2) = (1/3) / (5/6) = 2/5 = 0.4
    assert metrics["optimal_f1_evaluation"]["f1_score"] == pytest.approx(0.4, abs=1e-5)

    # Support consistency
    assert metrics["support"]["licit"] == 4
    assert metrics["support"]["fraud"] == 2
    assert metrics["prevalence"] == pytest.approx(2.0 / 6.0, abs=1e-5)


def test_undefined_metric_handling_zero_division():
    """
    Verify that when predicted positive count is zero, precision, recall, and F1
    are returned gracefully as 0.0 without ZeroDivisionError.
    """
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_prob = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)

    # High threshold = 0.99 (zero predicted positives)
    metrics = compute_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.99,
        recall_target_threshold=0.99,
    )

    opt = metrics["optimal_f1_evaluation"]
    assert opt["predicted_positive_count"] == 0
    assert opt["precision"] == 0.0
    assert opt["recall"] == 0.0
    assert opt["f1_score"] == 0.0
    assert opt["confusion_matrix"]["true_positive"] == 0
    assert opt["confusion_matrix"]["false_positive"] == 0


def test_deterministic_repeatability(canonical_fraud_fixture):
    """
    Verify that evaluating the same model configuration with fixed seed produces
    exact bitwise identical predictions and metrics.
    """
    graph, labels, _ = canonical_fraud_fixture
    train_idx, val_idx, test_idx = temporal_train_test_split(graph.num_nodes, 0.70, 0.15, 0.15)

    res1 = evaluate_single_model(
        graph=graph,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_name="logistic_regression",
        representation=RepresentationType.RAW,
        random_seed=42,
    )
    res2 = evaluate_single_model(
        graph=graph,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_name="logistic_regression",
        representation=RepresentationType.RAW,
        random_seed=42,
    )

    assert res1["validation_metrics"]["pr_auc"] == res2["validation_metrics"]["pr_auc"]
    assert res1["test_metrics"]["pr_auc"] == res2["test_metrics"]["pr_auc"]
    assert res1["test_metrics"]["roc_auc"] == res2["test_metrics"]["roc_auc"]
    assert res1["test_metrics"]["optimal_f1_evaluation"]["threshold"] == res2["test_metrics"]["optimal_f1_evaluation"]["threshold"]


def test_evaluate_single_model_schema_and_types(canonical_fraud_fixture):
    """Verify that evaluate_single_model returns complete and well-typed output dictionaries."""
    graph, labels, _ = canonical_fraud_fixture
    train_idx, val_idx, test_idx = temporal_train_test_split(graph.num_nodes, 0.70, 0.15, 0.15)

    rec = evaluate_single_model(
        graph=graph,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_name="logistic_regression",
        representation=RepresentationType.MEAN,
        random_seed=42,
    )

    assert rec["model_name"] == "logistic_regression"
    assert rec["representation"] == "mean"
    assert rec["context_policy"] == "causal_history"
    assert rec["feature_dim"] == 64

    # Timing check
    timing = rec["timing_ms"]
    assert "feature_enrichment_ms" in timing
    assert "model_training_ms" in timing
    assert "val_inference_ms" in timing
    assert "test_inference_ms" in timing
    assert timing["total_runtime_ms"] > 0.0

    # Metric bounds
    assert 0.0 <= rec["test_metrics"]["pr_auc"] <= 1.0
    assert 0.0 <= rec["test_metrics"]["roc_auc"] <= 1.0


def test_invalid_model_name_raises_value_error(canonical_fraud_fixture):
    """Verify ValueError when requesting unsupported model."""
    graph, labels, _ = canonical_fraud_fixture
    train_idx, val_idx, test_idx = temporal_train_test_split(graph.num_nodes, 0.70, 0.15, 0.15)

    with pytest.raises(ValueError, match="Unsupported model"):
        evaluate_single_model(
            graph=graph,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            model_name="quantum_neural_net",
        )
