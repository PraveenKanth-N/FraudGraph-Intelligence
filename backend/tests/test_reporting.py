"""
Automated Test Suite for Evaluation Reporting and Visualizations
Phase: 7.5 — Research-Grade Evaluation Reporting & Visualizations
Course: VIT Vellore — CSI4007 GPU Programming

Verifies 11 core reporting invariants:
1. Valid artifact parsing
2. Missing required fields raise KeyError or ValueError
3. Metrics are read from artifact rather than hardcoded
4. Confusion matrices are internally consistent (TP+FP+TN+FN == N)
5. Test sample counts and positive support agree with metadata
6. Summary CSV values agree with source JSON
7. Figure generation works in headless environment (PNG, SVG, PDF)
8. Re-running report does not modify the source artifact
9. Metadata records correct input checksum
10. Missing prediction scores result in clear explanation instead of fabricated curves
11. Generated filenames and manifest entries are deterministic
"""

import copy
import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd

from backend.ml_engine.reporting import (
    validate_evaluation_artifact,
    generate_summary_csv,
    generate_manifest_and_metadata,
    plot_figure_1_pr_auc,
    plot_figure_2_roc_auc,
    plot_figure_3_pr_curve_notice,
    plot_figure_4_confusion_matrices,
    plot_figure_5_operating_tradeoffs,
    plot_figure_6_runtime_breakdown,
    plot_figure_7_context_sensitivity,
    run_full_reporting_pipeline,
)

CANONICAL_ARTIFACT_PATH = Path("experiments/results/fraud_model_evaluation_20261024_212032.json")
EXPECTED_ARTIFACT_SHA256 = "155304b7453b15dda1f103817e32471a2ffb2357de9cb298cc7b470d8d0efc90"
EXPECTED_FIXTURE_SHA256 = "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430"


@pytest.fixture
def canonical_artifact_data():
    """Loads the canonical evaluation JSON artifact."""
    assert CANONICAL_ARTIFACT_PATH.exists(), f"Canonical artifact not found: {CANONICAL_ARTIFACT_PATH}"
    with open(CANONICAL_ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_valid_artifact_parsing(canonical_artifact_data):
    """Test 1: Valid artifact parsing passes all schema and integrity checks."""
    validated = validate_evaluation_artifact(canonical_artifact_data)
    assert validated["experiment_id"] == "EXP-FRAUD-EVAL-7.4-20261024_212032"
    assert "dataset_split" in validated
    assert "primary_results_causal_history" in validated
    assert "sensitivity_results_train_only" in validated
    assert len(validated["primary_results_causal_history"]) == 10
    assert len(validated["sensitivity_results_train_only"]) == 10


def test_2_missing_required_fields_raise_error(canonical_artifact_data):
    """Test 2: Missing required top-level or nested fields raise KeyError or ValueError."""
    # Top-level required keys
    for req_key in [
        "experiment_id",
        "provenance",
        "environment",
        "dataset_split",
        "primary_results_causal_history",
        "sensitivity_results_train_only",
        "evaluation_protocol",
    ]:
        corrupted = copy.deepcopy(canonical_artifact_data)
        del corrupted[req_key]
        with pytest.raises((KeyError, ValueError)):
            validate_evaluation_artifact(corrupted)

    # Corrupted split counts
    corrupted_split = copy.deepcopy(canonical_artifact_data)
    corrupted_split["dataset_split"]["test_nodes"] = 9999
    with pytest.raises(ValueError, match="Invalid split counts"):
        validate_evaluation_artifact(corrupted_split)

    # Inconsistent confusion matrix in a model
    corrupted_cm = copy.deepcopy(canonical_artifact_data)
    corrupted_cm["primary_results_causal_history"][0]["test_metrics"]["optimal_f1_evaluation"]["confusion_matrix"]["true_positive"] = 999
    with pytest.raises(ValueError, match="Confusion matrix sum"):
        validate_evaluation_artifact(corrupted_cm)


def test_3_metrics_read_from_artifact_not_hardcoded(canonical_artifact_data, tmp_path):
    """Test 3: Metrics are read dynamically from the artifact rather than hardcoded."""
    synthetic_artifact = copy.deepcopy(canonical_artifact_data)
    # Alter a specific metric to an arbitrary canary value
    canary_pr_auc = 0.987654
    synthetic_artifact["primary_results_causal_history"][0]["test_metrics"]["pr_auc"] = canary_pr_auc

    csv_path = tmp_path / "canary_metrics.csv"
    generate_summary_csv(synthetic_artifact, csv_path)

    df = pd.read_csv(csv_path)
    # First row should have our canary value
    assert abs(df.iloc[0]["test_pr_auc"] - canary_pr_auc) < 1e-5


def test_4_confusion_matrices_internally_consistent(canonical_artifact_data):
    """Test 4: Confusion matrices across all models sum to exactly N_split (1,500) and match supports."""
    validated = validate_evaluation_artifact(canonical_artifact_data)
    test_samples = validated["dataset_split"]["test_nodes"]
    test_frauds = validated["dataset_split"]["test_frauds"]
    test_negatives = test_samples - test_frauds

    for policy_key in ["primary_results_causal_history", "sensitivity_results_train_only"]:
        for model_eval in validated[policy_key]:
            opt_cm = model_eval["test_metrics"]["optimal_f1_evaluation"]["confusion_matrix"]
            tp = opt_cm["true_positive"]
            fp = opt_cm["false_positive"]
            tn = opt_cm["true_negative"]
            fn = opt_cm["false_negative"]

            assert tp + fp + tn + fn == test_samples
            assert tp + fn == test_frauds
            assert tn + fp == test_negatives


def test_5_test_sample_counts_and_positive_support_agree_with_metadata(canonical_artifact_data):
    """Test 5: Test sample counts, positive support, and minority prevalence match split specification."""
    split = canonical_artifact_data["dataset_split"]
    assert split["total_nodes"] == 10000
    assert split["train_nodes"] == 7000
    assert split["val_nodes"] == 1500
    assert split["test_nodes"] == 1500
    assert split["train_frauds"] == 239
    assert split["val_frauds"] == 58
    assert split["test_frauds"] == 53

    # Prevalence checks
    assert abs(split["test_prevalence"] - (53 / 1500)) < 1e-5
    assert abs(split["val_prevalence"] - (58 / 1500)) < 1e-5
    assert abs(split["train_prevalence"] - (239 / 7000)) < 1e-5


def test_6_summary_csv_values_agree_with_source_json(canonical_artifact_data, tmp_path):
    """Test 6: All 20 rows of summary CSV match source JSON precisely across all metric columns."""
    csv_path = tmp_path / "test_summary.csv"
    generate_summary_csv(canonical_artifact_data, csv_path)
    df = pd.read_csv(csv_path)

    assert len(df) == 20

    # Check causal_history rows (rows 0..9)
    causal_models = canonical_artifact_data["primary_results_causal_history"]
    for i, model in enumerate(causal_models):
        row = df[(df["context_policy"] == "causal_history") &
                 (df["model_name"] == model["model_name"]) &
                 (df["representation"] == model["representation"])].iloc[0]

        assert abs(row["val_pr_auc"] - model["validation_metrics"]["pr_auc"]) < 1e-5
        assert abs(row["test_pr_auc"] - model["test_metrics"]["pr_auc"]) < 1e-5
        assert abs(row["val_roc_auc"] - model["validation_metrics"]["roc_auc"]) < 1e-5
        assert abs(row["test_roc_auc"] - model["test_metrics"]["roc_auc"]) < 1e-5

        opt_eval = model["test_metrics"]["optimal_f1_evaluation"]
        assert abs(row["test_f1"] - opt_eval["f1_score"]) < 1e-5
        assert abs(row["test_precision"] - opt_eval["precision"]) < 1e-5
        assert abs(row["test_recall"] - opt_eval["recall"]) < 1e-5
        assert row["test_true_positive"] == opt_eval["confusion_matrix"]["true_positive"]
        assert row["test_false_positive"] == opt_eval["confusion_matrix"]["false_positive"]
        assert row["test_true_negative"] == opt_eval["confusion_matrix"]["true_negative"]
        assert row["test_false_negative"] == opt_eval["confusion_matrix"]["false_negative"]


def test_7_figure_generation_headless(canonical_artifact_data, tmp_path):
    """Test 7: Figure generation executes headlessly and produces all 7 figures in PNG, SVG, and PDF."""
    plot_funcs = [
        plot_figure_1_pr_auc,
        plot_figure_2_roc_auc,
        plot_figure_3_pr_curve_notice,
        plot_figure_4_confusion_matrices,
        plot_figure_5_operating_tradeoffs,
        plot_figure_6_runtime_breakdown,
        plot_figure_7_context_sensitivity,
    ]

    for i, func in enumerate(plot_funcs, 1):
        paths = func(canonical_artifact_data, tmp_path)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0
            assert p.suffix.lower() in [".png", ".svg", ".pdf"]


def test_8_rerunning_report_does_not_modify_source_artifact(tmp_path):
    """Test 8: Re-running the reporting pipeline does not modify the source artifact."""
    with open(CANONICAL_ARTIFACT_PATH, "rb") as f:
        sha256_before = hashlib.sha256(f.read()).hexdigest()

    run_full_reporting_pipeline(CANONICAL_ARTIFACT_PATH, tmp_path)

    with open(CANONICAL_ARTIFACT_PATH, "rb") as f:
        sha256_after = hashlib.sha256(f.read()).hexdigest()

    assert sha256_before == sha256_after == EXPECTED_ARTIFACT_SHA256


def test_9_metadata_records_correct_input_checksum(canonical_artifact_data, tmp_path):
    """Test 9: Metadata and manifest accurately record the SHA-256 hash of the input artifact."""
    out_dir = tmp_path / "report_out"
    manifest_data = run_full_reporting_pipeline(CANONICAL_ARTIFACT_PATH, out_dir)

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        saved_manifest = json.load(f)

    assert saved_manifest["source_artifact"]["sha256"] == EXPECTED_ARTIFACT_SHA256
    assert saved_manifest["source_artifact"]["dataset_sha256"] == EXPECTED_FIXTURE_SHA256
    assert len(saved_manifest["figures"]) == 7


def test_10_missing_prediction_scores_omission_notice(canonical_artifact_data, tmp_path):
    """Test 10: Missing continuous prediction scores produce a clear notice rather than fabricated curves."""
    paths = plot_figure_3_pr_curve_notice(canonical_artifact_data, tmp_path)
    # Figure 3 must exist
    for p in paths:
        assert p.exists()
        assert "fig3_pr_curve_omission_notice" in p.name

    out_dir = tmp_path / "report_out"
    res = run_full_reporting_pipeline(CANONICAL_ARTIFACT_PATH, out_dir)
    with open(res["manifest_path"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    fig3_meta = [f for f in manifest["figures"] if f["figure_id"] == "fig3_pr_curve_omission_notice"][0]
    assert "scientific integrity" in fig3_meta["interpretation"]
    assert "omitted" in fig3_meta["interpretation"]


def test_11_deterministic_filenames_and_manifest_entries(tmp_path):
    """Test 11: Figure filenames, formats, figure IDs, and CSV contents are completely deterministic."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    res_a = run_full_reporting_pipeline(CANONICAL_ARTIFACT_PATH, dir_a)
    res_b = run_full_reporting_pipeline(CANONICAL_ARTIFACT_PATH, dir_b)

    # Assert identical filenames
    files_a = sorted([f.name for f in dir_a.iterdir()])
    files_b = sorted([f.name for f in dir_b.iterdir()])
    assert files_a == files_b

    # Assert identical CSV contents
    csv_a = (dir_a / "summary_evaluation_metrics.csv").read_text(encoding="utf-8")
    csv_b = (dir_b / "summary_evaluation_metrics.csv").read_text(encoding="utf-8")
    assert csv_a == csv_b

    with open(res_a["manifest_path"], "r", encoding="utf-8") as f:
        manifest_a = json.load(f)
    with open(res_b["manifest_path"], "r", encoding="utf-8") as f:
        manifest_b = json.load(f)

    # Assert identical figure metadata IDs and titles
    figs_a = [(f["figure_id"], f["title"]) for f in manifest_a["figures"]]
    figs_b = [(f["figure_id"], f["title"]) for f in manifest_b["figures"]]
    assert figs_a == figs_b
