"""
Controlled Fraud Model Evaluation Engine
Module: FGI-MOD-07 (Fraud Scoring & Classifier Baseline)
Course: VIT Vellore — CSI4007 GPU Programming

Executes rigorous, leakage-proof comparative evaluations between baseline tabular features
and graph-enriched relational representations across Logistic Regression and Random Forest models.

Experimental Controls:
1. Chronological splitting: Train [0:7000), Val [7000:8500), Test [8500:10000).
2. Train-only preprocessing: StandardScaler fitted strictly on X_train (strict train-only preprocessing isolation).
3. Validation-only threshold selection: Optimal F1 and Recall-target thresholds chosen on Val,
   then frozen prior to Test set evaluation.
4. Separate context policies: Evaluates 'causal_history' (primary online streaming) and
   'train_only_context' (strict inductive store) as clean, unmixed cohorts.
5. Strict label quarantine: Ground-truth labels are never passed to graph aggregators or features.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Ensure project root is in sys.path
_root_dir = str(Path(__file__).resolve().parents[2])
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from backend.cpu_engine.csr_graph import CSRGraph
from backend.cpu_engine.synthetic_fraud_generator import load_labeled_graph, temporal_train_test_split
from backend.ml_engine.feature_enrichment import (
    AggregationBackend,
    ContextPolicy,
    RepresentationType,
    enrich_features_with_split,
    get_feature_dimension,
)


def select_optimal_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_thresholds: int = 100,
) -> tuple[float, float]:
    """
    Search validation set for the decision threshold maximizing binary minority F1-score.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_prob: Predicted probability of positive class in [0.0, 1.0].
        num_thresholds: Number of uniform threshold candidates.

    Returns:
        Tuple of (best_threshold, best_f1).
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_th = 0.5
    best_f1 = -1.0

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0.0
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)

    return best_th, best_f1


def select_recall_target_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.50,
    num_thresholds: int = 100,
) -> float:
    """
    Search validation set for the highest decision threshold achieving recall >= target_recall.
    If no candidate achieves target_recall, selects the threshold achieving maximum recall.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_prob: Predicted probability of positive class in [0.0, 1.0].
        target_recall: Minimum desired true positive rate.
        num_thresholds: Number of uniform threshold candidates.

    Returns:
        Selected frozen threshold (float).
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    eligible_thresholds = []

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        _, rec, _, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0.0
        )
        if rec >= target_recall:
            eligible_thresholds.append(float(th))

    if eligible_thresholds:
        # Highest threshold maintaining desired recall (minimizes false alarms)
        return float(np.max(eligible_thresholds))

    # Fallback if target recall unachieved: pick threshold with highest recall
    best_rec = -1.0
    fallback_th = 0.5
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        _, rec, _, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0.0
        )
        if rec > best_rec:
            best_rec = float(rec)
            fallback_th = float(th)

    return fallback_th


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    recall_target_threshold: float,
    target_recall_spec: float = 0.50,
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud classification metrics for a frozen prediction probability array.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Positive class probabilities.
        threshold: Frozen decision threshold (typically chosen on validation).
        recall_target_threshold: Frozen threshold targeting predefined recall.
        target_recall_spec: Predefined target recall value.

    Returns:
        Structured dictionary containing PR-AUC, ROC-AUC, threshold metrics, and confusion matrices.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    total_samples = len(y_true_arr)
    fraud_count = int(np.sum(y_true_arr))
    licit_count = total_samples - fraud_count
    prevalence = float(fraud_count / total_samples) if total_samples > 0 else 0.0

    # Curve-level metrics
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    # Optimal F1 threshold metrics
    preds_f1 = (y_prob_arr >= threshold).astype(int)
    p_f1, r_f1, f1_val, _ = precision_recall_fscore_support(
        y_true_arr, preds_f1, average="binary", zero_division=0.0
    )
    cm_f1 = confusion_matrix(y_true_arr, preds_f1, labels=[0, 1]).tolist()
    pred_pos_f1 = int(np.sum(preds_f1))

    # Target recall threshold metrics
    preds_rec = (y_prob_arr >= recall_target_threshold).astype(int)
    p_rec, r_rec, f1_rec, _ = precision_recall_fscore_support(
        y_true_arr, preds_rec, average="binary", zero_division=0.0
    )
    cm_rec = confusion_matrix(y_true_arr, preds_rec, labels=[0, 1]).tolist()
    pred_pos_rec = int(np.sum(preds_rec))

    return {
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "prevalence": round(prevalence, 6),
        "support": {"licit": licit_count, "fraud": fraud_count},
        "optimal_f1_evaluation": {
            "threshold": round(float(threshold), 4),
            "precision": round(float(p_f1), 6),
            "recall": round(float(r_f1), 6),
            "f1_score": round(float(f1_val), 6),
            "predicted_positive_count": pred_pos_f1,
            "confusion_matrix": {
                "true_negative": int(cm_f1[0][0]),
                "false_positive": int(cm_f1[0][1]),
                "false_negative": int(cm_f1[1][0]),
                "true_positive": int(cm_f1[1][1]),
            },
        },
        "target_recall_evaluation": {
            "target_recall_specified": target_recall_spec,
            "threshold": round(float(recall_target_threshold), 4),
            "achieved_recall": round(float(r_rec), 6),
            "precision_at_target_recall": round(float(p_rec), 6),
            "f1_score": round(float(f1_rec), 6),
            "predicted_positive_count": pred_pos_rec,
            "confusion_matrix": {
                "true_negative": int(cm_rec[0][0]),
                "false_positive": int(cm_rec[0][1]),
                "false_negative": int(cm_rec[1][0]),
                "true_positive": int(cm_rec[1][1]),
            },
        },
    }


def evaluate_single_model(
    graph: CSRGraph,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: Literal["logistic_regression", "random_forest"] = "logistic_regression",
    representation: str | RepresentationType = RepresentationType.RAW,
    context_policy: str | ContextPolicy = ContextPolicy.CAUSAL_HISTORY,
    backend: str | AggregationBackend = AggregationBackend.CPU_VECTORIZED,
    random_seed: int = 42,
    target_recall: float = 0.50,
) -> Dict[str, Any]:
    """
    Execute a single end-to-end model evaluation run with rigorous leakage controls.

    Workflow:
    1. Feature enrichment under specified context policy.
    2. Train-only preprocessing: StandardScaler fitted strictly on X_train.
    3. Model training on (X_train, y_train).
    4. Probability prediction on validation and test partitions.
    5. Validation-only threshold selection (freezing optimal F1 and target recall thresholds).
    6. Performance metric evaluation and timing recording.

    Returns:
        Structured experiment record dictionary.
    """
    model_clean = model_name.strip().lower()
    if model_clean not in ("logistic_regression", "random_forest"):
        raise ValueError(f"Unsupported model '{model_name}'. Supported: ['logistic_regression', 'random_forest']")

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    # 1. Feature Enrichment Timing
    t0_enrich = time.perf_counter()
    x_train, x_val, x_test = enrich_features_with_split(
        graph=graph,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        representation=representation,
        backend=backend,
        context_policy=context_policy,
    )
    t_enrich_sec = time.perf_counter() - t0_enrich

    # 2. Train-only Preprocessing: StandardScaler fitted on X_train only
    t0_prep = time.perf_counter()
    if model_clean == "logistic_regression":
        scaler = StandardScaler()
        x_train_fit = scaler.fit_transform(x_train)
        x_val_trans = scaler.transform(x_val)
        x_test_trans = scaler.transform(x_test)
    else:
        # Tree-based Random Forest is invariant to monotonic scaling; pass float32 contiguous arrays
        x_train_fit = np.ascontiguousarray(x_train, dtype=np.float32)
        x_val_trans = np.ascontiguousarray(x_val, dtype=np.float32)
        x_test_trans = np.ascontiguousarray(x_test, dtype=np.float32)
    t_prep_sec = time.perf_counter() - t0_prep

    # 3. Model Training
    t0_train = time.perf_counter()
    if model_clean == "logistic_regression":
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_seed,
            solver="lbfgs",
        )
    elif model_clean == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_seed,
            n_jobs=-1,
        )
    model.fit(x_train_fit, y_train)
    t_train_sec = time.perf_counter() - t0_train

    # 4. Inference Timing (Separated Val and Test)
    t0_val_infer = time.perf_counter()
    val_probs = model.predict_proba(x_val_trans)[:, 1]
    t_val_infer_sec = time.perf_counter() - t0_val_infer

    t0_test_infer = time.perf_counter()
    test_probs = model.predict_proba(x_test_trans)[:, 1]
    t_test_infer_sec = time.perf_counter() - t0_test_infer

    # 5. Validation-Only Threshold Selection (Frozen for Test Evaluation)
    opt_f1_th, val_f1_score = select_optimal_f1_threshold(y_val, val_probs)
    rec_target_th = select_recall_target_threshold(y_val, val_probs, target_recall=target_recall)

    # 6. Metric Computation
    val_metrics = compute_metrics(
        y_true=y_val,
        y_prob=val_probs,
        threshold=opt_f1_th,
        recall_target_threshold=rec_target_th,
        target_recall_spec=target_recall,
    )
    test_metrics = compute_metrics(
        y_true=y_test,
        y_prob=test_probs,
        threshold=opt_f1_th,
        recall_target_threshold=rec_target_th,
        target_recall_spec=target_recall,
    )

    return {
        "model_name": model_clean,
        "representation": str(representation.value if isinstance(representation, RepresentationType) else representation),
        "context_policy": str(context_policy.value if isinstance(context_policy, ContextPolicy) else context_policy),
        "feature_dim": int(x_train.shape[1]),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "timing_ms": {
            "feature_enrichment_ms": round(t_enrich_sec * 1000.0, 3),
            "preprocessing_ms": round(t_prep_sec * 1000.0, 3),
            "model_training_ms": round(t_train_sec * 1000.0, 3),
            "val_inference_ms": round(t_val_infer_sec * 1000.0, 3),
            "test_inference_ms": round(t_test_infer_sec * 1000.0, 3),
            "total_inference_ms": round((t_val_infer_sec + t_test_infer_sec) * 1000.0, 3),
            "total_runtime_ms": round((t_enrich_sec + t_prep_sec + t_train_sec + t_val_infer_sec + t_test_infer_sec) * 1000.0, 3),
        },
        "model_parameters": model.get_params(),
    }


def run_comprehensive_evaluation(
    fixture_path: str | Path = "data/fixtures/synthetic_fraud_10k.npz",
    output_dir: str | Path = "experiments/results",
    random_seed: int = 42,
    target_recall: float = 0.50,
) -> Dict[str, Any]:
    """
    Execute full comparative evaluation across models, representations, and context policies.

    Saves machine-readable JSON results to output_dir and returns structured report dict.
    """
    fixture = Path(fixture_path)
    if not fixture.exists():
        raise FileNotFoundError(f"Fixture not found at: {fixture}")

    fixture_bytes = fixture.read_bytes()
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()

    graph, labels, timestamps = load_labeled_graph(fixture)
    train_idx, val_idx, test_idx = temporal_train_test_split(graph.num_nodes, 0.70, 0.15, 0.15)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    experiment_id = f"EXP-FRAUD-EVAL-7.4-{timestamp_str}"

    models = ["logistic_regression", "random_forest"]
    primary_reps = [
        RepresentationType.RAW,
        RepresentationType.MEAN,
        RepresentationType.MAX,
        RepresentationType.MULTI_OPERATOR,
        RepresentationType.SUM,
    ]

    results: List[Dict[str, Any]] = []

    # 1. Primary Evaluation Cohort: Causal History Context Policy
    for model_name in models:
        for rep in primary_reps:
            rec = evaluate_single_model(
                graph=graph,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                model_name=model_name,
                representation=rep,
                context_policy=ContextPolicy.CAUSAL_HISTORY,
                backend=AggregationBackend.CPU_VECTORIZED,
                random_seed=random_seed,
                target_recall=target_recall,
            )
            results.append(rec)

    # 2. Sensitivity Analysis Cohort: Train-Only Context Policy
    sensitivity_results: List[Dict[str, Any]] = []
    for model_name in models:
        for rep in primary_reps:
            rec = evaluate_single_model(
                graph=graph,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                model_name=model_name,
                representation=rep,
                context_policy=ContextPolicy.TRAIN_ONLY_CONTEXT,
                backend=AggregationBackend.CPU_VECTORIZED,
                random_seed=random_seed,
                target_recall=target_recall,
            )
            sensitivity_results.append(rec)

    # Manifest and environmental metadata
    report: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "phase": "Phase 7.4 (Controlled Fraud Model Evaluation)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "provenance": {
            "dataset_name": "synthetic_fraud_10k",
            "dataset_provenance": "SYNTHETIC",
            "dataset_path": str(fixture),
            "dataset_sha256": fixture_sha256,
            "disclaimer": (
                "Controlled synthetic benchmark fixture for evaluating graph-based neighbor aggregation. "
                "The 3.5% fraud prevalence reflects real-world class imbalance, but does NOT represent "
                "live banking fraud distributions or empirical records."
            ),
        },
        "environment": {
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "dataset_split": {
            "train_nodes": len(train_idx),
            "train_frauds": int(np.sum(labels[train_idx])),
            "train_prevalence": round(float(np.mean(labels[train_idx])), 6),
            "val_nodes": len(val_idx),
            "val_frauds": int(np.sum(labels[val_idx])),
            "val_prevalence": round(float(np.mean(labels[val_idx])), 6),
            "test_nodes": len(test_idx),
            "test_frauds": int(np.sum(labels[test_idx])),
            "test_prevalence": round(float(np.mean(labels[test_idx])), 6),
            "total_nodes": graph.num_nodes,
            "total_edges": graph.num_edges,
            "total_frauds": int(np.sum(labels)),
        },
        "evaluation_protocol": {
            "primary_context_policy": "causal_history",
            "sensitivity_context_policy": "train_only_context",
            "target_recall_spec": target_recall,
            "threshold_selection_protocol": (
                "Validation-only grid search over 100 thresholds between 0.01 and 0.99. "
                "Optimal F1 threshold chosen to maximize binary F1 on validation predictions. "
                "Target recall threshold chosen as highest threshold with validation recall >= target_recall. "
                "Thresholds are frozen prior to test set evaluation."
            ),
            "preprocessing_protocol": (
                "StandardScaler fitted strictly on training partition features (X_train). "
                "Validation and test sets transformed using training statistics. "
                "Strict train-only preprocessing isolation (zero scaler snooping)."
            ),
        },
        "primary_results_causal_history": results,
        "sensitivity_results_train_only": sensitivity_results,
    }

    # Save to JSON
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_json_path = out_dir_path / f"fraud_model_evaluation_{timestamp_str}.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report["output_json_path"] = str(out_json_path)
    return report


if __name__ == "__main__":
    print("[+] Executing FGI Phase 7.4 Comprehensive Fraud Model Evaluation...")
    rep = run_comprehensive_evaluation()
    print(f"[OK] Evaluation complete! Artifact written to: {rep['output_json_path']}")
    print("\n--- Summary Table (Primary Cohort: Causal History) ---")
    header = f"{'Model':<20} | {'Rep':<15} | {'Dim':<4} | {'Val PR':<7} | {'Test PR':<7} | {'Test ROC':<8} | {'Test F1':<7} | {'F1 Th':<6} | {'Test Rec':<8} | {'Test Prec':<9}"
    print(header)
    print("-" * len(header))
    for r in rep["primary_results_causal_history"]:
        m = r["model_name"]
        rp = r["representation"]
        d = r["feature_dim"]
        v_pr = r["validation_metrics"]["pr_auc"]
        t_pr = r["test_metrics"]["pr_auc"]
        t_roc = r["test_metrics"]["roc_auc"]
        t_f1 = r["test_metrics"]["optimal_f1_evaluation"]["f1_score"]
        th = r["test_metrics"]["optimal_f1_evaluation"]["threshold"]
        t_rec = r["test_metrics"]["optimal_f1_evaluation"]["recall"]
        t_prec = r["test_metrics"]["optimal_f1_evaluation"]["precision"]
        print(f"{m:<20} | {rp:<15} | {d:<4} | {v_pr:<7.4f} | {t_pr:<7.4f} | {t_roc:<8.4f} | {t_f1:<7.4f} | {th:<6.2f} | {t_rec:<8.4f} | {t_prec:<9.4f}")
