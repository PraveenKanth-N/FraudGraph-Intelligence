"""
Machine Learning Engine — FraudGraph Intelligence (FGI)
Module: FGI-MOD-07 (Fraud Scoring & Classifier Baseline)
Course: VIT Vellore — CSI4007 GPU Programming
"""

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
from backend.ml_engine.model_evaluation import (
    compute_metrics,
    evaluate_single_model,
    run_comprehensive_evaluation,
    select_optimal_f1_threshold,
    select_recall_target_threshold,
)
from backend.ml_engine.reporting import (
    generate_manifest_and_metadata,
    generate_summary_csv,
    run_full_reporting_pipeline,
    validate_evaluation_artifact,
)

__all__ = [
    "AggregationBackend",
    "ContextPolicy",
    "RepresentationType",
    "apply_context_policy",
    "compute_metrics",
    "compute_neighbor_aggregation",
    "enrich_features",
    "enrich_features_with_split",
    "evaluate_single_model",
    "generate_manifest_and_metadata",
    "generate_summary_csv",
    "get_feature_dimension",
    "run_comprehensive_evaluation",
    "run_full_reporting_pipeline",
    "select_optimal_f1_threshold",
    "select_recall_target_threshold",
    "validate_evaluation_artifact",
]
