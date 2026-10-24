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

__all__ = [
    "AggregationBackend",
    "ContextPolicy",
    "RepresentationType",
    "apply_context_policy",
    "compute_neighbor_aggregation",
    "enrich_features",
    "enrich_features_with_split",
    "get_feature_dimension",
]
