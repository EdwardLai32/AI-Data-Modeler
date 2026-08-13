"""Deterministic dataset profiling.

Everything an agent later reasons about — column kinds, distributions,
correlations, leakage candidates, quality issues — is measured here by ordinary
pandas/scipy/scikit-learn code. No model is consulted, and the same frame always
produces the same profile.

The public entry point is :func:`profile_dataframe`; the supporting modules are
importable for callers that need a single measure:

*   :mod:`~automl_architect.profiling.stats` — per-column statistics and the
    association measures (Cramer's V, correlation ratio).
*   :mod:`~automl_architect.profiling.semantic` — column-kind and semantic-type
    inference.
*   :mod:`~automl_architect.profiling.leakage` — target-association scoring,
    leakage findings, and data-quality detection.
"""

from __future__ import annotations

from .leakage import (
    TargetAssociation,
    detect_leakage,
    detect_quality_issues,
    score_target_associations,
    target_is_classification_like,
)
from .profiler import dataset_fingerprint, profile_dataframe, summarise_profile
from .semantic import (
    ColumnInference,
    detect_semantic_type,
    infer_column,
    name_tokens,
    suspicious_outcome_name,
    try_parse_datetime,
)
from .stats import (
    FrameOverview,
    basic_counts,
    build_target_summary,
    correlation_ratio,
    cramers_v,
    dataframe_overview,
    datetime_summary,
    numeric_summary,
    outlier_summary,
    quantile_summary,
    string_summary,
    top_value_counts,
)

__all__ = [
    "ColumnInference",
    "FrameOverview",
    "TargetAssociation",
    "basic_counts",
    "build_target_summary",
    "correlation_ratio",
    "cramers_v",
    "dataframe_overview",
    "dataset_fingerprint",
    "datetime_summary",
    "detect_leakage",
    "detect_quality_issues",
    "detect_semantic_type",
    "infer_column",
    "name_tokens",
    "numeric_summary",
    "outlier_summary",
    "profile_dataframe",
    "quantile_summary",
    "score_target_associations",
    "string_summary",
    "summarise_profile",
    "suspicious_outcome_name",
    "target_is_classification_like",
    "top_value_counts",
    "try_parse_datetime",
]
