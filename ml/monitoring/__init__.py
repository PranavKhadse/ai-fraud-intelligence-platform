"""
ML & Model Monitoring Package for AI-Powered Fraud Detection Platform.

Provides:
- Drift severity & metric confidence policies.
- Baseline profile schemas and storage representations.
- Deterministic profile generator for features, predictions, and operational ground-truth benchmarks.
"""

from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
from ml.monitoring.schemas import (
    CategoricalDistributionDriftResult,
    CategoricalFeatureProfile,
    ConfusionMatrixData,
    FeatureBaselineProfile,
    FeatureDriftReport,
    MetricDegradationResult,
    NumericalFeatureProfile,
    OperationalDecisionMetrics,
    OverrideRateDriftResult,
    PerformanceBaselineProfile,
    PerformanceReport,
    PredictionBaselineProfile,
    PredictionDriftReport,
    RiskScoreBucket,
    RiskScoreDriftResult,
    ScoreDriftResult,
    ScoreHistogramBin,
    SingleFeatureDriftResult,
    ThresholdPerformanceMetrics,
    compute_content_sha256,
)
from ml.monitoring.stats import (
    calculate_jsd,
    calculate_missing_rate_delta,
    calculate_numerical_psi,
    calculate_psi,
    calculate_two_sample_ks,
    calculate_unseen_category_rate,
)
from ml.monitoring.feature_drift import FeatureDriftCalculator
from ml.monitoring.prediction_drift import PredictionDriftCalculator
from ml.monitoring.performance import ModelPerformanceCalculator

__all__ = [
    "DriftSeverity",
    "MetricConfidence",
    "MonitoringConfig",
    "default_monitoring_config",
    "NumericalFeatureProfile",
    "CategoricalFeatureProfile",
    "FeatureBaselineProfile",
    "ScoreHistogramBin",
    "RiskScoreBucket",
    "PredictionBaselineProfile",
    "ConfusionMatrixData",
    "ThresholdPerformanceMetrics",
    "OperationalDecisionMetrics",
    "PerformanceBaselineProfile",
    "SingleFeatureDriftResult",
    "FeatureDriftReport",
    "FeatureDriftCalculator",
    "ScoreDriftResult",
    "RiskScoreDriftResult",
    "CategoricalDistributionDriftResult",
    "OverrideRateDriftResult",
    "PredictionDriftReport",
    "PredictionDriftCalculator",
    "MetricDegradationResult",
    "PerformanceReport",
    "ModelPerformanceCalculator",
    "calculate_psi",
    "calculate_numerical_psi",
    "calculate_two_sample_ks",
    "calculate_jsd",
    "calculate_missing_rate_delta",
    "calculate_unseen_category_rate",
    "compute_content_sha256",
]



