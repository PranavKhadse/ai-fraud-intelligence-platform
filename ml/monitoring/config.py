"""
Configuration, Constants, Severity Enums, and Threshold Policies for Phase 13 ML & Model Monitoring.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class DriftSeverity(str, Enum):
    """Categorical alert severity level for drift and degradation metrics."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class MetricConfidence(str, Enum):
    """Sample-size confidence grade for ground-truth performance monitoring."""
    NORMAL_CONFIDENCE = "NORMAL_CONFIDENCE"
    LOW_SAMPLE = "LOW_SAMPLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class MonitoringConfig:
    """
    Typed, immutable configuration for model, prediction, and feature drift monitoring.
    """
    model_version: str = "1.0.0"
    random_seed: int = 42

    # Baseline Profile Generation Parameters
    reference_sample_size: int = 5000
    num_deciles: int = 10
    laplace_epsilon: float = 1e-4

    # Operating Decision Thresholds
    default_operating_threshold: float = 0.78
    comparison_threshold: float = 0.50

    # Feature Drift Thresholds
    feature_psi_warning: float = 0.10
    feature_psi_critical: float = 0.25
    feature_ks_pvalue_warning: float = 0.01
    feature_ks_pvalue_critical: float = 0.001
    missing_rate_delta_warning: float = 0.01
    missing_rate_delta_critical: float = 0.05
    unseen_category_rate_warning: float = 0.01
    unseen_category_rate_critical: float = 0.05

    # Prediction Drift Thresholds
    prediction_psi_warning: float = 0.10
    prediction_psi_critical: float = 0.25
    tier_jsd_warning: float = 0.10
    tier_jsd_critical: float = 0.25
    action_jsd_warning: float = 0.10
    action_jsd_critical: float = 0.25

    # Performance Degradation Relative Thresholds (Higher-is-better metrics)
    higher_is_better_rel_warning: float = 0.10
    higher_is_better_rel_critical: float = 0.25

    # Performance Degradation Relative Thresholds (Lower-is-better metrics: FPR)
    lower_is_better_rel_warning: float = 0.50
    lower_is_better_rel_critical: float = 1.50
    lower_is_better_abs_fpr_warning: float = 0.0050
    lower_is_better_abs_fpr_critical: float = 0.0150

    # Minimum Sample Sizes & Confidence Boundaries
    min_feature_samples: int = 100
    min_prediction_samples: int = 50
    min_performance_samples_insufficient: int = 20
    min_performance_samples_low_confidence: int = 100

    # Artifact Storage Paths
    artifacts_dir: Path = Path("ml/monitoring/artifacts")
    train_dataset_path: Path = Path("data/processed/features/train_features.parquet")
    val_dataset_path: Path = Path("data/processed/features/val_features.parquet")
    test_dataset_path: Path = Path("data/processed/features/test_features.parquet")

    feature_profile_path: Path = Path("ml/monitoring/artifacts/baseline_feature_profile_v1.0.0.json")
    prediction_profile_path: Path = Path("ml/monitoring/artifacts/baseline_prediction_profile_v1.0.0.json")
    performance_profile_path: Path = Path("ml/monitoring/artifacts/baseline_performance_profile_v1.0.0.json")


default_monitoring_config = MonitoringConfig()
