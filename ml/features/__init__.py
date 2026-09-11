"""
Behavioral Feature Engineering Package for AI-Powered Fraud Detection & Risk Intelligence Platform.
"""

from ml.features.config import (
    CANONICAL_COLUMNS,
    TEMPORAL_FEATURES,
    VELOCITY_FEATURES,
    SPENDING_FEATURES,
    DEVIATION_FEATURES,
    ACCOUNT_HISTORY_FEATURES,
    MERCHANT_INTERACTION_FEATURES,
    GEOGRAPHIC_FEATURES,
    ENGINEERED_FEATURE_COLUMNS,
    FULL_FEATURE_DATASET_COLUMNS,
    FEATURE_GROUPS,
    WINDOW_SECONDS,
    NIGHT_HOURS,
    IMPOSSIBLE_SPEED_THRESHOLD_KMH,
)
from ml.features.temporal import extract_temporal_features
from ml.features.velocity import extract_velocity_features
from ml.features.spending import extract_spending_features
from ml.features.deviation import extract_deviation_features
from ml.features.account import extract_account_features
from ml.features.merchant import extract_merchant_features
from ml.features.geographic import extract_geographic_features, haversine_distance_km
from ml.features.pipeline import (
    extract_all_engineered_features,
    engineer_features_for_partition,
    build_cross_partition_features,
)
from ml.features.validation import (
    validate_feature_schema_and_finiteness,
    audit_point_in_time_leakage,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "TEMPORAL_FEATURES",
    "VELOCITY_FEATURES",
    "SPENDING_FEATURES",
    "DEVIATION_FEATURES",
    "ACCOUNT_HISTORY_FEATURES",
    "MERCHANT_INTERACTION_FEATURES",
    "GEOGRAPHIC_FEATURES",
    "ENGINEERED_FEATURE_COLUMNS",
    "FULL_FEATURE_DATASET_COLUMNS",
    "FEATURE_GROUPS",
    "WINDOW_SECONDS",
    "NIGHT_HOURS",
    "IMPOSSIBLE_SPEED_THRESHOLD_KMH",
    "extract_temporal_features",
    "extract_velocity_features",
    "extract_spending_features",
    "extract_deviation_features",
    "extract_account_features",
    "extract_merchant_features",
    "extract_geographic_features",
    "haversine_distance_km",
    "extract_all_engineered_features",
    "engineer_features_for_partition",
    "build_cross_partition_features",
    "validate_feature_schema_and_finiteness",
    "audit_point_in_time_leakage",
]
