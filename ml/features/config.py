"""
Configuration, Constants, Schema Definitions, and Column Catalogs for Phase 3 Feature Engineering.
"""

from typing import List, Dict, Any

# 15 Canonical Columns from Phase 1
CANONICAL_COLUMNS: List[str] = [
    "transaction_id",
    "account_id",
    "timestamp",
    "unix_time",
    "amount",
    "currency",
    "merchant_id",
    "merchant_category",
    "cardholder_lat",
    "cardholder_long",
    "merchant_lat",
    "merchant_long",
    "city_pop",
    "job_category",
    "is_fraud",
]

# Group 1: Temporal Features (11 features)
TEMPORAL_FEATURES: List[str] = [
    "transaction_hour",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "is_night",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

# Group 2: Velocity Features (7 features)
VELOCITY_FEATURES: List[str] = [
    "txn_count_1h",
    "txn_count_6h",
    "txn_count_24h",
    "txn_count_7d",
    "txn_count_30d",
    "time_since_prev_txn_seconds",
    "is_first_account_txn",
]

# Group 3: Spending Features (8 features)
SPENDING_FEATURES: List[str] = [
    "amt_sum_1h",
    "amt_sum_24h",
    "amt_sum_7d",
    "amt_sum_30d",
    "amt_mean_24h",
    "amt_mean_7d",
    "amt_max_24h",
    "amt_median_30d",
]

# Group 4: Spending Deviation Features (5 features)
DEVIATION_FEATURES: List[str] = [
    "historical_amount_mean",
    "historical_amount_std",
    "historical_amount_median",
    "amount_zscore",
    "amount_ratio_to_historical_mean",
]

# Group 5: Account History Features (6 features)
ACCOUNT_HISTORY_FEATURES: List[str] = [
    "account_txn_count_before",
    "account_total_spend_before",
    "account_avg_amount_before",
    "account_max_amount_before",
    "account_unique_merchant_count_before",
    "account_unique_category_count_before",
]

# Group 6: Merchant / Category Interaction Features (6 features)
MERCHANT_INTERACTION_FEATURES: List[str] = [
    "account_merchant_txn_count_before",
    "account_category_txn_count_before",
    "account_merchant_spend_before",
    "account_category_spend_before",
    "merchant_txn_count_before",
    "category_txn_count_before",
]

# Group 7: Geographic / Travel Features (4 features)
GEOGRAPHIC_FEATURES: List[str] = [
    "cardholder_merchant_distance_km",
    "distance_from_prev_merchant_km",
    "implied_travel_speed_kmh",
    "is_impossible_travel_speed",
]

# All 47 Engineered Features in exact deterministic order
ENGINEERED_FEATURE_COLUMNS: List[str] = (
    TEMPORAL_FEATURES
    + VELOCITY_FEATURES
    + SPENDING_FEATURES
    + DEVIATION_FEATURES
    + ACCOUNT_HISTORY_FEATURES
    + MERCHANT_INTERACTION_FEATURES
    + GEOGRAPHIC_FEATURES
)

# Total 62 Output Columns (15 canonical + 47 engineered)
FULL_FEATURE_DATASET_COLUMNS: List[str] = (
    CANONICAL_COLUMNS + ENGINEERED_FEATURE_COLUMNS
)

# Feature Groups Mapping (Exactly 7 groups)
FEATURE_GROUPS: Dict[str, List[str]] = {
    "Temporal": TEMPORAL_FEATURES,
    "Velocity": VELOCITY_FEATURES,
    "Spending": SPENDING_FEATURES,
    "Spending Deviation": DEVIATION_FEATURES,
    "Account History": ACCOUNT_HISTORY_FEATURES,
    "Merchant / Category Interaction": MERCHANT_INTERACTION_FEATURES,
    "Geographic / Travel": GEOGRAPHIC_FEATURES,
}

# Rolling Window Definitions in seconds
WINDOW_SECONDS: Dict[str, int] = {
    "1h": 3600,
    "6h": 21600,
    "24h": 86400,
    "7d": 604800,
    "30d": 2592000,
}

# Empirical Constants & Thresholds from Phase 2 EDA
NIGHT_HOURS: List[int] = [22, 23, 0, 1, 2, 3, 4]
IMPOSSIBLE_SPEED_THRESHOLD_KMH: float = 800.0
EARTH_RADIUS_KM: float = 6371.0

# Cold Start Policy Defaults
COLD_START_DEFAULTS: Dict[str, Any] = {
    "count": 0,
    "sum": 0.0,
    "mean": 0.0,
    "std": 0.0,
    "median": 0.0,
    "zscore": 0.0,
    "ratio_to_mean": 1.0,
    "time_delta": 0.0,
    "is_first_txn": 1,
    "distance": 0.0,
    "speed": 0.0,
    "impossible_speed": 0,
}
