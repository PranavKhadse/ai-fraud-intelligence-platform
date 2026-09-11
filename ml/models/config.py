"""
Configuration, Constants, Column Policies, and Hyperparameters for Phase 4 ML Modeling.
"""

from typing import List, Dict, Any
from pathlib import Path

# Paths
TRAIN_FEATURES_PATH = Path("data/processed/features/train_features.parquet")
VAL_FEATURES_PATH = Path("data/processed/features/val_features.parquet")
TEST_FEATURES_PATH = Path("data/processed/features/test_features.parquet")
ARTIFACTS_DIR = Path("ml/models/artifacts")
DOCS_DIR = Path("docs")

# Target Variable
TARGET_COLUMN: str = "is_fraud"

# Excluded Columns Policy (7 columns excluded from predictive feature matrix X)
# 1. is_fraud: Target column (strictly separated to prevent label leakage)
# 2. transaction_id: High-cardinality unique record ID
# 3. account_id: High-cardinality entity ID (behavioral history captured in engineered features)
# 4. timestamp: Raw datetime object (cyclic/temporal features capture time patterns)
# 5. unix_time: Raw integer timestamp (monotonic timeline)
# 6. currency: Constant string ("USD")
# 7. merchant_id: High-cardinality merchant ID (interaction features capture merchant patterns)
EXCLUDED_COLUMNS: List[str] = [
    "is_fraud",
    "transaction_id",
    "account_id",
    "timestamp",
    "unix_time",
    "currency",
    "merchant_id",
]

# Categorical Predictors in X (2 features)
CATEGORICAL_PREDICTORS: List[str] = [
    "merchant_category",
    "job_category",
]

# Canonical Numeric Predictors in X (6 features)
CANONICAL_NUMERIC_PREDICTORS: List[str] = [
    "amount",
    "cardholder_lat",
    "cardholder_long",
    "merchant_lat",
    "merchant_long",
    "city_pop",
]

# Engineered Features from Phase 3 (47 features)
from ml.features.config import ENGINEERED_FEATURE_COLUMNS

# All 55 Predictive Features in exact deterministic order
PREDICTIVE_FEATURE_COLUMNS: List[str] = (
    CANONICAL_NUMERIC_PREDICTORS
    + CATEGORICAL_PREDICTORS
    + ENGINEERED_FEATURE_COLUMNS
)

# Random Seed for Reproducibility
RANDOM_SEED: int = 42

# Model Hyperparameters Configuration
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 500,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "n_jobs": 4,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.08,
        "n_estimators": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "n_jobs": 4,
        "random_state": RANDOM_SEED,
    },
    "lightgbm": {
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary",
        "importance_type": "gain",
        "num_threads": 1,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    },
}
