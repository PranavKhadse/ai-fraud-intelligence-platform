"""
Machine Learning Models Package for AI-Powered Fraud Detection & Risk Intelligence Platform.

Uses lazy attribute loading for all model architectures to prevent simultaneous loading of
conflicting native C-runtimes (OpenMP) between Scikit-Learn, XGBoost, and LightGBM on Windows.
"""

from typing import Any

from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
    EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    RANDOM_SEED,
    MODEL_CONFIGS,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    ARTIFACTS_DIR,
)
from ml.models.preprocessing import (
    prepare_features_and_target,
    TreePreprocessor,
    LinearPreprocessor,
)

__all__ = [
    "PREDICTIVE_FEATURE_COLUMNS",
    "CATEGORICAL_PREDICTORS",
    "EXCLUDED_COLUMNS",
    "TARGET_COLUMN",
    "RANDOM_SEED",
    "MODEL_CONFIGS",
    "TRAIN_FEATURES_PATH",
    "VAL_FEATURES_PATH",
    "TEST_FEATURES_PATH",
    "ARTIFACTS_DIR",
    "prepare_features_and_target",
    "TreePreprocessor",
    "LinearPreprocessor",
    "BaselineLogisticRegression",
    "RandomForestBaseline",
    "XGBoostFraudModel",
    "LightGBMFraudModel",
    "run_isolated_model_job",
]


def __getattr__(name: str) -> Any:
    """Lazy import models to maintain process-level OpenMP runtime isolation on Windows."""
    if name == "BaselineLogisticRegression":
        from ml.models.baseline import BaselineLogisticRegression
        return BaselineLogisticRegression
    if name == "RandomForestBaseline":
        from ml.models.random_forest import RandomForestBaseline
        return RandomForestBaseline
    if name == "XGBoostFraudModel":
        from ml.models.xgboost_model import XGBoostFraudModel
        return XGBoostFraudModel
    if name == "LightGBMFraudModel":
        from ml.models.lightgbm_model import LightGBMFraudModel
        return LightGBMFraudModel
    if name == "run_isolated_model_job":
        from ml.models.runner import run_isolated_model_job
        return run_isolated_model_job
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
