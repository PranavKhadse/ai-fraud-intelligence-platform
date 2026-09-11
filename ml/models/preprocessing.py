"""
Feature Extraction, Preprocessing Pipelines, and Data Validation for Phase 4 Modeling.

Guarantees:
- Target is_fraud and transaction_id are strictly excluded from X.
- Preprocessing encoders and scalers are fitted STRICTLY on Train data.
- Feature schema is identical across Train, Validation, and OOT Test sets.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

from ml.models.config import (
    TARGET_COLUMN,
    EXCLUDED_COLUMNS,
    CATEGORICAL_PREDICTORS,
    PREDICTIVE_FEATURE_COLUMNS,
)


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract predictive feature matrix X and target label vector y from a dataset partition.
    
    Enforces strict assertions:
    - Target is_fraud is separated into y and NOT present in X.
    - transaction_id and raw IDs are NOT present in X.
    - X has exactly 55 predictive features matching PREDICTIVE_FEATURE_COLUMNS.
    - All numerical inputs are finite with zero NaN / Inf values.
    
    Args:
        df: Parquet partition DataFrame conforming to 62-column Phase 3 schema.
        
    Returns:
        Tuple[X, y]: Feature DataFrame and Target Series.
    """
    assert TARGET_COLUMN in df.columns, f"Missing target column '{TARGET_COLUMN}' in DataFrame!"
    y = df[TARGET_COLUMN].astype(np.int64)
    
    # Verify and extract exactly the 55 predictive columns
    missing_cols = [c for c in PREDICTIVE_FEATURE_COLUMNS if c not in df.columns]
    assert len(missing_cols) == 0, f"DataFrame is missing expected predictive columns: {missing_cols}"
    
    if TARGET_COLUMN in df.columns and len(df.columns) == 56:
        X = df.drop(columns=[TARGET_COLUMN])
    else:
        X = df[PREDICTIVE_FEATURE_COLUMNS]

    
    # Defensive validations
    assert TARGET_COLUMN not in X.columns, "Target column is_fraud leaked into feature matrix X!"
    assert "transaction_id" not in X.columns, "transaction_id leaked into feature matrix X!"
    for col in EXCLUDED_COLUMNS:
        assert col not in X.columns, f"Excluded column '{col}' found in feature matrix X!"
        
    assert len(X.columns) == 55, f"Expected exactly 55 predictive features, found {len(X.columns)}!"
    assert list(X.columns) == PREDICTIVE_FEATURE_COLUMNS, "Predictive feature columns ordering mismatch!"
    assert len(X) == len(df), "Row count mismatch between X and input DataFrame!"
    
    # Validate numerical finiteness
    num_cols = [c for c in PREDICTIVE_FEATURE_COLUMNS if c not in CATEGORICAL_PREDICTORS]
    for col in num_cols:
        assert np.isfinite(X[col]).all(), f"Non-finite or NaN values detected in numerical feature '{col}'!"
    
    return X, y



class TreePreprocessor:
    """
    Lightweight, leakage-safe preprocessor for tree-based models (Random Forest, XGBoost, LightGBM).
    Encodes categorical features via OrdinalEncoder fitted strictly on Train.
    """
    def __init__(self):
        from sklearn.preprocessing import OrdinalEncoder
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        self.feature_names: List[str] = PREDICTIVE_FEATURE_COLUMNS
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "TreePreprocessor":
        """Fit categorical encoders strictly on training data."""
        self.encoder.fit(X[CATEGORICAL_PREDICTORS])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features to dense C-contiguous float32 array."""
        assert self.is_fitted, "TreePreprocessor must be fitted before transform!"
        encoded_cats = self.encoder.transform(X[CATEGORICAL_PREDICTORS]).astype(np.float32)
        out = np.empty((len(X), 55), dtype=np.float32, order="C")
        out[:, :6] = X[PREDICTIVE_FEATURE_COLUMNS[:6]].to_numpy(dtype=np.float32, copy=False)
        out[:, 6:8] = encoded_cats
        out[:, 8:] = X[PREDICTIVE_FEATURE_COLUMNS[8:]].to_numpy(dtype=np.float32, copy=False)
        return out


    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


class LinearPreprocessor:
    """
    Leakage-safe preprocessor for linear baseline models (Logistic Regression).
    Applies StandardScaler to numeric features and OneHotEncoder to categorical features,
    fitted strictly on Train.
    """
    def __init__(self):
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        self.numeric_cols = [c for c in PREDICTIVE_FEATURE_COLUMNS if c not in CATEGORICAL_PREDICTORS]
        self.categorical_cols = CATEGORICAL_PREDICTORS
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(min_frequency=0.005, handle_unknown="ignore", sparse_output=False)
        self.is_fitted: bool = False

    def _extract_numeric_array(self, X: pd.DataFrame) -> np.ndarray:
        num_part1 = X[PREDICTIVE_FEATURE_COLUMNS[:6]].to_numpy(dtype=np.float32, copy=False)
        num_part2 = X[PREDICTIVE_FEATURE_COLUMNS[8:]].to_numpy(dtype=np.float32, copy=False)
        return np.hstack([num_part1, num_part2])

    def fit(self, X: pd.DataFrame) -> "LinearPreprocessor":
        """Fit scaler and one-hot encoder strictly on training data."""
        X_num = self._extract_numeric_array(X)
        self.scaler.fit(X_num)
        self.encoder.fit(X[self.categorical_cols])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features to standard-scaled numeric matrix."""
        assert self.is_fitted, "LinearPreprocessor must be fitted before transform!"
        X_num = self._extract_numeric_array(X)
        X_num_scaled = self.scaler.transform(X_num).astype(np.float32)
        X_cat_encoded = self.encoder.transform(X[self.categorical_cols]).astype(np.float32)
        return np.hstack([X_num_scaled, X_cat_encoded])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


