"""
Model 1: Baseline Logistic Regression Classifier.

Serves as the linear, highly interpretable baseline for imbalanced fraud classification.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.models.config import MODEL_CONFIGS


class BaselineLogisticRegression:
    """
    Interpretable baseline classifier with balanced class weighting.
    """
    def __init__(self, **kwargs):
        config = MODEL_CONFIGS["logistic_regression"].copy()
        config.update(kwargs)
        self.model = LogisticRegression(**config)
        self.config = config
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaselineLogisticRegression":
        """Fit logistic regression on scaled training matrix."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated posterior fraud probability P(Fraud | X)."""
        assert self.is_fitted, "Model must be fitted before predict_proba!"
        return self.model.predict_proba(X)[:, 1].astype(np.float64)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary fraud labels based on decision threshold."""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(np.int64)

    def get_feature_importance(self, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Return coefficient magnitudes for linear feature attribution."""
        assert self.is_fitted, "Model must be fitted to extract coefficients!"
        coefs = self.model.coef_[0]
        n_features = min(len(feature_names), len(coefs))
        
        importance_list = [
            {"feature": feature_names[i], "importance": float(abs(coefs[i])), "raw_coefficient": float(coefs[i])}
            for i in range(n_features)
        ]
        return sorted(importance_list, key=lambda x: x["importance"], reverse=True)
