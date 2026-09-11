"""
Model 2: Random Forest Ensemble Baseline Classifier.

Non-linear tree ensemble baseline with balanced class weighting.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.models.config import MODEL_CONFIGS


class RandomForestBaseline:
    """
    Random Forest ensemble baseline for fraud detection under severe class imbalance.
    """
    def __init__(self, **kwargs):
        config = MODEL_CONFIGS["random_forest"].copy()
        config.update(kwargs)
        self.model = RandomForestClassifier(**config)
        self.config = config
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestBaseline":
        """Fit Random Forest ensemble on training data."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior fraud probability P(Fraud | X)."""
        assert self.is_fitted, "Model must be fitted before predict_proba!"
        return self.model.predict_proba(X)[:, 1].astype(np.float64)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary fraud labels based on decision threshold."""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(np.int64)

    def get_feature_importance(self, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Return Gini impurity feature importances."""
        assert self.is_fitted, "Model must be fitted to extract feature importance!"
        importances = self.model.feature_importances_
        
        importance_list = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in range(len(feature_names))
        ]
        return sorted(importance_list, key=lambda x: x["importance"], reverse=True)
