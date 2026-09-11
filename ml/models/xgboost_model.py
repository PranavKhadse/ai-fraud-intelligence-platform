"""
Model 3: Production XGBoost Gradient Boosted Classifier.

Implements extreme gradient boosting with positive class weighting (scale_pos_weight)
for imbalanced financial risk intelligence.
Uses lazy runtime import to prevent native OpenMP collisions on Windows.
"""

from typing import Dict, Any, List
import numpy as np

from ml.models.config import MODEL_CONFIGS


class XGBoostFraudModel:
    """
    XGBoost Classifier optimized for Precision-Recall under extreme class imbalance.
    """
    def __init__(self, **kwargs):
        config = MODEL_CONFIGS["xgboost"].copy()
        config.update(kwargs)
        self.config = config
        self.model = None
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostFraudModel":
        """
        Fit XGBoost classifier on training matrix.
        Calculates scale_pos_weight = N_neg / N_pos to handle class imbalance.
        """
        import xgboost as xgb
        
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
        
        cfg = self.config.copy()
        cfg["scale_pos_weight"] = scale_pos_weight
        
        self.model = xgb.XGBClassifier(**cfg)
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
        """Return gain-based feature importances."""
        assert self.is_fitted, "Model must be fitted to extract feature importance!"
        importances = self.model.feature_importances_
        
        importance_list = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in range(len(feature_names))
        ]
        return sorted(importance_list, key=lambda x: x["importance"], reverse=True)
