"""
Model 4: Production LightGBM Gradient Boosted Decision Tree Classifier.

High-throughput, memory-efficient GBDT model optimized for fast inference and PR-AUC.
Uses LightGBM native C-API Booster engine with contiguous float64 arrays and single-thread execution
for deterministic, cross-platform stability on Windows.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from ml.models.config import MODEL_CONFIGS


class LightGBMFraudModel:
    """
    LightGBM Classifier with positive class weighting and gain-based feature attribution.
    """
    def __init__(self, **kwargs):
        config = MODEL_CONFIGS["lightgbm"].copy()
        config.update(kwargs)
        self.config = config
        self.booster: Optional[Any] = None
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LightGBMFraudModel":
        """
        Fit LightGBM booster on training matrix.
        Calculates scale_pos_weight = N_neg / N_pos for class imbalance compensation.
        Enforces C-contiguous float64 inputs and single-threaded dataset creation for Windows stability.
        """
        import lightgbm as lgb
        
        y_raw = y_train.values if hasattr(y_train, "values") else y_train
        X_mat = np.ascontiguousarray(X_train, dtype=np.float64)
        y_vec = np.ascontiguousarray(y_raw, dtype=np.float64)
        
        n_pos = int(np.sum(y_vec == 1.0))
        n_neg = int(np.sum(y_vec == 0.0))
        scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
        num_threads = self.config.get("num_threads", 1)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": self.config.get("num_leaves", 31),
            "max_depth": self.config.get("max_depth", 6),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "scale_pos_weight": scale_pos_weight,
            "random_state": self.config.get("random_state", 42),
            "num_threads": num_threads,
            "verbosity": -1,
        }
        num_boost_round = self.config.get("n_estimators", 150)
        
        dataset_params = {
            "num_threads": num_threads,
            "verbosity": -1,
        }
        train_data = lgb.Dataset(X_mat, label=y_vec, params=dataset_params, free_raw_data=False)
        self.booster = lgb.train(params, train_data, num_boost_round=num_boost_round)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior fraud probability P(Fraud | X)."""
        assert self.is_fitted, "Model must be fitted before predict_proba!"
        X_mat = np.ascontiguousarray(X, dtype=np.float64)
        num_threads = self.config.get("num_threads", 1)
        return self.booster.predict(
            X_mat,
            num_iteration=self.booster.best_iteration,
            num_threads=num_threads,
        ).astype(np.float64)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary fraud labels based on decision threshold."""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(np.int64)

    def get_feature_importance(self, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Return gain-based feature importances."""
        assert self.is_fitted, "Model must be fitted to extract feature importance!"
        importances = self.booster.feature_importance(importance_type="gain")
        importance_list = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in range(len(feature_names))
        ]
        return sorted(importance_list, key=lambda x: x["importance"], reverse=True)
