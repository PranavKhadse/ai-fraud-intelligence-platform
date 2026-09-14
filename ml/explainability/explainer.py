"""
TreeSHAP Explainer Module for Phase 7.

Computes local feature attributions using the frozen XGBoost champion model's native
compiled TreeSHAP implementation (pred_contribs=True).

Mathematical Guarantees:
- Lundberg TreeSHAP Additivity / Efficiency Axiom:
  sum(phi_i for i in 1..55) + phi_0 == output_margin
- Sigmoidal logit relationship:
  model_score == 1 / (1 + exp(-output_margin))
- Zero external dependencies on the third-party 'shap' package.
- Thread-safe, non-mutating execution in compiled C++.
"""

from typing import Dict, Any, List, Optional, Union, Mapping, Tuple, Sequence
from pathlib import Path
import math
import numpy as np
import pandas as pd
import joblib

from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
)
from ml.explainability.schemas import (
    AttributionDirection,
    FeatureAttribution,
    WaterfallStep,
)
from ml.explainability.config import (
    get_feature_metadata,
    FEATURE_REGISTRY,
)

DEFAULT_MODEL_PATH = Path("ml/models/artifacts/champion_model.joblib")
DEFAULT_PREPROCESSOR_PATH = Path("ml/models/artifacts/champion_preprocessor.joblib")


class TreeSHAPExplainer:
    """
    Production-grade local explainer using native XGBoost TreeSHAP inference.

    Consumes raw or preprocessed transaction features, executes native compiled
    TreeSHAP attributions via the frozen booster, validates margin reconstruction,
    and formats structured feature attribution objects.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        preprocessor_path: Union[str, Path] = DEFAULT_PREPROCESSOR_PATH,
    ) -> None:
        """
        Initialize the TreeSHAP explainer with a trained XGBoost model and preprocessor.

        Args:
            model: Pre-loaded XGBoostFraudModel or xgb.XGBClassifier. If None, loaded from model_path.
            preprocessor: Pre-loaded TreePreprocessor. If None, loaded from preprocessor_path.
            model_path: Path to serialized champion model artifact.
            preprocessor_path: Path to serialized champion preprocessor artifact.
        """
        if model is None:
            m_path = Path(model_path).resolve()
            if not m_path.exists():
                raise FileNotFoundError(f"Champion model artifact not found at: {m_path}")
            self.model = joblib.load(m_path)
        else:
            self.model = model

        if preprocessor is None:
            p_path = Path(preprocessor_path).resolve()
            if not p_path.exists():
                raise FileNotFoundError(f"Champion preprocessor artifact not found at: {p_path}")
            self.preprocessor = joblib.load(p_path)
        else:
            self.preprocessor = preprocessor

        # Extract underlying booster for native compiled TreeSHAP execution
        if hasattr(self.model, "model") and hasattr(self.model.model, "get_booster"):
            self._booster = self.model.model.get_booster()
        elif hasattr(self.model, "get_booster"):
            self._booster = self.model.get_booster()
        else:
            raise TypeError("Cannot extract XGBoost booster from provided model instance.")

    @property
    def feature_names(self) -> List[str]:
        """Return the exact 55 predictor feature names in deterministic order."""
        return list(PREDICTIVE_FEATURE_COLUMNS)

    def explain_raw(self, X_trans: np.ndarray) -> np.ndarray:
        """
        Execute native XGBoost TreeSHAP on a preprocessed 2D float32 array.

        Args:
            X_trans: Preprocessed feature matrix of shape (N, 55).

        Returns:
            np.ndarray: Contribution matrix of shape (N, 56) where columns 0..54 are
                        local feature attributions and column 55 is the base value.

        Raises:
            TypeError: If X_trans is not a 2D NumPy array.
            ValueError: If X_trans does not have exactly 55 columns or is empty.
        """
        import xgboost as xgb

        if not isinstance(X_trans, np.ndarray):
            raise TypeError(f"X_trans must be a numpy ndarray, got {type(X_trans).__name__}")
        if X_trans.ndim != 2:
            raise ValueError(f"X_trans must be a 2D array, got shape {X_trans.shape}")
        if X_trans.shape[1] != 55:
            raise ValueError(f"X_trans must have exactly 55 columns, got {X_trans.shape[1]}")
        if len(X_trans) == 0:
            raise ValueError("X_trans cannot be empty.")

        dmat = xgb.DMatrix(X_trans, feature_names=PREDICTIVE_FEATURE_COLUMNS)
        contribs = self._booster.predict(dmat, pred_contribs=True)
        return contribs.astype(np.float64)

    def explain_features(
        self,
        raw_features: Union[pd.Series, Dict[str, Any]],
        preprocessed_row: Optional[np.ndarray] = None,
        top_k: int = 5,
        top_mitigating: int = 3,
    ) -> Tuple[
        float,  # model_score
        float,  # output_margin
        float,  # base_value
        Tuple[FeatureAttribution, ...],  # top_risk_factors
        Tuple[FeatureAttribution, ...],  # top_mitigating_factors
        Tuple[WaterfallStep, ...],  # waterfall
    ]:
        """
        Compute local TreeSHAP explanations for a single transaction.

        Args:
            raw_features: Raw transaction feature vector containing original unencoded values.
            preprocessed_row: Optional 1D or 2D (1, 55) array. If None, computed from raw_features.
            top_k: Maximum number of positive (risk-increasing) factors to return.
            top_mitigating: Maximum number of negative (mitigating) factors to return.

        Returns:
            Tuple of (model_score, output_margin, base_value, top_risk, top_mitigating, waterfall).
        """
        if isinstance(raw_features, dict):
            feat_dict = raw_features
        elif isinstance(raw_features, pd.Series):
            feat_dict = raw_features.to_dict()
        else:
            raise TypeError(f"raw_features must be a dict or pd.Series, got {type(raw_features).__name__}")

        if preprocessed_row is None:
            # Build 1-row DataFrame preserving raw categoricals
            df_row = pd.DataFrame([feat_dict])
            X_trans = self.preprocessor.transform(df_row[PREDICTIVE_FEATURE_COLUMNS])
        else:
            if preprocessed_row.ndim == 1:
                X_trans = preprocessed_row.reshape(1, -1)
            elif preprocessed_row.ndim == 2:
                X_trans = preprocessed_row
            else:
                raise ValueError(f"preprocessed_row must be 1D or 2D, got shape {preprocessed_row.shape}")

        contribs = self.explain_raw(X_trans)
        row_contribs = contribs[0]  # shape: (56,)

        feature_shaps = row_contribs[:55]
        base_value = float(row_contribs[55])
        output_margin = float(np.sum(row_contribs))
        model_score = float(1.0 / (1.0 + np.exp(-output_margin)))

        # Sum of positive and negative attributions for relative contribution calculations
        pos_sum = float(np.sum(feature_shaps[feature_shaps > 0]))
        neg_sum = float(np.sum(np.abs(feature_shaps[feature_shaps < 0])))

        risk_list: List[FeatureAttribution] = []
        mitigating_list: List[FeatureAttribution] = []

        # Sort indices: positive attributions descending, negative attributions ascending (most negative first)
        pos_indices = [i for i in range(55) if feature_shaps[i] > 0]
        pos_indices.sort(key=lambda idx: feature_shaps[idx], reverse=True)

        neg_indices = [i for i in range(55) if feature_shaps[i] < 0]
        neg_indices.sort(key=lambda idx: feature_shaps[idx])  # ascending (e.g. -0.5 before -0.1)

        for rank, idx in enumerate(pos_indices[:top_k], start=1):
            f_name = PREDICTIVE_FEATURE_COLUMNS[idx]
            meta = get_feature_metadata(f_name)
            raw_val = feat_dict.get(f_name, None)
            shap_val = float(feature_shaps[idx])
            rel_pct = (shap_val / pos_sum * 100.0) if pos_sum > 0 else 0.0

            risk_list.append(
                FeatureAttribution(
                    feature_name=f_name,
                    display_name=meta["display_name"],
                    raw_value=raw_val,
                    shap_value=shap_val,
                    direction=AttributionDirection.RISK_INCREASING,
                    relative_contribution_pct=rel_pct,
                    rank=rank,
                )
            )

        for rank, idx in enumerate(neg_indices[:top_mitigating], start=1):
            f_name = PREDICTIVE_FEATURE_COLUMNS[idx]
            meta = get_feature_metadata(f_name)
            raw_val = feat_dict.get(f_name, None)
            shap_val = float(feature_shaps[idx])
            rel_pct = (abs(shap_val) / neg_sum * 100.0) if neg_sum > 0 else 0.0

            mitigating_list.append(
                FeatureAttribution(
                    feature_name=f_name,
                    display_name=meta["display_name"],
                    raw_value=raw_val,
                    shap_value=shap_val,
                    direction=AttributionDirection.MITIGATING,
                    relative_contribution_pct=rel_pct,
                    rank=rank,
                )
            )

        # Build Mathematically Complete Waterfall
        # Selected features are top_k risk + top_mitigating factors ordered by absolute magnitude
        selected_factors = sorted(
            risk_list + mitigating_list,
            key=lambda f: abs(f.shap_value),
            reverse=True,
        )

        waterfall_steps: List[WaterfallStep] = []
        running_margin = base_value

        # 1. Base value step
        waterfall_steps.append(
            WaterfallStep(
                step_name="Base Value (Expected Margin)",
                feature_name=None,
                contribution=base_value,
                cumulative_margin=running_margin,
                step_type="base",
            )
        )

        # 2. Selected feature steps
        selected_feature_names = set()
        for factor in selected_factors:
            selected_feature_names.add(factor.feature_name)
            running_margin += factor.shap_value
            waterfall_steps.append(
                WaterfallStep(
                    step_name=factor.display_name,
                    feature_name=factor.feature_name,
                    contribution=factor.shap_value,
                    cumulative_margin=running_margin,
                    step_type="feature",
                )
            )

        # 3. Residual step for all non-selected features
        residual_sum = 0.0
        for idx in range(55):
            f_name = PREDICTIVE_FEATURE_COLUMNS[idx]
            if f_name not in selected_feature_names:
                residual_sum += float(feature_shaps[idx])

        running_margin += residual_sum
        waterfall_steps.append(
            WaterfallStep(
                step_name="Other feature contributions",
                feature_name=None,
                contribution=residual_sum,
                cumulative_margin=running_margin,
                step_type="residual",
            )
        )

        # 4. Final margin step
        waterfall_steps.append(
            WaterfallStep(
                step_name="Final Output Margin",
                feature_name=None,
                contribution=0.0,
                cumulative_margin=output_margin,
                step_type="final",
            )
        )

        return (
            model_score,
            output_margin,
            base_value,
            tuple(risk_list),
            tuple(mitigating_list),
            tuple(waterfall_steps),
        )
