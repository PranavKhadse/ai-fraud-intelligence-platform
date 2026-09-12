"""
Risk Evaluator Integration Module for Phase 6.

Integrates the frozen Phase 4 XGBoost champion model and preprocessor with the
configurable Phase 6 Decision Policy Engine.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib

from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
)
from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    DecisionPolicyConfig,
)
from ml.risk_engine.policy import (
    DecisionResult,
    DecisionPolicyEngine,
)

DEFAULT_MODEL_PATH = Path("ml/models/artifacts/champion_model.joblib")
DEFAULT_PREPROCESSOR_PATH = Path("ml/models/artifacts/champion_preprocessor.joblib")


class RiskEvaluator:
    """
    End-to-end Risk Evaluator that consumes a transaction DataFrame conforming to
    PREDICTIVE_FEATURE_COLUMNS, executes the frozen champion ML pipeline in read-only mode,
    and returns structured DecisionResult instances from the DecisionPolicyEngine.

    Governance Rules:
    - Does NOT recompute or alter Phase 3 feature engineering.
    - Operates strictly on pre-computed 55-predictor feature vectors.
    - Frozen champion artifacts are loaded in read-only mode and never modified.
    - Model outputs are treated as continuous ranking scores in [0.0, 1.0], not calibrated probabilities.
    - Does not mutate input DataFrames.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        preprocessor_path: Union[str, Path] = DEFAULT_PREPROCESSOR_PATH,
        policy_config: Optional[DecisionPolicyConfig] = None,
    ) -> None:
        """
        Initialize the evaluator by loading frozen artifacts and configuring the policy engine.

        Args:
            model_path: Path to serialized champion model artifact.
            preprocessor_path: Path to serialized champion preprocessor artifact.
            policy_config: DecisionPolicyConfig instance. If None, default TRI_TIER config is used.
        """
        self.model_path = Path(model_path).resolve()
        self.preprocessor_path = Path(preprocessor_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Champion model artifact not found at: {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Champion preprocessor artifact not found at: {self.preprocessor_path}")

        # Read-only load of frozen artifacts
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        # Initialize policy engine
        self.policy_engine = DecisionPolicyEngine(config=policy_config)

    @property
    def config(self) -> DecisionPolicyConfig:
        """Return the immutable policy configuration."""
        return self.policy_engine.config

    def _extract_and_validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and extract the exact 55 predictive columns in deterministic order.
        Guarantees that input DataFrame is NOT mutated.

        Args:
            df: Input DataFrame containing predictive features.

        Returns:
            pd.DataFrame: Sliced DataFrame containing exactly the 55 predictive features.

        Raises:
            TypeError: If input is not a pandas DataFrame.
            ValueError: If DataFrame is empty, missing required columns, or contains NaN/inf values.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(df).__name__}")
        if len(df) == 0:
            raise ValueError("Input DataFrame is empty.")

        missing = [c for c in PREDICTIVE_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing {len(missing)} required predictor columns: {missing[:5]}"
            )

        # Extract exactly the 55 predictive columns in deterministic order without mutating input
        X = df[PREDICTIVE_FEATURE_COLUMNS].copy(deep=False)

        # Validate numerical finiteness
        num_cols = [c for c in PREDICTIVE_FEATURE_COLUMNS if c not in CATEGORICAL_PREDICTORS]
        for c in num_cols:
            vals = X[c].to_numpy()
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"Numerical feature column '{c}' contains NaN or non-finite values.")

        return X

    def evaluate_dataframe(self, df: pd.DataFrame) -> List[DecisionResult]:
        """
        Evaluate a batch of transactions provided in a pandas DataFrame.

        Preserves input row ordering.

        Args:
            df: DataFrame containing the 55 predictive features.

        Returns:
            List[DecisionResult]: List of decision results matching input DataFrame rows.
        """
        X = self._extract_and_validate_features(df)
        X_trans = self.preprocessor.transform(X)
        model_scores = self.model.predict_proba(X_trans)
        return self.policy_engine.evaluate_batch(model_scores)

    def evaluate_transaction(
        self, df_or_series: Union[pd.DataFrame, pd.Series, Dict[str, Any]]
    ) -> DecisionResult:
        """
        Evaluate a single transaction provided as a 1-row DataFrame, Series, or Dictionary.

        Args:
            df_or_series: 1-row DataFrame, Series, or Dictionary containing the 55 features.

        Returns:
            DecisionResult: Evaluation result for the single transaction.
        """
        if isinstance(df_or_series, dict):
            df = pd.DataFrame([df_or_series])
        elif isinstance(df_or_series, pd.Series):
            df = pd.DataFrame([df_or_series.to_dict()])
        elif isinstance(df_or_series, pd.DataFrame):
            if len(df_or_series) != 1:
                raise ValueError(
                    f"evaluate_transaction expects a 1-row DataFrame, got {len(df_or_series)} rows. "
                    "Use evaluate_dataframe for batch evaluation."
                )
            df = df_or_series
        else:
            raise TypeError(
                f"df_or_series must be a 1-row DataFrame, Series, or dict, got {type(df_or_series).__name__}"
            )

        results = self.evaluate_dataframe(df)
        return results[0]
