"""
Risk Evaluator Integration Module for Phase 6.

Integrates the frozen Phase 4 XGBoost champion model and preprocessor with the
configurable Phase 6 Decision Policy Engine, supporting end-to-end evaluation,
metadata provenance, and batch summary metrics.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Any, List, Optional, Union, Mapping
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
DEFAULT_METADATA_PATH = Path("ml/models/artifacts/model_metadata.json")


@dataclass(frozen=True)
class BatchDecisionSummary:
    """
    Structured aggregate summary of a batch evaluation run.
    Guarantees deep immutability across all nested mapping structures.

    Attributes:
        total_transactions: Total number of transactions evaluated.
        action_counts: Absolute count per decision action (APPROVE, REVIEW, BLOCK).
        action_percentages: Percentage of transactions per decision action.
        risk_tier_counts: Absolute count per risk tier (LOW, MEDIUM, HIGH, CRITICAL).
        risk_tier_percentages: Percentage of transactions per risk tier.
        model_score_stats: Summary statistics (mean, min, max) of raw model scores.
        risk_score_stats: Summary statistics (mean, min, max) of normalized risk scores.
        policy_mode: Policy mode string used during batch evaluation.
        thresholds_applied: Exact threshold boundaries applied during evaluation.
        model_version: Provenance model version string if available.
    """
    total_transactions: int
    action_counts: Mapping[str, int]
    action_percentages: Mapping[str, float]
    risk_tier_counts: Mapping[str, int]
    risk_tier_percentages: Mapping[str, float]
    model_score_stats: Mapping[str, float]
    risk_score_stats: Mapping[str, float]
    policy_mode: str
    thresholds_applied: Mapping[str, float]
    model_version: Optional[str]

    def __post_init__(self) -> None:
        """Enforce deep immutability by wrapping all mapping fields in MappingProxyType."""
        mapping_fields = [
            "action_counts",
            "action_percentages",
            "risk_tier_counts",
            "risk_tier_percentages",
            "model_score_stats",
            "risk_score_stats",
            "thresholds_applied",
        ]
        for f in mapping_fields:
            val = getattr(self, f)
            if not isinstance(val, (dict, MappingProxyType, Mapping)):
                raise TypeError(f"{f} must be a mapping, got {type(val).__name__}")
            object.__setattr__(self, f, MappingProxyType(dict(val)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch summary to a clean JSON-serializable dictionary with independent data."""
        return {
            "total_transactions": self.total_transactions,
            "action_counts": dict(self.action_counts),
            "action_percentages": dict(self.action_percentages),
            "risk_tier_counts": dict(self.risk_tier_counts),
            "risk_tier_percentages": dict(self.risk_tier_percentages),
            "model_score_stats": dict(self.model_score_stats),
            "risk_score_stats": dict(self.risk_score_stats),
            "policy_mode": self.policy_mode,
            "thresholds_applied": dict(self.thresholds_applied),
            "model_version": self.model_version,
        }


class RiskEvaluator:
    """
    End-to-end Risk Evaluator that consumes transaction payloads conforming to
    PREDICTIVE_FEATURE_COLUMNS, executes the frozen champion ML pipeline in read-only mode,
    and returns structured DecisionResult instances from the DecisionPolicyEngine.

    Governance Rules:
    - Does NOT recompute or alter Phase 3 feature engineering.
    - Operates strictly on pre-computed 55-predictor feature vectors.
    - Extra metadata columns (e.g. transaction_id, account_id, timestamp, is_fraud) are safely ignored.
    - Frozen champion artifacts are loaded in read-only mode and never modified.
    - Model outputs are treated as continuous ranking scores in [0.0, 1.0], not calibrated probabilities.
    - Does not mutate input DataFrames, Series, or dicts.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        preprocessor_path: Union[str, Path] = DEFAULT_PREPROCESSOR_PATH,
        metadata_path: Optional[Union[str, Path]] = DEFAULT_METADATA_PATH,
        policy_config: Optional[DecisionPolicyConfig] = None,
    ) -> None:
        """
        Initialize the evaluator by loading frozen artifacts and configuring the policy engine.

        Args:
            model_path: Path to serialized champion model artifact.
            preprocessor_path: Path to serialized champion preprocessor artifact.
            metadata_path: Path to model metadata JSON artifact for trustworthy provenance version.
            policy_config: DecisionPolicyConfig instance. If None, default TRI_TIER config is used.
        """
        self.model_path = Path(model_path).resolve()
        self.preprocessor_path = Path(preprocessor_path).resolve()
        self.metadata_path = Path(metadata_path).resolve() if metadata_path else None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Champion model artifact not found at: {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Champion preprocessor artifact not found at: {self.preprocessor_path}")

        # Read-only load of frozen artifacts
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        # Extract trustworthy model version from metadata if available (otherwise None)
        self.model_version: Optional[str] = self._extract_trustworthy_model_version(self.metadata_path)

        # Initialize policy engine with model version provenance
        self.policy_engine = DecisionPolicyEngine(
            config=policy_config,
            model_version=self.model_version,
        )

    @property
    def config(self) -> DecisionPolicyConfig:
        """Return the immutable policy configuration."""
        return self.policy_engine.config

    @staticmethod
    def _extract_trustworthy_model_version(metadata_path: Optional[Path]) -> Optional[str]:
        """
        Extract model_version string from model metadata JSON if it exists and is valid.
        Does not invent version strings if missing or untrustworthy.
        """
        if metadata_path is not None and metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ver = data.get("model_version")
                if isinstance(ver, str) and ver.strip():
                    return ver.strip()
            except Exception:
                return None
        return None

    def _extract_and_validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and extract the exact 55 predictive columns in deterministic order.
        Guarantees that caller-owned input DataFrame is NOT mutated.

        Args:
            df: Input DataFrame containing predictive features (and optional extra metadata).

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
            df: DataFrame containing the 55 predictive features (plus optional metadata columns).

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

        Metadata columns (e.g. transaction_id, account_id, timestamp, is_fraud) are safely ignored.

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

    def evaluate_dataframe_summary(self, df: pd.DataFrame) -> BatchDecisionSummary:
        """
        Evaluate a batch DataFrame and compute comprehensive summary statistics.

        Args:
            df: DataFrame containing the 55 predictive features (plus optional metadata).

        Returns:
            BatchDecisionSummary: Aggregated metrics across actions, risk tiers, and score statistics.

        Raises:
            ValueError: If df is empty or invalid.
        """
        results = self.evaluate_dataframe(df)
        total = len(results)
        if total == 0:
            raise ValueError("Cannot compute summary on an empty evaluation result.")

        # Action counts & percentages
        action_counts = {a.value: 0 for a in DecisionAction}
        for r in results:
            action_counts[r.action.value] += 1
        action_percentages = {
            k: round((v / total) * 100.0, 4) for k, v in action_counts.items()
        }

        # Risk tier counts & percentages
        tier_counts = {t.value: 0 for t in RiskTier}
        for r in results:
            tier_counts[r.risk_tier.value] += 1
        tier_percentages = {
            k: round((v / total) * 100.0, 4) for k, v in tier_counts.items()
        }

        # Score distributions
        model_scores = [r.model_score for r in results]
        risk_scores = [r.risk_score for r in results]

        model_score_stats = {
            "mean": round(float(np.mean(model_scores)), 6),
            "min": round(float(np.min(model_scores)), 6),
            "max": round(float(np.max(model_scores)), 6),
        }
        risk_score_stats = {
            "mean": round(float(np.mean(risk_scores)), 4),
            "min": int(np.min(risk_scores)),
            "max": int(np.max(risk_scores)),
        }

        return BatchDecisionSummary(
            total_transactions=total,
            action_counts=action_counts,
            action_percentages=action_percentages,
            risk_tier_counts=tier_counts,
            risk_tier_percentages=tier_percentages,
            model_score_stats=model_score_stats,
            risk_score_stats=risk_score_stats,
            policy_mode=self.config.policy_mode.value,
            thresholds_applied={
                "review_threshold": self.config.review_threshold,
                "block_threshold": self.config.block_threshold,
            },
            model_version=self.model_version,
        )
