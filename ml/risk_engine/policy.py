"""
Decision Policy Engine & Evaluation Result Structures for the Risk Engine.

Evaluates raw model ranking scores against configured decision policies to determine
the operational transaction action (APPROVE, REVIEW, or BLOCK).
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import numpy as np

from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    DecisionPolicyConfig,
)
from ml.risk_engine.normalization import (
    normalize_model_score,
    normalize_model_scores,
    risk_tier_from_score,
)


@dataclass(frozen=True)
class DecisionResult:
    """
    Immutable, serializable decision result produced by the DecisionPolicyEngine.

    Attributes:
        action: Operational decision action (APPROVE, REVIEW, BLOCK).
        risk_score: Standardized integer Risk Score in [0, 100].
        risk_tier: Semantic risk band (LOW, MEDIUM, HIGH, CRITICAL).
        model_score: Raw continuous model ranking score in [0.0, 1.0].
        policy_mode: Policy mode used during evaluation (TRI_TIER or BINARY_AUTO).
        reason: Human-readable explanation of the policy decision boundary.
    """
    action: DecisionAction
    risk_score: int
    risk_tier: RiskTier
    model_score: float
    policy_mode: PolicyMode
    reason: str

    def __post_init__(self) -> None:
        """Defensive validation of result fields."""
        if not isinstance(self.action, DecisionAction):
            raise TypeError(f"action must be a DecisionAction enum, got {type(self.action).__name__}")
        if not isinstance(self.risk_score, int) or isinstance(self.risk_score, bool):
            raise TypeError(f"risk_score must be an integer, got {type(self.risk_score).__name__}")
        if self.risk_score < 0 or self.risk_score > 100:
            raise ValueError(f"risk_score must be in [0, 100], got {self.risk_score}")
        if not isinstance(self.risk_tier, RiskTier):
            raise TypeError(f"risk_tier must be a RiskTier enum, got {type(self.risk_tier).__name__}")
        if not isinstance(self.model_score, (float, int)) or isinstance(self.model_score, bool):
            raise TypeError(f"model_score must be a float, got {type(self.model_score).__name__}")
        if not isinstance(self.policy_mode, PolicyMode):
            raise TypeError(f"policy_mode must be a PolicyMode enum, got {type(self.policy_mode).__name__}")
        if not isinstance(self.reason, str):
            raise TypeError(f"reason must be a string, got {type(self.reason).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision result to a clean JSON-serializable dictionary."""
        return {
            "action": self.action.value,
            "risk_score": self.risk_score,
            "risk_tier": self.risk_tier.value,
            "model_score": round(float(self.model_score), 6),
            "policy_mode": self.policy_mode.value,
            "reason": self.reason,
        }


class DecisionPolicyEngine:
    """
    Deterministic Decision Policy Engine that evaluates raw model scores against
    configured threshold policies (TRI_TIER or BINARY_AUTO).
    """

    def __init__(self, config: Optional[DecisionPolicyConfig] = None) -> None:
        """
        Initialize the policy engine with a validated policy configuration.

        Args:
            config: DecisionPolicyConfig instance. If None, default configuration is used.
        """
        if config is None:
            config = DecisionPolicyConfig()
        elif not isinstance(config, DecisionPolicyConfig):
            raise TypeError(
                f"config must be an instance of DecisionPolicyConfig, got {type(config).__name__}"
            )
        self._config = config

    @property
    def config(self) -> DecisionPolicyConfig:
        """Return the immutable policy configuration."""
        return self._config

    def evaluate(self, model_score: Union[float, int, np.floating, np.integer]) -> DecisionResult:
        """
        Evaluate a single raw model ranking score against policy thresholds.

        Routing Logic:
        - TRI_TIER Mode:
            - model_score >= block_threshold  -> BLOCK
            - model_score >= review_threshold -> REVIEW
            - else                            -> APPROVE
        - BINARY_AUTO Mode:
            - model_score >= block_threshold  -> BLOCK
            - else                            -> APPROVE

        Args:
            model_score: Raw model ranking score in [0.0, 1.0].

        Returns:
            DecisionResult: Immutable evaluation outcome.

        Raises:
            TypeError: If model_score is boolean or non-numeric.
            ValueError: If model_score is NaN, infinite, or out of [0.0, 1.0].
        """
        # Normalize and validate raw score
        risk_score = normalize_model_score(model_score)
        risk_tier = risk_tier_from_score(risk_score)
        score_val = float(model_score)

        mode = self._config.policy_mode
        b_thresh = self._config.block_threshold
        r_thresh = self._config.review_threshold

        if mode == PolicyMode.TRI_TIER:
            if score_val >= b_thresh:
                action = DecisionAction.BLOCK
                reason = (
                    f"Model score ({score_val:.4f}) meets or exceeds block threshold ({b_thresh:.4f}). "
                    f"Action: BLOCK."
                )
            elif score_val >= r_thresh:
                action = DecisionAction.REVIEW
                reason = (
                    f"Model score ({score_val:.4f}) meets or exceeds review threshold ({r_thresh:.4f}) "
                    f"but is below block threshold ({b_thresh:.4f}). Action: REVIEW."
                )
            else:
                action = DecisionAction.APPROVE
                reason = (
                    f"Model score ({score_val:.4f}) is below review threshold ({r_thresh:.4f}). "
                    f"Action: APPROVE."
                )
        elif mode == PolicyMode.BINARY_AUTO:
            if score_val >= b_thresh:
                action = DecisionAction.BLOCK
                reason = (
                    f"Model score ({score_val:.4f}) meets or exceeds block threshold ({b_thresh:.4f}) "
                    f"in BINARY_AUTO mode. Action: BLOCK."
                )
            else:
                action = DecisionAction.APPROVE
                reason = (
                    f"Model score ({score_val:.4f}) is below block threshold ({b_thresh:.4f}) "
                    f"in BINARY_AUTO mode. Action: APPROVE."
                )
        else:
            raise ValueError(f"Unsupported policy mode: {mode}")

        return DecisionResult(
            action=action,
            risk_score=risk_score,
            risk_tier=risk_tier,
            model_score=score_val,
            policy_mode=mode,
            reason=reason,
        )

    def evaluate_batch(self, model_scores: np.ndarray) -> List[DecisionResult]:
        """
        Evaluate a 1D NumPy array of raw model scores against policy thresholds.

        Preserves input array order.

        Args:
            model_scores: 1D NumPy array of raw model scores in [0.0, 1.0].

        Returns:
            List[DecisionResult]: List of decision results matching input array ordering.

        Raises:
            TypeError: If input is not a numeric 1D NumPy array.
            ValueError: If array is empty, multidimensional, or contains out-of-range/NaN elements.
        """
        # Validate array structure via normalize_model_scores
        _ = normalize_model_scores(model_scores)

        # Evaluate deterministically in sequence
        return [self.evaluate(s) for s in model_scores]
