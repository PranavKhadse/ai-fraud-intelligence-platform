"""
Decision Policy Engine & Evaluation Result Structures for the Risk Engine.

Evaluates raw model ranking scores against configured decision policies to determine
the operational transaction action (APPROVE, REVIEW, or BLOCK).
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Any, List, Optional, Union, Mapping, Tuple
import numpy as np

from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    DecisionPolicyConfig,
    DecisionReasonCode,
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
        thresholds_applied: Exact raw score threshold boundaries applied during evaluation (read-only mapping).
        model_version: Provenance version of the champion model artifact if available.
        reason_codes: Immutable tuple of policy-level decision reason codes.
    """
    action: DecisionAction
    risk_score: int
    risk_tier: RiskTier
    model_score: float
    policy_mode: PolicyMode
    reason: str
    thresholds_applied: Mapping[str, float] = field(default_factory=dict)
    model_version: Optional[str] = None
    reason_codes: Tuple[DecisionReasonCode, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Defensive validation of result fields and enforcement of deep immutability."""
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
        if not isinstance(self.thresholds_applied, (dict, MappingProxyType, Mapping)):
            raise TypeError(f"thresholds_applied must be a mapping, got {type(self.thresholds_applied).__name__}")
        if self.model_version is not None and not isinstance(self.model_version, str):
            raise TypeError(f"model_version must be a str or None, got {type(self.model_version).__name__}")

        # Enforce deep immutability on thresholds_applied via read-only MappingProxyType
        object.__setattr__(
            self, "thresholds_applied", MappingProxyType(dict(self.thresholds_applied))
        )

        # Validate and enforce tuple immutability on reason_codes
        if not isinstance(self.reason_codes, (tuple, list)):
            raise TypeError(f"reason_codes must be a tuple or list, got {type(self.reason_codes).__name__}")
        for code in self.reason_codes:
            if not isinstance(code, DecisionReasonCode):
                raise TypeError(
                    f"reason_codes elements must be DecisionReasonCode enum instances, got {type(code).__name__}"
                )
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision result to a clean JSON-serializable dictionary."""
        return {
            "action": self.action.value,
            "risk_score": self.risk_score,
            "risk_tier": self.risk_tier.value,
            "model_score": round(float(self.model_score), 6),
            "policy_mode": self.policy_mode.value,
            "reason": self.reason,
            "thresholds_applied": {
                k: round(float(v), 6) for k, v in self.thresholds_applied.items()
            },
            "model_version": self.model_version,
            "reason_codes": [code.value for code in self.reason_codes],
        }


class DecisionPolicyEngine:
    """
    Deterministic Decision Policy Engine that evaluates raw model scores against
    configured threshold policies (TRI_TIER or BINARY_AUTO).
    """

    def __init__(
        self,
        config: Optional[DecisionPolicyConfig] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """
        Initialize the policy engine with a validated policy configuration.

        Args:
            config: DecisionPolicyConfig instance. If None, default configuration is used.
            model_version: Optional model version string recorded for decision provenance.
        """
        if config is None:
            config = DecisionPolicyConfig()
        elif not isinstance(config, DecisionPolicyConfig):
            raise TypeError(
                f"config must be an instance of DecisionPolicyConfig, got {type(config).__name__}"
            )
        self._config = config
        self._model_version = str(model_version).strip() if model_version is not None else None

    @property
    def config(self) -> DecisionPolicyConfig:
        """Return the immutable policy configuration."""
        return self._config

    @property
    def model_version(self) -> Optional[str]:
        """Return the model version string used for provenance."""
        return self._model_version

    def evaluate(self, model_score: Union[float, int, np.floating, np.integer]) -> DecisionResult:
        """
        Evaluate a single raw model ranking score against policy thresholds.

        Routing Logic:
        - TRI_TIER Mode:
            - model_score >= block_threshold  -> BLOCK (BLOCK_THRESHOLD_REACHED)
            - model_score >= review_threshold -> REVIEW (REVIEW_THRESHOLD_REACHED)
            - else                            -> APPROVE (BELOW_REVIEW_THRESHOLD)
        - BINARY_AUTO Mode:
            - model_score >= block_threshold  -> BLOCK (BINARY_AUTO_BLOCK_THRESHOLD_REACHED)
            - else                            -> APPROVE (BINARY_AUTO_APPROVE_BELOW_BLOCK_THRESHOLD)

        Args:
            model_score: Raw model ranking score in [0.0, 1.0].

        Returns:
            DecisionResult: Immutable evaluation outcome with full decision provenance.

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

        thresholds_applied = {
            "review_threshold": r_thresh,
            "block_threshold": b_thresh,
        }

        if mode == PolicyMode.TRI_TIER:
            if score_val >= b_thresh:
                action = DecisionAction.BLOCK
                reason_codes = (DecisionReasonCode.BLOCK_THRESHOLD_REACHED,)
                reason = (
                    f"Model score ({score_val:.4f}) meets or exceeds block threshold ({b_thresh:.4f}). "
                    f"Action: BLOCK."
                )
            elif score_val >= r_thresh:
                action = DecisionAction.REVIEW
                reason_codes = (DecisionReasonCode.REVIEW_THRESHOLD_REACHED,)
                reason = (
                    f"Model score ({score_val:.4f}) meets or exceeds review threshold ({r_thresh:.4f}) "
                    f"but is below block threshold ({b_thresh:.4f}). Action: REVIEW."
                )
            else:
                action = DecisionAction.APPROVE
                reason_codes = (DecisionReasonCode.BELOW_REVIEW_THRESHOLD,)
                if r_thresh == b_thresh:
                    reason = (
                        f"Model score ({score_val:.4f}) is below review/block threshold ({b_thresh:.4f}) "
                        f"(zero-width review band). Action: APPROVE."
                    )
                else:
                    reason = (
                        f"Model score ({score_val:.4f}) is below review threshold ({r_thresh:.4f}). "
                        f"Action: APPROVE."
                    )
        elif mode == PolicyMode.BINARY_AUTO:
            if score_val >= b_thresh:
                action = DecisionAction.BLOCK
                reason_codes = (DecisionReasonCode.BINARY_AUTO_BLOCK_THRESHOLD_REACHED,)
                reason = (
                    f"Model score ({score_val:.4f}) meets or exceeds block threshold ({b_thresh:.4f}) "
                    f"in BINARY_AUTO mode. Action: BLOCK."
                )
            else:
                action = DecisionAction.APPROVE
                reason_codes = (DecisionReasonCode.BINARY_AUTO_APPROVE_BELOW_BLOCK_THRESHOLD,)
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
            thresholds_applied=thresholds_applied,
            model_version=self._model_version,
            reason_codes=reason_codes,
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
