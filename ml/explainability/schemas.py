"""
Data Contracts, Enums, and Immutable Schemas for Phase 7 Explainability & Reason Codes.

Defines typed, immutable data models representing local TreeSHAP feature attributions,
plain-English reason codes, mathematically complete waterfall steps, and comprehensive
transaction explanation payloads.
"""

from enum import Enum
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Any, List, Optional, Union, Mapping, Tuple
import math
import numpy as np

from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    RuleOutcome,
)
from ml.risk_engine.rules import (
    RuleMatch,
)


class AttributionDirection(str, Enum):
    """Direction of a feature's SHAP attribution relative to fraud risk."""
    RISK_INCREASING = "RISK_INCREASING"
    MITIGATING = "MITIGATING"


class ReasonSource(str, Enum):
    """Origin of an explanation reason code."""
    MODEL = "MODEL"
    RULE = "RULE"


class ReasonSeverity(str, Enum):
    """Operational severity of a reason code."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass(frozen=True)
class FeatureAttribution:
    """
    Immutable record of a single feature's local TreeSHAP attribution.

    Attributes:
        feature_name: Exact predictor column name matching PREDICTIVE_FEATURE_COLUMNS.
        display_name: Human-friendly display label for UI and reports.
        raw_value: Unencoded raw value of the feature from the transaction payload.
        shap_value: Additive TreeSHAP attribution in raw margin (log-odds) space.
        direction: Direction of risk influence (RISK_INCREASING if shap_value > 0 else MITIGATING).
        relative_contribution_pct: Percentage of total positive (or negative) attribution.
        rank: 1-indexed relative importance rank within its direction group.
    """
    feature_name: str
    display_name: str
    raw_value: Any
    shap_value: float
    direction: AttributionDirection
    relative_contribution_pct: float
    rank: int

    def __post_init__(self) -> None:
        """Validate attribution types, bounds, and immutability."""
        if not isinstance(self.feature_name, str) or not self.feature_name.strip():
            raise ValueError(f"feature_name must be a non-empty string, got {self.feature_name!r}")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError(f"display_name must be a non-empty string, got {self.display_name!r}")
        if not isinstance(self.shap_value, (int, float, np.floating)) or isinstance(self.shap_value, bool):
            raise TypeError(f"shap_value must be a float, got {type(self.shap_value).__name__}")
        if not math.isfinite(float(self.shap_value)):
            raise ValueError(f"shap_value must be finite, got {self.shap_value}")
        if not isinstance(self.direction, AttributionDirection):
            raise TypeError(f"direction must be an AttributionDirection enum, got {type(self.direction).__name__}")
        if not isinstance(self.relative_contribution_pct, (int, float, np.floating)) or isinstance(self.relative_contribution_pct, bool):
            raise TypeError(f"relative_contribution_pct must be a float, got {type(self.relative_contribution_pct).__name__}")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise TypeError(f"rank must be an integer, got {type(self.rank).__name__}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")

        object.__setattr__(self, "shap_value", round(float(self.shap_value), 6))
        object.__setattr__(self, "relative_contribution_pct", round(float(self.relative_contribution_pct), 4))

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature attribution to a clean JSON-serializable dictionary."""
        val = self.raw_value
        if isinstance(val, (np.floating, float)):
            val = round(float(val), 6)
        elif isinstance(val, (np.integer, int)) and not isinstance(val, bool):
            val = int(val)
        elif isinstance(val, (np.bool_, bool)):
            val = bool(val)

        return {
            "feature_name": self.feature_name,
            "display_name": self.display_name,
            "raw_value": val,
            "shap_value": self.shap_value,
            "direction": self.direction.value,
            "relative_contribution_pct": self.relative_contribution_pct,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class ReasonCodeDetail:
    """
    Standardized, plain-English reason code explaining transaction risk or mitigation.

    Attributes:
        code: Standard uppercase identifier (e.g. VELOCITY_BURST_1H, RULE_AMT_ZSCORE_DEVIATION_REVIEW).
        headline: Concise, professional summary headline.
        description: Interpolated, plain-English explanation of the risk driver or rule match.
        category: Domain categorization (VELOCITY, AMOUNT, GEOGRAPHY, ACCOUNT_HISTORY, etc.).
        source: Origin of the reason code (MODEL for ML SHAP, RULE for deterministic rules).
        severity: Operational severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO).
        rank: 1-indexed presentation rank.
    """
    code: str
    headline: str
    description: str
    category: str
    source: ReasonSource
    severity: ReasonSeverity
    rank: int

    def __post_init__(self) -> None:
        """Validate reason code fields and types."""
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError(f"code must be a non-empty string, got {self.code!r}")
        if not isinstance(self.headline, str) or not self.headline.strip():
            raise ValueError(f"headline must be a non-empty string, got {self.headline!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(f"description must be a non-empty string, got {self.description!r}")
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError(f"category must be a non-empty string, got {self.category!r}")
        if not isinstance(self.source, ReasonSource):
            raise TypeError(f"source must be a ReasonSource enum, got {type(self.source).__name__}")
        if not isinstance(self.severity, ReasonSeverity):
            raise TypeError(f"severity must be a ReasonSeverity enum, got {type(self.severity).__name__}")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise TypeError(f"rank must be an integer, got {type(self.rank).__name__}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert reason code detail to a clean JSON-serializable dictionary."""
        return {
            "code": self.code,
            "headline": self.headline,
            "description": self.description,
            "category": self.category,
            "source": self.source.value,
            "severity": self.severity.value,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class WaterfallStep:
    """
    Mathematically verified step in the TreeSHAP margin waterfall visualization.

    Attributes:
        step_name: Display label for the step (e.g. "Base Value (Expected Margin)", feature name, or residual).
        feature_name: Name of the associated feature, or None for base/residual/final steps.
        contribution: Log-odds margin contribution of this step.
        cumulative_margin: Cumulative log-odds margin after applying this step.
        step_type: Semantic step type ("base", "feature", "residual", "final").
    """
    step_name: str
    feature_name: Optional[str]
    contribution: float
    cumulative_margin: float
    step_type: str

    def __post_init__(self) -> None:
        """Validate waterfall step types and finiteness."""
        if not isinstance(self.step_name, str) or not self.step_name.strip():
            raise ValueError(f"step_name must be a non-empty string, got {self.step_name!r}")
        if self.feature_name is not None and (not isinstance(self.feature_name, str) or not self.feature_name.strip()):
            raise ValueError(f"feature_name must be a non-empty string or None, got {self.feature_name!r}")
        if not isinstance(self.contribution, (int, float, np.floating)) or isinstance(self.contribution, bool):
            raise TypeError(f"contribution must be a float, got {type(self.contribution).__name__}")
        if not isinstance(self.cumulative_margin, (int, float, np.floating)) or isinstance(self.cumulative_margin, bool):
            raise TypeError(f"cumulative_margin must be a float, got {type(self.cumulative_margin).__name__}")
        if not math.isfinite(float(self.contribution)):
            raise ValueError(f"contribution must be finite, got {self.contribution}")
        if not math.isfinite(float(self.cumulative_margin)):
            raise ValueError(f"cumulative_margin must be finite, got {self.cumulative_margin}")
        if self.step_type not in ("base", "feature", "residual", "final"):
            raise ValueError(f"Invalid step_type '{self.step_type}'. Allowed: 'base', 'feature', 'residual', 'final'")

        object.__setattr__(self, "contribution", round(float(self.contribution), 6))
        object.__setattr__(self, "cumulative_margin", round(float(self.cumulative_margin), 6))

    def to_dict(self) -> Dict[str, Any]:
        """Convert waterfall step to a clean JSON-serializable dictionary."""
        return {
            "step_name": self.step_name,
            "feature_name": self.feature_name,
            "contribution": self.contribution,
            "cumulative_margin": self.cumulative_margin,
            "step_type": self.step_type,
        }


@dataclass(frozen=True)
class TransactionExplanation:
    """
    Comprehensive, immutable explanation payload for a single transaction.

    Attributes:
        model_score: Continuous ML fraud probability in [0.0, 1.0].
        output_margin: Raw model log-odds margin (logit of model_score).
        base_value: Global TreeSHAP expected margin across the training background.
        risk_score: Calibrated integer risk score in [0, 100].
        risk_tier: Semantic risk band (LOW, MEDIUM, HIGH, CRITICAL).
        action: Operational decision action (APPROVE, REVIEW, BLOCK).
        policy_mode: Operating policy mode (TRI_TIER or BINARY_AUTO).
        top_risk_factors: Immutable tuple of top positive (risk-increasing) feature attributions.
        top_mitigating_factors: Immutable tuple of top negative (risk-mitigating) feature attributions.
        reason_codes: Immutable tuple of combined human-readable reason codes (rules + model).
        waterfall: Immutable tuple of complete waterfall steps reconstructing output_margin.
        is_overridden: Whether a business rule overrode the baseline ML policy action.
        rule_action: Highest-precedence rule outcome enacted if an override occurred.
        rules_triggered: Immutable tuple of triggered business rule IDs.
        rule_matches: Immutable tuple of detailed RuleMatch objects.
        model_version: Provenance version of the champion model artifact if available.
    """
    model_score: float
    output_margin: float
    base_value: float
    risk_score: int
    risk_tier: RiskTier
    action: DecisionAction
    baseline_action: Optional[DecisionAction] = None
    policy_mode: PolicyMode = PolicyMode.TRI_TIER
    top_risk_factors: Tuple[FeatureAttribution, ...] = field(default_factory=tuple)
    top_mitigating_factors: Tuple[FeatureAttribution, ...] = field(default_factory=tuple)
    reason_codes: Tuple[ReasonCodeDetail, ...] = field(default_factory=tuple)
    waterfall: Tuple[WaterfallStep, ...] = field(default_factory=tuple)
    is_overridden: bool = False
    rule_action: Optional[RuleOutcome] = None
    rules_triggered: Tuple[str, ...] = field(default_factory=tuple)
    rule_matches: Tuple[RuleMatch, ...] = field(default_factory=tuple)
    model_version: Optional[str] = None

    def __post_init__(self) -> None:
        """Defensive validation and enforcement of deep immutability."""
        # Validate numerics
        for f, val in [
            ("model_score", self.model_score),
            ("output_margin", self.output_margin),
            ("base_value", self.base_value),
        ]:
            if not isinstance(val, (int, float, np.floating)) or isinstance(val, bool):
                raise TypeError(f"{f} must be a float, got {type(val).__name__}")
            if not math.isfinite(float(val)):
                raise ValueError(f"{f} must be finite, got {val}")

        if not isinstance(self.risk_score, int) or isinstance(self.risk_score, bool):
            raise TypeError(f"risk_score must be an integer, got {type(self.risk_score).__name__}")
        if self.risk_score < 0 or self.risk_score > 100:
            raise ValueError(f"risk_score must be in [0, 100], got {self.risk_score}")

        if not isinstance(self.risk_tier, RiskTier):
            raise TypeError(f"risk_tier must be a RiskTier enum, got {type(self.risk_tier).__name__}")
        if not isinstance(self.action, DecisionAction):
            raise TypeError(f"action must be a DecisionAction enum, got {type(self.action).__name__}")

        if self.baseline_action is None:
            object.__setattr__(self, "baseline_action", self.action)
        elif not isinstance(self.baseline_action, DecisionAction):
            raise TypeError(f"baseline_action must be a DecisionAction enum, got {type(self.baseline_action).__name__}")
        if not isinstance(self.policy_mode, PolicyMode):
            raise TypeError(f"policy_mode must be a PolicyMode enum, got {type(self.policy_mode).__name__}")
        if not isinstance(self.is_overridden, bool):
            raise TypeError(f"is_overridden must be a bool, got {type(self.is_overridden).__name__}")

        # Validate and enforce tuple immutability on collections
        object.__setattr__(self, "model_score", float(self.model_score))
        object.__setattr__(self, "output_margin", float(self.output_margin))
        object.__setattr__(self, "base_value", float(self.base_value))

        for f_name, item_type in [
            ("top_risk_factors", FeatureAttribution),
            ("top_mitigating_factors", FeatureAttribution),
            ("reason_codes", ReasonCodeDetail),
            ("waterfall", WaterfallStep),
            ("rule_matches", RuleMatch),
        ]:
            val = getattr(self, f_name)
            if not isinstance(val, (list, tuple)):
                raise TypeError(f"{f_name} must be a tuple or list, got {type(val).__name__}")
            for item in val:
                if not isinstance(item, item_type):
                    raise TypeError(f"All elements in {f_name} must be {item_type.__name__}, got {type(item).__name__}")
            object.__setattr__(self, f_name, tuple(val))

        # rules_triggered tuple of strings
        if not isinstance(self.rules_triggered, (list, tuple)):
            raise TypeError(f"rules_triggered must be a tuple or list, got {type(self.rules_triggered).__name__}")
        for r_id in self.rules_triggered:
            if not isinstance(r_id, str):
                raise TypeError(f"rules_triggered elements must be strings, got {type(r_id).__name__}")
        object.__setattr__(self, "rules_triggered", tuple(self.rules_triggered))

        # rule_action validation
        ra = self.rule_action
        if ra is not None:
            if isinstance(ra, str):
                try:
                    ra = RuleOutcome(ra)
                except ValueError:
                    raise ValueError(f"Invalid rule_action '{ra}'")
            elif not isinstance(ra, RuleOutcome):
                raise TypeError(f"rule_action must be a RuleOutcome enum, str, or None, got {type(ra).__name__}")
            object.__setattr__(self, "rule_action", ra)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction explanation to a clean JSON-serializable dictionary."""
        return {
            "model_score": self.model_score,
            "output_margin": self.output_margin,
            "base_value": self.base_value,
            "risk_score": self.risk_score,
            "risk_tier": self.risk_tier.value,
            "action": self.action.value,
            "baseline_action": self.baseline_action.value if self.baseline_action is not None else self.action.value,
            "policy_mode": self.policy_mode.value,
            "top_risk_factors": [f.to_dict() for f in self.top_risk_factors],
            "top_mitigating_factors": [f.to_dict() for f in self.top_mitigating_factors],
            "reason_codes": [r.to_dict() for r in self.reason_codes],
            "waterfall": [w.to_dict() for w in self.waterfall],
            "is_overridden": self.is_overridden,
            "rule_action": self.rule_action.value if self.rule_action is not None else None,
            "rules_triggered": list(self.rules_triggered),
            "rule_matches": [rm.to_dict() for rm in self.rule_matches],
            "model_version": self.model_version,
        }
