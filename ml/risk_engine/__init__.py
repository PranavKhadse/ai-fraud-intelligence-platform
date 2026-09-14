"""
Risk Engine & Decision Framework Package.

Exposes core configuration, normalization functions, semantic risk tiers,
the decision policy engine, the end-to-end RiskEvaluator, and batch summaries.
"""

from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    DecisionPolicyConfig,
    DecisionReasonCode,
    RuleType,
    RuleOutcome,
    RuleOperator,
)
from ml.risk_engine.normalization import (
    normalize_model_score,
    normalize_model_scores,
    risk_tier_from_score,
)
from ml.risk_engine.policy import (
    DecisionResult,
    DecisionPolicyEngine,
)
from ml.risk_engine.evaluator import (
    RiskEvaluator,
    BatchDecisionSummary,
)
from ml.risk_engine.rules import (
    RiskRule,
    RuleMatch,
    RuleEngine,
)
from ml.risk_engine.catalog import (
    STANDARD_RULES,
    get_standard_rule_catalog,
)

__all__ = [
    "DecisionAction",
    "RiskTier",
    "PolicyMode",
    "DecisionPolicyConfig",
    "DecisionReasonCode",
    "RuleType",
    "RuleOutcome",
    "RuleOperator",
    "normalize_model_score",
    "normalize_model_scores",
    "risk_tier_from_score",
    "DecisionResult",
    "DecisionPolicyEngine",
    "RiskEvaluator",
    "BatchDecisionSummary",
    "RiskRule",
    "RuleMatch",
    "RuleEngine",
    "STANDARD_RULES",
    "get_standard_rule_catalog",
]
