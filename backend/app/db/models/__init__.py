"""
SQLAlchemy ORM Models Package for the AI-Powered Fraud Detection & Risk Intelligence Platform.

Exports all core declarative models, mixins, and enumerations.
"""

from backend.app.db.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin
from backend.app.db.models.enums import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    RuleOutcome,
    RuleType,
    AttributionDirection,
    ReasonSource,
    ReasonSeverity,
    AuditActorType,
    AuditEntityType,
)
from backend.app.db.models.transaction import Transaction
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.rule_match import EvaluationRuleMatch
from backend.app.db.models.reason_code import EvaluationReasonCode
from backend.app.db.models.feature_attribution import EvaluationFeatureAttribution
from backend.app.db.models.audit_log import AuditLog

__all__ = [
    # Base and Mixins
    "Base",
    "UUIDPrimaryKeyMixin",
    "TimestampMixin",
    # Enums
    "DecisionAction",
    "RiskTier",
    "PolicyMode",
    "RuleOutcome",
    "RuleType",
    "AttributionDirection",
    "ReasonSource",
    "ReasonSeverity",
    "AuditActorType",
    "AuditEntityType",
    # Models
    "Transaction",
    "RiskEvaluation",
    "EvaluationRuleMatch",
    "EvaluationReasonCode",
    "EvaluationFeatureAttribution",
    "AuditLog",
]
