"""
Persistence Enumerations for SQLAlchemy ORM Models.

Reuses existing domain enum concepts to ensure complete interoperability across
the Risk Engine, Explainability, and Persistence layers.
"""

from enum import Enum

from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    RuleOutcome,
    RuleType,
)
from ml.explainability.schemas import (
    AttributionDirection,
    ReasonSource,
    ReasonSeverity,
)


class AuditActorType(str, Enum):
    """Origin/actor responsible for an audit log event."""
    SYSTEM = "SYSTEM"
    ANALYST = "ANALYST"
    ADMIN = "ADMIN"
    API_CLIENT = "API_CLIENT"


class AuditEntityType(str, Enum):
    """Categorization of entities tracked in the audit log."""
    TRANSACTION = "TRANSACTION"
    RISK_EVALUATION = "RISK_EVALUATION"
    POLICY = "POLICY"
    SYSTEM = "SYSTEM"
    CASE = "CASE"


class CaseStatus(str, Enum):
    """Lifecycle status of a human review case."""
    OPEN = "OPEN"
    IN_REVIEW = "IN_REVIEW"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class CasePriority(str, Enum):
    """Queue triage priority of a human review case."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CaseDisposition(str, Enum):
    """Authoritative human review outcome for a case."""
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    LEGITIMATE = "LEGITIMATE"
    SUSPICIOUS_RESOLVED = "SUSPICIOUS_RESOLVED"


class CaseTriggerSource(str, Enum):
    """Origin mechanism responsible for opening a case."""
    AUTOMATED_REVIEW_POLICY = "AUTOMATED_REVIEW_POLICY"
    AUTOMATED_RULE_OVERRIDE = "AUTOMATED_RULE_OVERRIDE"
    MANUAL_ANALYST_ESCALATION = "MANUAL_ANALYST_ESCALATION"


class CaseNoteType(str, Enum):
    """Categorization of investigation notes."""
    INVESTIGATION = "INVESTIGATION"
    ESCALATION = "ESCALATION"
    DISPOSITION = "DISPOSITION"
    SYSTEM_AUDIT = "SYSTEM_AUDIT"


__all__ = [
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
    "CaseStatus",
    "CasePriority",
    "CaseDisposition",
    "CaseTriggerSource",
    "CaseNoteType",
]
