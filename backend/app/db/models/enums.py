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
]
