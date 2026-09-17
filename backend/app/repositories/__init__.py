"""
Repositories Package for Data Persistence Abstractions.

Exports:
- `TransactionRepository`: Asynchronous repository for `Transaction` persistence.
- `RiskEvaluationRepository`: Asynchronous repository for `RiskEvaluation` persistence.
- `RuleMatchRepository` / `EvaluationRuleMatchRepository`: Asynchronous repository for `EvaluationRuleMatch` persistence.
- `ReasonCodeRepository` / `EvaluationReasonCodeRepository`: Asynchronous repository for `EvaluationReasonCode` persistence.
- `FeatureAttributionRepository` / `EvaluationFeatureAttributionRepository`: Asynchronous repository for `EvaluationFeatureAttribution` persistence.
- `AuditLogRepository`: Asynchronous repository for append-oriented `AuditLog` persistence.
- `PersistenceError`: Base domain persistence exception.
- `PersistenceConflictError`: Constraint and conflict exception.
- `PersistenceNotFoundError`: Entity not found exception.
"""

from backend.app.repositories.audit_log_repository import AuditLogRepository
from backend.app.repositories.dashboard_repository import DashboardRepository
from backend.app.repositories.exceptions import (
    PersistenceConflictError,
    PersistenceError,
    PersistenceNotFoundError,
)
from backend.app.repositories.feature_attribution_repository import (
    EvaluationFeatureAttributionRepository,
    FeatureAttributionRepository,
)
from backend.app.repositories.reason_code_repository import (
    EvaluationReasonCodeRepository,
    ReasonCodeRepository,
)
from backend.app.repositories.risk_evaluation_repository import RiskEvaluationRepository
from backend.app.repositories.rule_match_repository import (
    EvaluationRuleMatchRepository,
    RuleMatchRepository,
)
from backend.app.repositories.transaction_repository import TransactionRepository

__all__ = [
    "DashboardRepository",
    "TransactionRepository",
    "RiskEvaluationRepository",
    "RuleMatchRepository",
    "EvaluationRuleMatchRepository",
    "ReasonCodeRepository",
    "EvaluationReasonCodeRepository",
    "FeatureAttributionRepository",
    "EvaluationFeatureAttributionRepository",
    "AuditLogRepository",
    "PersistenceError",
    "PersistenceConflictError",
    "PersistenceNotFoundError",
]

