"""
Services Package for FastAPI Fraud Detection & Risk Intelligence API.

Exports:
- `RiskService`, `get_risk_service`, `set_risk_service`: ML & Rule evaluation services.
- `FraudPersistenceUnitOfWork`, `UnitOfWork`: Unit of Work abstraction for transactional repository coordination.
- `FraudPersistenceService`: Application orchestration service for full evaluation aggregate persistence.
- `PersistRiskEvaluationCommand`, `PersistedRiskEvaluationResult`: Persistence command and result DTOs.
- `TransactionData`, `RiskEvaluationData`, `RuleMatchData`, `ReasonCodeData`, `FeatureAttributionData`, `AuditLogData`: Component DTOs.
"""

from backend.app.services.persistence_service import (
    AuditLogData,
    FeatureAttributionData,
    FraudPersistenceService,
    PersistRiskEvaluationCommand,
    PersistedRiskEvaluationResult,
    ReasonCodeData,
    RiskEvaluationData,
    RuleMatchData,
    TransactionData,
)
from backend.app.services.risk_service import (
    RiskService,
    get_risk_service,
    set_risk_service,
)
from backend.app.services.unit_of_work import (
    FraudPersistenceUnitOfWork,
    UnitOfWork,
)

__all__ = [
    "RiskService",
    "get_risk_service",
    "set_risk_service",
    "FraudPersistenceUnitOfWork",
    "UnitOfWork",
    "FraudPersistenceService",
    "PersistRiskEvaluationCommand",
    "PersistedRiskEvaluationResult",
    "TransactionData",
    "RiskEvaluationData",
    "RuleMatchData",
    "ReasonCodeData",
    "FeatureAttributionData",
    "AuditLogData",
]
