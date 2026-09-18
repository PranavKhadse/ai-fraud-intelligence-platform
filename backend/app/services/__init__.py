"""
Services Package for FastAPI Fraud Detection & Risk Intelligence API.

Exports:
- `RiskService`, `get_risk_service`, `set_risk_service`: ML & Rule evaluation services.
- `FraudPersistenceUnitOfWork`, `UnitOfWork`: Unit of Work abstraction for transactional repository coordination.
- `FraudPersistenceService`: Application orchestration service for full evaluation aggregate persistence.
- `PersistRiskEvaluationCommand`, `PersistedRiskEvaluationResult`: Persistence command and result DTOs.
- `TransactionData`, `RiskEvaluationData`, `RuleMatchData`, `ReasonCodeData`, `FeatureAttributionData`, `AuditLogData`: Component DTOs.
"""

from backend.app.services.case_service import (
    ActorContext,
    CaseService,
    CreateManualCaseCommand,
    generate_case_number,
    get_case_service,
)
from backend.app.services.persistence_service import (
    AuditLogData,
    FeatureAttributionData,
    FraudPersistenceService,
    get_persistence_service,
    PersistRiskEvaluationCommand,
    PersistedRiskEvaluationResult,
    ReasonCodeData,
    RiskEvaluationData,
    RuleMatchData,
    TransactionData,
)
from backend.app.services.risk_persistence_mapper import (
    RiskEvaluationContext,
    RiskPersistenceMapper,
    compute_request_fingerprint,
    is_payload_equivalent,
    map_explanation_to_command,
    map_prediction_to_command,
    reconstruct_prediction_response,
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
    "get_persistence_service",
    "CaseService",
    "get_case_service",
    "ActorContext",
    "CreateManualCaseCommand",
    "generate_case_number",
    "PersistRiskEvaluationCommand",
    "PersistedRiskEvaluationResult",
    "TransactionData",
    "RiskEvaluationData",
    "RuleMatchData",
    "ReasonCodeData",
    "FeatureAttributionData",
    "AuditLogData",
    "RiskPersistenceMapper",
    "RiskEvaluationContext",
    "map_prediction_to_command",
    "map_explanation_to_command",
    "compute_request_fingerprint",
    "is_payload_equivalent",
    "reconstruct_prediction_response",
]
