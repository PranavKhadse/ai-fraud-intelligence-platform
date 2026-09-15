"""
Fraud Persistence Orchestration Service for Full Evaluation Aggregates.

Coordinates the persistence of Transactions, RiskEvaluations, EvaluationRuleMatches,
EvaluationReasonCodes, EvaluationFeatureAttributions, and AuditLogs under a single
Unit of Work transaction boundary.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import (
    AttributionDirection,
    AuditActorType,
    AuditEntityType,
    DecisionAction,
    PolicyMode,
    ReasonSeverity,
    ReasonSource,
    RiskTier,
    RuleOutcome,
    RuleType,
)
from backend.app.db.models.feature_attribution import EvaluationFeatureAttribution
from backend.app.db.models.reason_code import EvaluationReasonCode
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.rule_match import EvaluationRuleMatch
from backend.app.db.models.transaction import Transaction
from backend.app.repositories.exceptions import PersistenceConflictError
from backend.app.services.unit_of_work import FraudPersistenceUnitOfWork


# ==============================================================================
# Helpers
# ==============================================================================

def _ensure_datetime(val: Union[datetime, str, None]) -> datetime:
    """Helper to convert string/datetime to timezone-aware UTC datetime."""
    if val is None:
        return datetime.now(timezone.utc)
    if isinstance(val, str):
        dt = datetime.fromisoformat(val)
    else:
        dt = val
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_decimal(val: Union[Decimal, float, int, str, None], precision: Optional[int] = None) -> Optional[Decimal]:
    """Helper to safely convert numeric values to Decimal without binary floating point drift."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    if precision is not None and isinstance(val, float):
        return Decimal(str(round(val, precision)))
    return Decimal(str(val))


# ==============================================================================
# Input Contract / Command DTOs
# ==============================================================================

@dataclass(frozen=True)
class TransactionData:
    """Input contract for transaction persistence."""
    account_id: str
    merchant_category: str
    job_category: str
    amount: Union[Decimal, float, str]
    cardholder_lat: Union[Decimal, float, str]
    cardholder_long: Union[Decimal, float, str]
    merchant_lat: Union[Decimal, float, str]
    merchant_long: Union[Decimal, float, str]
    city_pop: int
    transaction_timestamp: Union[datetime, str]
    features_snapshot: Dict[str, Any]
    external_transaction_id: Optional[str] = None
    merchant_id: Optional[str] = None
    currency: str = "USD"


@dataclass(frozen=True)
class RiskEvaluationData:
    """Input contract for risk evaluation persistence."""
    model_version: str
    policy_mode: Union[PolicyMode, str]
    model_score: Union[Decimal, float, str]
    risk_score: int
    risk_tier: Union[RiskTier, str]
    decision_action: Union[DecisionAction, str]
    baseline_action: Optional[Union[DecisionAction, str]] = None
    is_overridden: bool = False
    rule_action: Optional[Union[RuleOutcome, str]] = None
    decision_reason: str = ""
    output_margin: Optional[Union[Decimal, float, str]] = None
    base_value: Optional[Union[Decimal, float, str]] = None
    evaluation_latency_ms: Optional[Union[Decimal, float, str]] = None
    correlation_id: Optional[str] = None
    evaluated_at: Optional[Union[datetime, str]] = None


@dataclass(frozen=True)
class RuleMatchData:
    """Input contract for evaluation rule match persistence."""
    rule_id: str
    description: str
    feature_name: str
    operator: str
    comparison_value: str
    outcome: Union[RuleOutcome, str]
    rule_type: Union[RuleType, str]
    priority: int


@dataclass(frozen=True)
class ReasonCodeData:
    """Input contract for evaluation reason code persistence."""
    code: str
    headline: str
    description: str
    category: str
    source: Union[ReasonSource, str]
    severity: Union[ReasonSeverity, str]
    rank: int


@dataclass(frozen=True)
class FeatureAttributionData:
    """Input contract for evaluation feature attribution persistence."""
    feature_name: str
    display_name: str
    shap_value: Union[Decimal, float, str]
    direction: Union[AttributionDirection, str]
    relative_contribution_pct: Union[Decimal, float, str]
    rank: int
    raw_value: Optional[Any] = None


@dataclass(frozen=True)
class AuditLogData:
    """Input contract for audit record creation."""
    event_type: str = "RISK_EVALUATION_PERSISTED"
    action: str = "PERSIST_EVALUATION"
    actor_type: Union[AuditActorType, str] = AuditActorType.SYSTEM
    actor_id: Optional[str] = "fraud_persistence_service"
    correlation_id: Optional[str] = None
    client_ip: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    event_timestamp: Optional[Union[datetime, str]] = None


@dataclass(frozen=True)
class PersistRiskEvaluationCommand:
    """
    Comprehensive command object encapsulating all entities for persisting
    a completed fraud-risk evaluation aggregate.
    """
    transaction: TransactionData
    evaluation: RiskEvaluationData
    rule_matches: Sequence[RuleMatchData] = field(default_factory=list)
    reason_codes: Sequence[ReasonCodeData] = field(default_factory=list)
    feature_attributions: Sequence[FeatureAttributionData] = field(default_factory=list)
    audit: Optional[AuditLogData] = None


@dataclass(frozen=True)
class PersistedRiskEvaluationResult:
    """
    Result payload returned after successful aggregate persistence.
    """
    transaction_id: uuid.UUID
    evaluation_id: uuid.UUID
    external_transaction_id: Optional[str]
    is_new_transaction: bool
    is_duplicate: bool
    rule_matches_count: int
    reason_codes_count: int
    feature_attributions_count: int
    audit_log_id: Optional[uuid.UUID]
    persisted_at: datetime


# ==============================================================================
# Service Implementation
# ==============================================================================

class FraudPersistenceService:
    """
    Application service orchestrating the atomic persistence of fraud evaluation aggregates.

    Design Principles:
    - Dependency Injection: Operates strictly through an injected `FraudPersistenceUnitOfWork`.
    - Idempotency & Conflict Policy: Rejects duplicate external transaction IDs with `PersistenceConflictError`.
    - Foreign-Key Safe Ordering: Stages entities in strict dependency order (Transaction -> Flush -> RiskEvaluation -> Flush -> Children -> AuditLog).
    - Single Commit: Commits exactly once via `uow.commit()` after all staging operations succeed.
    - Defensive Rollback: Automatically rolls back via `uow.rollback()` on any failure and propagates original exception.
    """

    def __init__(self, uow: FraudPersistenceUnitOfWork) -> None:
        """
        Initialize the persistence service with an injected Unit of Work.

        Args:
            uow: Injected FraudPersistenceUnitOfWork instance.

        Raises:
            ValueError: If uow is None.
        """
        if uow is None:
            raise ValueError("FraudPersistenceUnitOfWork must not be None.")
        self._uow = uow

    async def persist_evaluation(
        self,
        command: PersistRiskEvaluationCommand,
    ) -> PersistedRiskEvaluationResult:
        """
        Persist a complete fraud risk evaluation aggregate transactionally.

        Args:
            command: PersistRiskEvaluationCommand containing transaction, evaluation,
                     child explanation entities, and audit metadata.

        Returns:
            PersistedRiskEvaluationResult with generated entity identifiers and counts.

        Raises:
            PersistenceConflictError: If external_transaction_id already exists in the database.
            PersistenceError: If a database constraint or query failure occurs.
            Exception: Any unexpected exception is propagated without being swallowed.
        """
        try:
            # 1. Check external transaction ID for duplicate conflict
            ext_id = command.transaction.external_transaction_id
            if ext_id:
                exists = await self._uow.transactions.exists_by_external_id(ext_id)
                if exists:
                    raise PersistenceConflictError(
                        f"Transaction with external_transaction_id '{ext_id}' already exists.",
                        details={"external_transaction_id": ext_id},
                    )

            # 2. Map and stage Transaction
            tx_id = uuid.uuid4()
            tx_timestamp = _ensure_datetime(command.transaction.transaction_timestamp)
            tx_entity = Transaction(
                id=tx_id,
                external_transaction_id=ext_id,
                account_id=command.transaction.account_id,
                merchant_id=command.transaction.merchant_id,
                merchant_category=command.transaction.merchant_category,
                job_category=command.transaction.job_category,
                amount=Decimal(str(command.transaction.amount)),
                currency=command.transaction.currency,
                cardholder_lat=Decimal(str(command.transaction.cardholder_lat)),
                cardholder_long=Decimal(str(command.transaction.cardholder_long)),
                merchant_lat=Decimal(str(command.transaction.merchant_lat)),
                merchant_long=Decimal(str(command.transaction.merchant_long)),
                city_pop=int(command.transaction.city_pop),
                transaction_timestamp=tx_timestamp,
                features_snapshot=command.transaction.features_snapshot,
            )
            await self._uow.transactions.add(tx_entity)

            # 3. Flush to ensure transaction is materialized in session
            await self._uow.flush()

            # 4. Map and stage RiskEvaluation
            eval_id = uuid.uuid4()
            eval_timestamp = _ensure_datetime(command.evaluation.evaluated_at)
            policy_mode = (
                PolicyMode(command.evaluation.policy_mode)
                if isinstance(command.evaluation.policy_mode, str)
                else command.evaluation.policy_mode
            )
            risk_tier = (
                RiskTier(command.evaluation.risk_tier)
                if isinstance(command.evaluation.risk_tier, str)
                else command.evaluation.risk_tier
            )
            decision_action = (
                DecisionAction(command.evaluation.decision_action)
                if isinstance(command.evaluation.decision_action, str)
                else command.evaluation.decision_action
            )
            baseline_action = (
                DecisionAction(command.evaluation.baseline_action)
                if isinstance(command.evaluation.baseline_action, str)
                else (command.evaluation.baseline_action or decision_action)
            )
            rule_action = (
                RuleOutcome(command.evaluation.rule_action)
                if isinstance(command.evaluation.rule_action, str)
                else command.evaluation.rule_action
            )

            eval_entity = RiskEvaluation(
                id=eval_id,
                transaction_id=tx_entity.id,
                model_version=command.evaluation.model_version,
                policy_mode=policy_mode,
                model_score=Decimal(str(round(float(command.evaluation.model_score), 6))),
                risk_score=int(command.evaluation.risk_score),
                risk_tier=risk_tier,
                decision_action=decision_action,
                baseline_action=baseline_action,
                is_overridden=command.evaluation.is_overridden,
                rule_action=rule_action,
                decision_reason=command.evaluation.decision_reason,
                output_margin=_to_decimal(command.evaluation.output_margin, 6),
                base_value=_to_decimal(command.evaluation.base_value, 6),
                evaluation_latency_ms=_to_decimal(command.evaluation.evaluation_latency_ms, 2),
                correlation_id=command.evaluation.correlation_id,
                evaluated_at=eval_timestamp,
            )
            await self._uow.risk_evaluations.add(eval_entity)

            # 5. Flush to ensure RiskEvaluation is materialized in session
            await self._uow.flush()

            # 6. Map and stage EvaluationRuleMatch records
            rule_match_entities: List[EvaluationRuleMatch] = []
            for rm in command.rule_matches:
                rm_outcome = RuleOutcome(rm.outcome) if isinstance(rm.outcome, str) else rm.outcome
                rm_type = RuleType(rm.rule_type) if isinstance(rm.rule_type, str) else rm.rule_type
                rule_match_entities.append(
                    EvaluationRuleMatch(
                        id=uuid.uuid4(),
                        evaluation_id=eval_entity.id,
                        rule_id=rm.rule_id,
                        description=rm.description,
                        feature_name=rm.feature_name,
                        operator=str(rm.operator),
                        comparison_value=str(rm.comparison_value),
                        outcome=rm_outcome,
                        rule_type=rm_type,
                        priority=int(rm.priority),
                    )
                )
            if rule_match_entities:
                await self._uow.rule_matches.add_many(rule_match_entities)

            # 7. Map and stage EvaluationReasonCode records
            reason_code_entities: List[EvaluationReasonCode] = []
            for rc in command.reason_codes:
                rc_source = ReasonSource(rc.source) if isinstance(rc.source, str) else rc.source
                rc_severity = ReasonSeverity(rc.severity) if isinstance(rc.severity, str) else rc.severity
                reason_code_entities.append(
                    EvaluationReasonCode(
                        id=uuid.uuid4(),
                        evaluation_id=eval_entity.id,
                        code=rc.code,
                        headline=rc.headline,
                        description=rc.description,
                        category=rc.category,
                        source=rc_source,
                        severity=rc_severity,
                        rank=int(rc.rank),
                    )
                )
            if reason_code_entities:
                await self._uow.reason_codes.add_many(reason_code_entities)

            # 8. Map and stage EvaluationFeatureAttribution records
            feature_attr_entities: List[EvaluationFeatureAttribution] = []
            for fa in command.feature_attributions:
                fa_direction = (
                    AttributionDirection(fa.direction)
                    if isinstance(fa.direction, str)
                    else fa.direction
                )
                feature_attr_entities.append(
                    EvaluationFeatureAttribution(
                        id=uuid.uuid4(),
                        evaluation_id=eval_entity.id,
                        feature_name=fa.feature_name,
                        display_name=fa.display_name,
                        raw_value=fa.raw_value,
                        shap_value=Decimal(str(round(float(fa.shap_value), 6))),
                        direction=fa_direction,
                        relative_contribution_pct=Decimal(str(round(float(fa.relative_contribution_pct), 4))),
                        rank=int(fa.rank),
                    )
                )
            if feature_attr_entities:
                await self._uow.feature_attributions.add_many(feature_attr_entities)

            # 9. Map and stage AuditLog record
            audit_data = command.audit or AuditLogData(
                correlation_id=command.evaluation.correlation_id,
            )
            audit_actor_type = (
                AuditActorType(audit_data.actor_type)
                if isinstance(audit_data.actor_type, str)
                else audit_data.actor_type
            )
            audit_ts = _ensure_datetime(audit_data.event_timestamp)
            audit_log_id = uuid.uuid4()
            audit_entity = AuditLog(
                id=audit_log_id,
                event_type=audit_data.event_type,
                entity_type=AuditEntityType.RISK_EVALUATION,
                entity_id=eval_entity.id,
                action=audit_data.action,
                actor_type=audit_actor_type,
                actor_id=audit_data.actor_id,
                correlation_id=audit_data.correlation_id or command.evaluation.correlation_id,
                client_ip=audit_data.client_ip,
                payload=audit_data.payload or {
                    "external_transaction_id": ext_id,
                    "model_score": float(eval_entity.model_score),
                    "risk_score": eval_entity.risk_score,
                    "decision_action": eval_entity.decision_action.value,
                },
                event_timestamp=audit_ts,
            )
            await self._uow.audit_logs.add(audit_entity)

            # 10. Commit all staged entities in a single atomic transaction
            await self._uow.commit()

            # 11. Return typed result
            return PersistedRiskEvaluationResult(
                transaction_id=tx_entity.id,
                evaluation_id=eval_entity.id,
                external_transaction_id=ext_id,
                is_new_transaction=True,
                is_duplicate=False,
                rule_matches_count=len(rule_match_entities),
                reason_codes_count=len(reason_code_entities),
                feature_attributions_count=len(feature_attr_entities),
                audit_log_id=audit_log_id,
                persisted_at=datetime.now(timezone.utc),
            )

        except Exception:
            # Execute defensive rollback via Unit of Work and re-raise
            await self._uow.rollback()
            raise


__all__ = [
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
