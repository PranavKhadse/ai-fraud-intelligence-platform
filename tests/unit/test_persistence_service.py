"""
Unit Tests for Phase 9 Milestone 9.5 Increment 2: Persistence Orchestration Service.

Validates:
- Dependency injection and constructor validation.
- Successful transactional persistence of the entire fraud evaluation aggregate.
- Foreign-key safe execution ordering (duplicate check -> tx -> flush -> eval -> flush -> children -> audit -> commit).
- Duplicate external transaction ID conflict handling (PersistenceConflictError).
- Failure handling and defensive rollback at each stage (tx, eval, children, audit, commit).
- Data type preservation (Decimals, UUIDs, timezone-aware datetimes, feature snapshot).
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import pytest
from sqlalchemy.exc import SQLAlchemyError

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
from backend.app.repositories.exceptions import PersistenceConflictError, PersistenceError
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
from backend.app.services.unit_of_work import FraudPersistenceUnitOfWork


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_uow():
    """Mock FraudPersistenceUnitOfWork with tracked method calls."""
    uow = MagicMock(spec=FraudPersistenceUnitOfWork)
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()
    uow.flush = AsyncMock()

    # Mock repositories
    uow.transactions = MagicMock()
    uow.transactions.exists_by_external_id = AsyncMock(return_value=False)
    uow.transactions.add = AsyncMock(side_effect=lambda entity: entity)

    uow.risk_evaluations = MagicMock()
    uow.risk_evaluations.add = AsyncMock(side_effect=lambda entity: entity)

    uow.rule_matches = MagicMock()
    uow.rule_matches.add_many = AsyncMock(side_effect=lambda entities: list(entities))

    uow.reason_codes = MagicMock()
    uow.reason_codes.add_many = AsyncMock(side_effect=lambda entities: list(entities))

    uow.feature_attributions = MagicMock()
    uow.feature_attributions.add_many = AsyncMock(side_effect=lambda entities: list(entities))

    uow.audit_logs = MagicMock()
    uow.audit_logs.add = AsyncMock(side_effect=lambda entity: entity)

    return uow


@pytest.fixture
def sample_command():
    """Valid PersistRiskEvaluationCommand containing full 55-feature vector and explanations."""
    return PersistRiskEvaluationCommand(
        transaction=TransactionData(
            external_transaction_id="TX_EXT_999",
            account_id="ACC_12345",
            merchant_id="MERCH_67890",
            merchant_category="grocery_pos",
            job_category="engineer",
            amount=Decimal("150.75"),
            currency="USD",
            cardholder_lat=Decimal("40.712800"),
            cardholder_long=Decimal("-74.006000"),
            merchant_lat=Decimal("40.713000"),
            merchant_long=Decimal("-74.005800"),
            city_pop=500000,
            transaction_timestamp=datetime.now(timezone.utc),
            features_snapshot={"amt": 150.75, "city_pop": 500000, "txn_count_1h": 2},
        ),
        evaluation=RiskEvaluationData(
            model_version="1.0.0",
            policy_mode=PolicyMode.TRI_TIER,
            model_score=Decimal("0.650000"),
            risk_score=65,
            risk_tier=RiskTier.HIGH,
            decision_action=DecisionAction.REVIEW,
            baseline_action=DecisionAction.APPROVE,
            is_overridden=True,
            rule_action=RuleOutcome.REVIEW,
            decision_reason="Rule match escalated decision to REVIEW",
            output_margin=Decimal("0.619000"),
            base_value=Decimal("-1.500000"),
            evaluation_latency_ms=Decimal("15.20"),
            correlation_id="corr-test-123",
            evaluated_at=datetime.now(timezone.utc),
        ),
        rule_matches=[
            RuleMatchData(
                rule_id="RULE_VELOCITY_BURST",
                description="Rapid transaction count in past hour",
                feature_name="txn_count_1h",
                operator=">",
                comparison_value="5",
                outcome=RuleOutcome.REVIEW,
                rule_type=RuleType.VELOCITY,
                priority=10,
            )
        ],
        reason_codes=[
            ReasonCodeData(
                code="RC_RAPID_TXN",
                headline="High Transaction Frequency",
                description="Unusually frequent transactions within 1 hour",
                category="VELOCITY",
                source=ReasonSource.RULE,
                severity=ReasonSeverity.HIGH,
                rank=1,
            )
        ],
        feature_attributions=[
            FeatureAttributionData(
                feature_name="amt",
                display_name="Transaction Amount",
                raw_value=150.75,
                shap_value=Decimal("0.450000"),
                direction=AttributionDirection.RISK_INCREASING,
                relative_contribution_pct=Decimal("0.5500"),
                rank=1,
            )
        ],
        audit=AuditLogData(
            event_type="RISK_EVALUATION_PERSISTED",
            action="PERSIST_EVALUATION",
            actor_type=AuditActorType.SYSTEM,
            actor_id="test_runner",
            correlation_id="corr-test-123",
        ),
    )


# ==============================================================================
# Dependency Injection & Validation Tests
# ==============================================================================

class TestFraudPersistenceServiceInstantiation:
    """Validates service constructor and injection rules."""

    def test_service_accepts_unit_of_work(self, mock_uow):
        service = FraudPersistenceService(mock_uow)
        assert service._uow is mock_uow

    def test_service_rejects_none_unit_of_work(self):
        with pytest.raises(ValueError, match="FraudPersistenceUnitOfWork must not be None"):
            FraudPersistenceService(None)  # type: ignore[arg-type]

    def test_service_does_not_create_engine_or_session(self, mock_uow):
        with patch("backend.app.db.session.get_async_engine") as mock_engine:
            FraudPersistenceService(mock_uow)
            mock_engine.assert_not_called()


# ==============================================================================
# Successful Persistence & Ordering Tests
# ==============================================================================

class TestFraudPersistenceServiceExecution:
    """Validates aggregate staging, ordering, and atomic commit."""

    @pytest.mark.asyncio
    async def test_persist_evaluation_successful(self, mock_uow, sample_command):
        service = FraudPersistenceService(mock_uow)
        result = await service.persist_evaluation(sample_command)

        # 1. Returned result verification
        assert isinstance(result, PersistedRiskEvaluationResult)
        assert isinstance(result.transaction_id, uuid.UUID)
        assert isinstance(result.evaluation_id, uuid.UUID)
        assert result.external_transaction_id == "TX_EXT_999"
        assert result.is_new_transaction is True
        assert result.is_duplicate is False
        assert result.rule_matches_count == 1
        assert result.reason_codes_count == 1
        assert result.feature_attributions_count == 1
        assert isinstance(result.audit_log_id, uuid.UUID)

        # 2. Duplicate check called
        mock_uow.transactions.exists_by_external_id.assert_awaited_once_with("TX_EXT_999")

        # 3. Entities staged
        mock_uow.transactions.add.assert_awaited_once()
        mock_uow.risk_evaluations.add.assert_awaited_once()
        mock_uow.rule_matches.add_many.assert_awaited_once()
        mock_uow.reason_codes.add_many.assert_awaited_once()
        mock_uow.feature_attributions.add_many.assert_awaited_once()
        mock_uow.audit_logs.add.assert_awaited_once()

        # 4. Flushes and single commit
        assert mock_uow.flush.await_count == 2
        mock_uow.commit.assert_awaited_once()
        mock_uow.rollback.assert_not_called()

        # 5. Entity relationship verification
        added_tx: Transaction = mock_uow.transactions.add.call_args[0][0]
        added_eval: RiskEvaluation = mock_uow.risk_evaluations.add.call_args[0][0]
        assert added_eval.transaction_id == added_tx.id
        assert added_tx.amount == Decimal("150.75")
        assert added_eval.model_score == Decimal("0.650000")

    @pytest.mark.asyncio
    async def test_persistence_execution_order(self, mock_uow, sample_command):
        """Validates that operations execute in strict foreign-key safe order."""
        call_log: List[str] = []

        mock_uow.transactions.exists_by_external_id = AsyncMock(
            side_effect=lambda *args: call_log.append("duplicate_check") or False
        )
        mock_uow.transactions.add = AsyncMock(
            side_effect=lambda entity: call_log.append("tx_add") or entity
        )
        mock_uow.flush = AsyncMock(
            side_effect=lambda: call_log.append("flush")
        )
        mock_uow.risk_evaluations.add = AsyncMock(
            side_effect=lambda entity: call_log.append("eval_add") or entity
        )
        mock_uow.rule_matches.add_many = AsyncMock(
            side_effect=lambda entities: call_log.append("rule_matches_add") or list(entities)
        )
        mock_uow.reason_codes.add_many = AsyncMock(
            side_effect=lambda entities: call_log.append("reason_codes_add") or list(entities)
        )
        mock_uow.feature_attributions.add_many = AsyncMock(
            side_effect=lambda entities: call_log.append("attributions_add") or list(entities)
        )
        mock_uow.audit_logs.add = AsyncMock(
            side_effect=lambda entity: call_log.append("audit_add") or entity
        )
        mock_uow.commit = AsyncMock(
            side_effect=lambda: call_log.append("commit")
        )

        service = FraudPersistenceService(mock_uow)
        await service.persist_evaluation(sample_command)

        expected_order = [
            "duplicate_check",
            "tx_add",
            "flush",
            "eval_add",
            "flush",
            "rule_matches_add",
            "reason_codes_add",
            "attributions_add",
            "audit_add",
            "commit",
        ]
        assert call_log == expected_order


# ==============================================================================
# Duplicate Handling Tests
# ==============================================================================

class TestFraudPersistenceServiceDuplicateHandling:
    """Validates conflict detection and rollback on duplicate external transaction IDs."""

    @pytest.mark.asyncio
    async def test_duplicate_external_transaction_id_raises_conflict_error(self, mock_uow, sample_command):
        mock_uow.transactions.exists_by_external_id.return_value = True

        service = FraudPersistenceService(mock_uow)
        with pytest.raises(PersistenceConflictError, match="already exists") as exc_info:
            await service.persist_evaluation(sample_command)

        assert exc_info.value.details == {"external_transaction_id": "TX_EXT_999"}
        mock_uow.transactions.add.assert_not_called()
        mock_uow.risk_evaluations.add.assert_not_called()
        mock_uow.commit.assert_not_called()
        mock_uow.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_none_external_transaction_id_skips_duplicate_check(self, mock_uow, sample_command):
        # When external_transaction_id is None, duplicate check should be skipped
        tx_data = TransactionData(
            account_id="ACC_12345",
            merchant_category="grocery_pos",
            job_category="engineer",
            amount=Decimal("150.75"),
            cardholder_lat=Decimal("40.712800"),
            cardholder_long=Decimal("-74.006000"),
            merchant_lat=Decimal("40.713000"),
            merchant_long=Decimal("-74.005800"),
            city_pop=500000,
            transaction_timestamp=datetime.now(timezone.utc),
            features_snapshot={"amt": 150.75},
            external_transaction_id=None,
        )
        command = PersistRiskEvaluationCommand(
            transaction=tx_data,
            evaluation=sample_command.evaluation,
        )

        service = FraudPersistenceService(mock_uow)
        result = await service.persist_evaluation(command)

        assert result.external_transaction_id is None
        mock_uow.transactions.exists_by_external_id.assert_not_called()
        mock_uow.commit.assert_awaited_once()


# ==============================================================================
# Failure & Rollback Tests
# ==============================================================================

class TestFraudPersistenceServiceFailures:
    """Validates failure handling and defensive rollback across stages."""

    @pytest.mark.asyncio
    async def test_transaction_add_failure_triggers_rollback(self, mock_uow, sample_command):
        mock_uow.transactions.add.side_effect = PersistenceError("DB write failed")

        service = FraudPersistenceService(mock_uow)
        with pytest.raises(PersistenceError, match="DB write failed"):
            await service.persist_evaluation(sample_command)

        mock_uow.commit.assert_not_called()
        mock_uow.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_risk_evaluation_add_failure_triggers_rollback(self, mock_uow, sample_command):
        mock_uow.risk_evaluations.add.side_effect = PersistenceError("Eval insert error")

        service = FraudPersistenceService(mock_uow)
        with pytest.raises(PersistenceError, match="Eval insert error"):
            await service.persist_evaluation(sample_command)

        mock_uow.commit.assert_not_called()
        mock_uow.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rule_match_add_failure_triggers_rollback(self, mock_uow, sample_command):
        mock_uow.rule_matches.add_many.side_effect = PersistenceError("Child rule error")

        service = FraudPersistenceService(mock_uow)
        with pytest.raises(PersistenceError, match="Child rule error"):
            await service.persist_evaluation(sample_command)

        mock_uow.commit.assert_not_called()
        mock_uow.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_audit_log_add_failure_triggers_rollback(self, mock_uow, sample_command):
        mock_uow.audit_logs.add.side_effect = PersistenceError("Audit write error")

        service = FraudPersistenceService(mock_uow)
        with pytest.raises(PersistenceError, match="Audit write error"):
            await service.persist_evaluation(sample_command)

        mock_uow.commit.assert_not_called()
        mock_uow.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_commit_failure_triggers_rollback(self, mock_uow, sample_command):
        mock_uow.commit.side_effect = SQLAlchemyError("Deadlock detected")

        service = FraudPersistenceService(mock_uow)
        with pytest.raises(SQLAlchemyError, match="Deadlock detected"):
            await service.persist_evaluation(sample_command)

        mock_uow.rollback.assert_awaited_once()


# ==============================================================================
# Data Preservation Tests
# ==============================================================================

class TestFraudPersistenceServiceDataPreservation:
    """Validates preservation of Decimals, timestamps, and explanation contracts."""

    @pytest.mark.asyncio
    async def test_decimal_precision_and_strings_preserved(self, mock_uow, sample_command):
        service = FraudPersistenceService(mock_uow)
        await service.persist_evaluation(sample_command)

        tx: Transaction = mock_uow.transactions.add.call_args[0][0]
        assert isinstance(tx.amount, Decimal)
        assert tx.amount == Decimal("150.75")
        assert tx.cardholder_lat == Decimal("40.712800")
        assert tx.cardholder_long == Decimal("-74.006000")

        eval_entity: RiskEvaluation = mock_uow.risk_evaluations.add.call_args[0][0]
        assert isinstance(eval_entity.model_score, Decimal)
        assert eval_entity.model_score == Decimal("0.650000")
        assert eval_entity.risk_score == 65
        assert eval_entity.policy_mode == PolicyMode.TRI_TIER
        assert eval_entity.decision_action == DecisionAction.REVIEW
        assert eval_entity.is_overridden is True

        rules: List[EvaluationRuleMatch] = mock_uow.rule_matches.add_many.call_args[0][0]
        assert len(rules) == 1
        assert rules[0].rule_id == "RULE_VELOCITY_BURST"
        assert rules[0].outcome == RuleOutcome.REVIEW

        reasons: List[EvaluationReasonCode] = mock_uow.reason_codes.add_many.call_args[0][0]
        assert len(reasons) == 1
        assert reasons[0].code == "RC_RAPID_TXN"

        attrs: List[EvaluationFeatureAttribution] = mock_uow.feature_attributions.add_many.call_args[0][0]
        assert len(attrs) == 1
        assert attrs[0].feature_name == "amt"
        assert attrs[0].shap_value == Decimal("0.450000")
        assert attrs[0].direction == AttributionDirection.RISK_INCREASING

        audit: AuditLog = mock_uow.audit_logs.add.call_args[0][0]
        assert audit.event_type == "RISK_EVALUATION_PERSISTED"
        assert audit.entity_type == AuditEntityType.RISK_EVALUATION
        assert audit.entity_id == eval_entity.id
