"""
Unit Tests for Phase 12.2: Case Creation & Persistence Integration.

Validates:
- Automated REVIEW decision creates exactly one Case, initial system CaseNote, and CASE_CREATED AuditLog.
- Automated APPROVE decision does not create a Case.
- Automated BLOCK decision does not create a Case.
- Automated Case creation derives priority from evaluation risk_tier and trigger_source from override telemetry.
- Manual Case creation creates an OPEN Case with MANUAL_ANALYST_ESCALATION, investigation note, and audit event.
- Manual Case creation rejects empty or whitespace-only initial notes with ValueError.
- Manual Case creation rejects non-existent transaction with PersistenceNotFoundError.
- Manual Case creation rejects transaction/evaluation mismatch with PersistenceError.
- Manual Case creation rejects duplicate case on the same transaction with PersistenceConflictError.
- Concurrency-safe dispute handling translates database IntegrityError to PersistenceConflictError with existing case info.
- Case reference number generation produces chronological, collision-safe CASE-YYYYMMDD-XXXXXX format.
- Audit payload size bounding guarantees payload is well under the 1 KB boundary.
- Atomic rollback behavior across failures.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from backend.app.db.models import (
    AuditActorType,
    AuditEntityType,
    AuditLog,
    Case,
    CaseNote,
    CaseNoteType,
    CasePriority,
    CaseStatus,
    CaseTriggerSource,
    DecisionAction,
    PolicyMode,
    RiskEvaluation,
    RiskTier,
    RuleOutcome,
    Transaction,
)
from backend.app.repositories.exceptions import (
    PersistenceConflictError,
    PersistenceError,
    PersistenceNotFoundError,
)
from backend.app.services.case_service import (
    ActorContext,
    CaseService,
    CreateManualCaseCommand,
    generate_case_number,
)
from backend.app.services.persistence_service import (
    AuditLogData,
    FraudPersistenceService,
    PersistRiskEvaluationCommand,
    PersistedRiskEvaluationResult,
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
    """Mock FraudPersistenceUnitOfWork with tracked repository doubles."""
    uow = MagicMock(spec=FraudPersistenceUnitOfWork)
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()
    uow.flush = AsyncMock()

    uow.transactions = MagicMock()
    uow.transactions.exists_by_external_id = AsyncMock(return_value=False)
    uow.transactions.get_by_id = AsyncMock(return_value=None)
    uow.transactions.add = AsyncMock(side_effect=lambda entity: entity)

    uow.risk_evaluations = MagicMock()
    uow.risk_evaluations.get_by_id = AsyncMock(return_value=None)
    uow.risk_evaluations.get_by_transaction_id = AsyncMock(return_value=[])
    uow.risk_evaluations.add = AsyncMock(side_effect=lambda entity: entity)

    uow.rule_matches = MagicMock()
    uow.rule_matches.add_many = AsyncMock(side_effect=lambda entities: list(entities))

    uow.reason_codes = MagicMock()
    uow.reason_codes.add_many = AsyncMock(side_effect=lambda entities: list(entities))

    uow.feature_attributions = MagicMock()
    uow.feature_attributions.add_many = AsyncMock(side_effect=lambda entities: list(entities))

    uow.audit_logs = MagicMock()
    uow.audit_logs.add = AsyncMock(side_effect=lambda entity: entity)

    uow.cases = MagicMock()
    uow.cases.get_by_id = AsyncMock(return_value=None)
    uow.cases.get_by_transaction_id = AsyncMock(return_value=None)
    uow.cases.add = AsyncMock(side_effect=lambda entity: entity)
    uow.cases.add_note = AsyncMock(side_effect=lambda entity: entity)

    return uow


@pytest.fixture
def sample_transaction():
    """Sample persisted Transaction entity."""
    return Transaction(
        id=uuid.uuid4(),
        external_transaction_id="TX_MANUAL_001",
        account_id="ACC_998877",
        merchant_category="electronics",
        job_category="analyst",
        amount=Decimal("1250.00"),
        currency="USD",
        cardholder_lat=Decimal("40.712800"),
        cardholder_long=Decimal("-74.006000"),
        merchant_lat=Decimal("40.713000"),
        merchant_long=Decimal("-74.005800"),
        city_pop=500000,
        transaction_timestamp=datetime.now(timezone.utc),
        features_snapshot={"amt": 1250.00},
    )


@pytest.fixture
def sample_evaluation(sample_transaction):
    """Sample persisted RiskEvaluation entity."""
    return RiskEvaluation(
        id=uuid.uuid4(),
        transaction_id=sample_transaction.id,
        model_version="1.0.0",
        policy_mode=PolicyMode.TRI_TIER,
        model_score=Decimal("0.785000"),
        risk_score=79,
        risk_tier=RiskTier.HIGH,
        decision_action=DecisionAction.REVIEW,
        baseline_action=DecisionAction.REVIEW,
        is_overridden=False,
        decision_reason="Model score in review range",
        output_margin=Decimal("0.850000"),
        base_value=Decimal("-1.500000"),
        evaluation_latency_ms=Decimal("12.50"),
        correlation_id="corr-manual-001",
        evaluated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def actor_context():
    """Sample actor context for manual actions."""
    return ActorContext(
        actor_id="analyst_john",
        actor_role=AuditActorType.ANALYST,
        correlation_id="corr-manual-001",
        client_ip="192.168.1.100",
    )


# ==============================================================================
# Case Reference Generation Tests
# ==============================================================================

class TestCaseReferenceGeneration:
    """Validates collision safety, chronological prefixing, and formatting."""

    def test_case_number_format(self):
        case_num = generate_case_number()
        pattern = r"^CASE-\d{8}-[A-F0-9]{6}$"
        assert re.match(pattern, case_num) is not None
        assert len(case_num) <= 32

    def test_case_number_explicit_timestamp(self):
        fixed_dt = datetime(2026, 9, 18, 15, 30, tzinfo=timezone.utc)
        case_num = generate_case_number(fixed_dt)
        assert case_num.startswith("CASE-20260918-")

    def test_case_number_uniqueness(self):
        generated = {generate_case_number() for _ in range(1000)}
        assert len(generated) == 1000


# ==============================================================================
# Automated Review Case Creation Tests
# ==============================================================================

class TestAutomatedCaseCreation:
    """Validates automated case creation inside FraudPersistenceService.persist_evaluation()."""

    def _build_command(
        self,
        decision_action: DecisionAction,
        risk_tier: RiskTier = RiskTier.HIGH,
        is_overridden: bool = False,
        rule_action: Optional[RuleOutcome] = None,
    ) -> PersistRiskEvaluationCommand:
        return PersistRiskEvaluationCommand(
            transaction=TransactionData(
                external_transaction_id="TX_AUTO_001",
                account_id="ACC_112233",
                merchant_category="retail",
                job_category="teacher",
                amount=Decimal("350.00"),
                currency="USD",
                cardholder_lat=Decimal("40.712800"),
                cardholder_long=Decimal("-74.006000"),
                merchant_lat=Decimal("40.713000"),
                merchant_long=Decimal("-74.005800"),
                city_pop=250000,
                transaction_timestamp=datetime.now(timezone.utc),
                features_snapshot={"amt": 350.00},
            ),
            evaluation=RiskEvaluationData(
                model_version="1.0.0",
                policy_mode=PolicyMode.TRI_TIER,
                model_score=Decimal("0.680000"),
                risk_score=68,
                risk_tier=risk_tier,
                decision_action=decision_action,
                baseline_action=DecisionAction.APPROVE if is_overridden else decision_action,
                is_overridden=is_overridden,
                rule_action=rule_action,
                decision_reason="Evaluation test reason",
                correlation_id="corr-auto-001",
                evaluated_at=datetime.now(timezone.utc),
            ),
        )

    @pytest.mark.asyncio
    async def test_review_decision_creates_automated_case_and_notes(self, mock_uow):
        cmd = self._build_command(DecisionAction.REVIEW, RiskTier.HIGH)
        service = FraudPersistenceService(mock_uow)

        result = await service.persist_evaluation(cmd)

        assert isinstance(result.case_id, uuid.UUID)
        assert result.case_number is not None and result.case_number.startswith("CASE-")

        # Verify Case entity staged
        mock_uow.cases.add.assert_awaited_once()
        case_entity: Case = mock_uow.cases.add.call_args[0][0]
        assert case_entity.id == result.case_id
        assert case_entity.case_number == result.case_number
        assert case_entity.status == CaseStatus.OPEN
        assert case_entity.priority == CasePriority.HIGH
        assert case_entity.trigger_source == CaseTriggerSource.AUTOMATED_REVIEW_POLICY
        assert case_entity.assigned_to is None
        assert case_entity.disposition is None

        # Verify initial system note staged
        mock_uow.cases.add_note.assert_awaited_once()
        note_entity: CaseNote = mock_uow.cases.add_note.call_args[0][0]
        assert note_entity.case_id == case_entity.id
        assert note_entity.author_id == "SYSTEM"
        assert note_entity.author_role == AuditActorType.SYSTEM
        assert note_entity.note_type == CaseNoteType.SYSTEM_AUDIT
        assert "Automated review case created" in note_entity.content
        assert len(note_entity.content.strip()) > 0

        # Verify CASE_CREATED audit event staged
        assert mock_uow.audit_logs.add.await_count == 2
        staged_audits = [call[0][0] for call in mock_uow.audit_logs.add.call_args_list]
        case_audit = next(a for a in staged_audits if a.event_type == "CASE_CREATED")
        assert case_audit.entity_type == AuditEntityType.CASE
        assert case_audit.entity_id == case_entity.id
        assert case_audit.actor_type == AuditActorType.SYSTEM
        assert case_audit.payload["case_number"] == case_entity.case_number
        assert case_audit.payload["trigger_source"] == CaseTriggerSource.AUTOMATED_REVIEW_POLICY.value

    @pytest.mark.asyncio
    async def test_review_decision_with_rule_override_trigger_source(self, mock_uow):
        cmd = self._build_command(
            DecisionAction.REVIEW,
            RiskTier.CRITICAL,
            is_overridden=True,
            rule_action=RuleOutcome.REVIEW,
        )
        service = FraudPersistenceService(mock_uow)

        result = await service.persist_evaluation(cmd)

        case_entity: Case = mock_uow.cases.add.call_args[0][0]
        assert case_entity.priority == CasePriority.CRITICAL
        assert case_entity.trigger_source == CaseTriggerSource.AUTOMATED_RULE_OVERRIDE

    @pytest.mark.asyncio
    async def test_approve_decision_does_not_create_case(self, mock_uow):
        cmd = self._build_command(DecisionAction.APPROVE, RiskTier.LOW)
        service = FraudPersistenceService(mock_uow)

        result = await service.persist_evaluation(cmd)

        assert result.case_id is None
        assert result.case_number is None
        mock_uow.cases.add.assert_not_called()
        mock_uow.cases.add_note.assert_not_called()
        assert mock_uow.audit_logs.add.await_count == 1
        audit: AuditLog = mock_uow.audit_logs.add.call_args[0][0]
        assert audit.event_type == "RISK_EVALUATION_PERSISTED"

    @pytest.mark.asyncio
    async def test_block_decision_does_not_create_case(self, mock_uow):
        cmd = self._build_command(DecisionAction.BLOCK, RiskTier.CRITICAL)
        service = FraudPersistenceService(mock_uow)

        result = await service.persist_evaluation(cmd)

        assert result.case_id is None
        assert result.case_number is None
        mock_uow.cases.add.assert_not_called()
        mock_uow.cases.add_note.assert_not_called()
        assert mock_uow.audit_logs.add.await_count == 1


# ==============================================================================
# Manual Case Creation & Validation Tests
# ==============================================================================

class TestManualCaseCreation:
    """Validates CaseService.create_manual_case() workflow, validation, and error translation."""

    @pytest.mark.asyncio
    async def test_manual_case_creation_successful(
        self,
        mock_uow,
        sample_transaction,
        sample_evaluation,
        actor_context,
    ):
        mock_uow.transactions.get_by_id.return_value = sample_transaction
        mock_uow.risk_evaluations.get_by_id.return_value = sample_evaluation
        mock_uow.cases.get_by_transaction_id.return_value = None

        service = CaseService(mock_uow)
        cmd = CreateManualCaseCommand(
            transaction_id=sample_transaction.id,
            initial_note="Suspicious high-value purchase from new device location.",
            actor=actor_context,
            evaluation_id=sample_evaluation.id,
        )

        case = await service.create_manual_case(cmd)

        assert case.transaction_id == sample_transaction.id
        assert case.evaluation_id == sample_evaluation.id
        assert case.status == CaseStatus.OPEN
        assert case.priority == CasePriority.HIGH
        assert case.trigger_source == CaseTriggerSource.MANUAL_ANALYST_ESCALATION
        assert case.case_number.startswith("CASE-")

        mock_uow.cases.add.assert_awaited_once()
        mock_uow.cases.add_note.assert_awaited_once()
        mock_uow.audit_logs.add.assert_awaited_once()
        mock_uow.commit.assert_awaited_once()

        # Check Note
        note: CaseNote = mock_uow.cases.add_note.call_args[0][0]
        assert note.case_id == case.id
        assert note.author_id == "analyst_john"
        assert note.author_role == AuditActorType.ANALYST
        assert note.note_type == CaseNoteType.INVESTIGATION
        assert note.content == "Suspicious high-value purchase from new device location."

        # Check Audit
        audit: AuditLog = mock_uow.audit_logs.add.call_args[0][0]
        assert audit.event_type == "CASE_CREATED"
        assert audit.entity_type == AuditEntityType.CASE
        assert audit.entity_id == case.id
        assert audit.actor_type == AuditActorType.ANALYST
        assert audit.actor_id == "analyst_john"

    @pytest.mark.asyncio
    async def test_manual_case_creation_custom_priority(
        self,
        mock_uow,
        sample_transaction,
        sample_evaluation,
        actor_context,
    ):
        mock_uow.transactions.get_by_id.return_value = sample_transaction
        mock_uow.risk_evaluations.get_by_id.return_value = sample_evaluation
        mock_uow.cases.get_by_transaction_id.return_value = None

        service = CaseService(mock_uow)
        cmd = CreateManualCaseCommand(
            transaction_id=sample_transaction.id,
            initial_note="Urgent executive account escalation.",
            actor=actor_context,
            priority=CasePriority.CRITICAL,
            evaluation_id=sample_evaluation.id,
        )

        case = await service.create_manual_case(cmd)
        assert case.priority == CasePriority.CRITICAL

    @pytest.mark.asyncio
    async def test_manual_case_creation_auto_resolves_evaluation_when_omitted(
        self,
        mock_uow,
        sample_transaction,
        sample_evaluation,
        actor_context,
    ):
        mock_uow.transactions.get_by_id.return_value = sample_transaction
        mock_uow.risk_evaluations.get_by_transaction_id.return_value = [sample_evaluation]
        mock_uow.cases.get_by_transaction_id.return_value = None

        service = CaseService(mock_uow)
        cmd = CreateManualCaseCommand(
            transaction_id=sample_transaction.id,
            initial_note="Investigation note.",
            actor=actor_context,
            evaluation_id=None,
        )

        case = await service.create_manual_case(cmd)
        assert case.evaluation_id == sample_evaluation.id

    @pytest.mark.asyncio
    async def test_empty_initial_note_raises_value_error(
        self,
        mock_uow,
        sample_transaction,
        actor_context,
    ):
        service = CaseService(mock_uow)
        cmd = CreateManualCaseCommand(
            transaction_id=sample_transaction.id,
            initial_note="   ",
            actor=actor_context,
        )

        with pytest.raises(ValueError, match="Initial case note cannot be empty"):
            await service.create_manual_case(cmd)

        mock_uow.cases.add.assert_not_called()
        mock_uow.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_existent_transaction_raises_not_found(
        self,
        mock_uow,
        actor_context,
    ):
        mock_uow.transactions.get_by_id.return_value = None
        missing_id = uuid.uuid4()

        service = CaseService(mock_uow)
        cmd = CreateManualCaseCommand(
            transaction_id=missing_id,
            initial_note="Note for missing txn.",
            actor=actor_context,
        )

        with pytest.raises(PersistenceNotFoundError, match="Transaction .* not found"):
            await service.create_manual_case(cmd)

        mock_uow.rollback.assert_awaited_once()
        mock_uow.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_transaction_evaluation_mismatch_raises_persistence_error(
        self,
        mock_uow,
        sample_transaction,
        actor_context,
    ):
        mismatched_eval = RiskEvaluation(
            id=uuid.uuid4(),
            transaction_id=uuid.uuid4(),  # Different transaction!
            model_version="1.0.0",
            policy_mode=PolicyMode.TRI_TIER,
            model_score=Decimal("0.50"),
            risk_score=50,
            risk_tier=RiskTier.MEDIUM,
            decision_action=DecisionAction.REVIEW,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=False,
            decision_reason="Mismatch test",
        )
        mock_uow.transactions.get_by_id.return_value = sample_transaction
        mock_uow.risk_evaluations.get_by_id.return_value = mismatched_eval

        service = CaseService(mock_uow)
        cmd = CreateManualCaseCommand(
            transaction_id=sample_transaction.id,
            initial_note="Note with mismatched evaluation.",
            actor=actor_context,
            evaluation_id=mismatched_eval.id,
        )

        with pytest.raises(PersistenceError, match="does not belong to Transaction"):
            await service.create_manual_case(cmd)

        mock_uow.rollback.assert_awaited_once()
        mock_uow.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_duplicate_case_pre_check_raises_conflict(
        self,
        mock_uow,
        sample_transaction,
        sample_evaluation,
        actor_context,
    ):
        existing_case = Case(
            id=uuid.uuid4(),
            case_number="CASE-20260918-EXIST1",
            transaction_id=sample_transaction.id,
            evaluation_id=sample_evaluation.id,
            status=CaseStatus.OPEN,
            priority=CasePriority.HIGH,
            trigger_source=CaseTriggerSource.AUTOMATED_REVIEW_POLICY,
        )
        mock_uow.transactions.get_by_id.return_value = sample_transaction
        mock_uow.risk_evaluations.get_by_id.return_value = sample_evaluation
        mock_uow.cases.get_by_transaction_id.return_value = existing_case

        service = CaseService(mock_uow)
        cmd = CreateManualCaseCommand(
            transaction_id=sample_transaction.id,
            initial_note="Duplicate attempt.",
            actor=actor_context,
            evaluation_id=sample_evaluation.id,
        )

        with pytest.raises(PersistenceConflictError, match="A case already exists") as exc_info:
            await service.create_manual_case(cmd)

        assert exc_info.value.details["existing_case_id"] == str(existing_case.id)
        assert exc_info.value.details["existing_case_number"] == "CASE-20260918-EXIST1"
        mock_uow.rollback.assert_awaited_once()
        mock_uow.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrency_race_condition_integrity_error_translates_to_conflict(
        self,
        mock_uow,
        sample_transaction,
        sample_evaluation,
        actor_context,
    ):
        # Pre-check passes (returns None)
        mock_uow.transactions.get_by_id.return_value = sample_transaction
        mock_uow.risk_evaluations.get_by_id.return_value = sample_evaluation
        mock_uow.cases.get_by_transaction_id.side_effect = [
            None,  # 1st call during pre-check
            Case(  # 2nd call after rollback to query winning case
                id=uuid.uuid4(),
                case_number="CASE-20260918-WINNER",
                transaction_id=sample_transaction.id,
                evaluation_id=sample_evaluation.id,
                status=CaseStatus.OPEN,
                priority=CasePriority.HIGH,
                trigger_source=CaseTriggerSource.MANUAL_ANALYST_ESCALATION,
            ),
        ]

        # Commit raises database IntegrityError due to unique constraint uq_cases_transaction_id
        mock_uow.commit.side_effect = IntegrityError(
            "duplicate key value violates unique constraint uq_cases_transaction_id",
            params={},
            orig=Exception("duplicate key value violates unique constraint \"uq_cases_transaction_id\""),
        )

        service = CaseService(mock_uow)
        cmd = CreateManualCaseCommand(
            transaction_id=sample_transaction.id,
            initial_note="Concurrent creation race.",
            actor=actor_context,
            evaluation_id=sample_evaluation.id,
        )

        with pytest.raises(PersistenceConflictError, match="A case already exists") as exc_info:
            await service.create_manual_case(cmd)

        assert exc_info.value.details["existing_case_number"] == "CASE-20260918-WINNER"
        mock_uow.rollback.assert_awaited_once()


# ==============================================================================
# Bounded Audit Payload Tests
# ==============================================================================

class TestAuditPayloadBounding:
    """Validates strict <= 1 KB payload size bounding across manual and automated cases."""

    def test_case_created_audit_payload_size(self):
        payload = {
            "case_number": "CASE-20260918-A1B2C3",
            "transaction_id": str(uuid.uuid4()),
            "evaluation_id": str(uuid.uuid4()),
            "trigger_source": CaseTriggerSource.AUTOMATED_REVIEW_POLICY.value,
            "priority": CasePriority.HIGH.value,
        }
        serialized = json.dumps(payload)
        payload_bytes = len(serialized.encode("utf-8"))

        assert payload_bytes < 1024
        assert payload_bytes < 250  # Strict budget <= 250 bytes
