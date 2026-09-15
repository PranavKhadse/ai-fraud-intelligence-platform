"""
Unit Tests for Phase 9 Milestone 9.4 Increment 2: Supporting Persistence Repositories.

Validates:
- Public repository exports and imports.
- Session injection and boundary neutrality (no commit/rollback/close).
- Operations and query structure for:
  * RiskEvaluationRepository
  * RuleMatchRepository / EvaluationRuleMatchRepository
  * ReasonCodeRepository / EvaluationReasonCodeRepository
  * FeatureAttributionRepository / EvaluationFeatureAttributionRepository
  * AuditLogRepository
- Deterministic query ordering and statement generation.
- Empty list batch handling (add_many([])).
- Error translation and exception chaining.
- Immutability and append-oriented design of AuditLogRepository.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import Select

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
from backend.app.repositories import (
    AuditLogRepository,
    EvaluationFeatureAttributionRepository,
    EvaluationReasonCodeRepository,
    EvaluationRuleMatchRepository,
    FeatureAttributionRepository,
    PersistenceError,
    ReasonCodeRepository,
    RiskEvaluationRepository,
    RuleMatchRepository,
    TransactionRepository,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_session():
    """Mock AsyncSession for verifying transaction-boundary neutrality."""
    session = MagicMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def sample_evaluation():
    return RiskEvaluation(
        id=uuid.uuid4(),
        transaction_id=uuid.uuid4(),
        model_version="1.0.0",
        policy_mode=PolicyMode.TRI_TIER,
        model_score=Decimal("0.850000"),
        risk_score=85,
        risk_tier=RiskTier.HIGH,
        decision_action=DecisionAction.REVIEW,
        baseline_action=DecisionAction.REVIEW,
        is_overridden=False,
        rule_action=None,
        decision_reason="Score in high risk range",
        output_margin=Decimal("1.734600"),
        base_value=Decimal("-1.500000"),
        evaluation_latency_ms=Decimal("12.50"),
        correlation_id="corr-12345",
        evaluated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_rule_match():
    return EvaluationRuleMatch(
        id=uuid.uuid4(),
        evaluation_id=uuid.uuid4(),
        rule_id="RULE_GEO_VELOCITY",
        description="High geographical velocity between transactions",
        feature_name="geo_velocity_kmh",
        operator=">",
        comparison_value="800",
        outcome=RuleOutcome.REVIEW,
        rule_type=RuleType.VELOCITY,
        priority=10,
    )


@pytest.fixture
def sample_reason_code():
    return EvaluationReasonCode(
        id=uuid.uuid4(),
        evaluation_id=uuid.uuid4(),
        code="RC_HIGH_AMOUNT",
        headline="Unusually High Transaction Amount",
        description="Transaction amount deviates significantly from account profile",
        category="AMOUNT",
        source=ReasonSource.MODEL,
        severity=ReasonSeverity.HIGH,
        rank=1,
    )


@pytest.fixture
def sample_feature_attribution():
    return EvaluationFeatureAttribution(
        id=uuid.uuid4(),
        evaluation_id=uuid.uuid4(),
        feature_name="amt",
        display_name="Transaction Amount",
        raw_value=1500.00,
        shap_value=Decimal("0.345000"),
        direction=AttributionDirection.RISK_INCREASING,
        relative_contribution_pct=Decimal("0.4500"),
        rank=1,
    )


@pytest.fixture
def sample_audit_log():
    return AuditLog(
        id=uuid.uuid4(),
        event_type="PREDICTION_EXECUTED",
        entity_type=AuditEntityType.RISK_EVALUATION,
        entity_id=uuid.uuid4(),
        action="EVALUATE",
        actor_type=AuditActorType.SYSTEM,
        actor_id="risk_service",
        correlation_id="corr-12345",
        event_timestamp=datetime.now(timezone.utc),
    )


# ==============================================================================
# Common Repository Rules & Imports
# ==============================================================================

class TestPublicExportsAndCommonRules:
    """Validates public package exports and common architectural rules."""

    def test_all_repositories_exported(self):
        assert RiskEvaluationRepository is not None
        assert RuleMatchRepository is not None
        assert EvaluationRuleMatchRepository is RuleMatchRepository
        assert ReasonCodeRepository is not None
        assert EvaluationReasonCodeRepository is ReasonCodeRepository
        assert FeatureAttributionRepository is not None
        assert EvaluationFeatureAttributionRepository is FeatureAttributionRepository
        assert AuditLogRepository is not None
        assert TransactionRepository is not None

    @pytest.mark.parametrize(
        "repo_class",
        [
            RiskEvaluationRepository,
            RuleMatchRepository,
            ReasonCodeRepository,
            FeatureAttributionRepository,
            AuditLogRepository,
        ],
    )
    def test_repository_accepts_injected_session_and_creates_no_engine(self, repo_class, mock_session):
        with patch("backend.app.db.session.get_async_engine") as mock_engine:
            repo = repo_class(mock_session)
            assert repo._session is mock_session
            mock_engine.assert_not_called()


# ==============================================================================
# RiskEvaluationRepository Tests
# ==============================================================================

class TestRiskEvaluationRepository:
    """Validates RiskEvaluation persistence operations."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session, sample_evaluation):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_evaluation
        mock_session.execute.return_value = mock_result

        repo = RiskEvaluationRepository(mock_session)
        result = await repo.get_by_id(sample_evaluation.id)

        assert result is sample_evaluation
        mock_session.execute.assert_awaited_once()
        stmt = mock_session.execute.call_args[0][0]
        assert isinstance(stmt, Select)

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = RiskEvaluationRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_transaction_id_returns_latest(self, mock_session, sample_evaluation):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_evaluation
        mock_session.execute.return_value = mock_result

        repo = RiskEvaluationRepository(mock_session)
        result = await repo.get_by_transaction_id(sample_evaluation.transaction_id)

        assert result is sample_evaluation
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_by_transaction_id_returns_deterministic_list(self, mock_session, sample_evaluation):
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_evaluation]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = RiskEvaluationRepository(mock_session)
        results = await repo.list_by_transaction_id(sample_evaluation.transaction_id)

        assert results == [sample_evaluation]
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_attaches_and_returns_entity(self, mock_session, sample_evaluation):
        repo = RiskEvaluationRepository(mock_session)
        result = await repo.add(sample_evaluation)

        assert result is sample_evaluation
        mock_session.add.assert_called_once_with(sample_evaluation)
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()
        mock_session.flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_failure_raises_persistence_error(self, mock_session):
        mock_session.execute.side_effect = SQLAlchemyError("Query execution failed")

        repo = RiskEvaluationRepository(mock_session)
        with pytest.raises(PersistenceError) as exc_info:
            await repo.get_by_id(uuid.uuid4())

        assert "Failed to retrieve risk evaluation by ID" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, SQLAlchemyError)


# ==============================================================================
# RuleMatchRepository Tests
# ==============================================================================

class TestRuleMatchRepository:
    """Validates EvaluationRuleMatch persistence operations."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session, sample_rule_match):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_rule_match
        mock_session.execute.return_value = mock_result

        repo = RuleMatchRepository(mock_session)
        result = await repo.get_by_id(sample_rule_match.id)

        assert result is sample_rule_match
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = RuleMatchRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_evaluation_id_deterministic_order(self, mock_session, sample_rule_match):
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_rule_match]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = RuleMatchRepository(mock_session)
        results = await repo.list_by_evaluation_id(sample_rule_match.evaluation_id)

        assert results == [sample_rule_match]
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_single(self, mock_session, sample_rule_match):
        repo = RuleMatchRepository(mock_session)
        result = await repo.add(sample_rule_match)

        assert result is sample_rule_match
        mock_session.add.assert_called_once_with(sample_rule_match)
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_many_multiple_items(self, mock_session, sample_rule_match):
        second_match = EvaluationRuleMatch(
            id=uuid.uuid4(),
            evaluation_id=sample_rule_match.evaluation_id,
            rule_id="RULE_AMOUNT_HIGH",
            description="Amount above normal tier",
            feature_name="amt",
            operator=">",
            comparison_value="1000",
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.AMOUNT,
            priority=20,
        )
        items = [sample_rule_match, second_match]

        repo = RuleMatchRepository(mock_session)
        results = await repo.add_many(items)

        assert results == items
        mock_session.add_all.assert_called_once_with(items)
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_many_empty_list_no_db_operation(self, mock_session):
        repo = RuleMatchRepository(mock_session)
        results = await repo.add_many([])

        assert results == []
        mock_session.add_all.assert_not_called()
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_failure_raises_persistence_error(self, mock_session):
        mock_session.execute.side_effect = SQLAlchemyError("Database query failed")

        repo = RuleMatchRepository(mock_session)
        with pytest.raises(PersistenceError) as exc_info:
            await repo.list_by_evaluation_id(uuid.uuid4())

        assert "Failed to list rule matches" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, SQLAlchemyError)


# ==============================================================================
# ReasonCodeRepository Tests
# ==============================================================================

class TestReasonCodeRepository:
    """Validates EvaluationReasonCode persistence operations."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session, sample_reason_code):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_reason_code
        mock_session.execute.return_value = mock_result

        repo = ReasonCodeRepository(mock_session)
        result = await repo.get_by_id(sample_reason_code.id)

        assert result is sample_reason_code

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = ReasonCodeRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_evaluation_id(self, mock_session, sample_reason_code):
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_reason_code]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = ReasonCodeRepository(mock_session)
        results = await repo.list_by_evaluation_id(sample_reason_code.evaluation_id)

        assert results == [sample_reason_code]

    @pytest.mark.asyncio
    async def test_add_single(self, mock_session, sample_reason_code):
        repo = ReasonCodeRepository(mock_session)
        result = await repo.add(sample_reason_code)

        assert result is sample_reason_code
        mock_session.add.assert_called_once_with(sample_reason_code)
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_many_multiple_items(self, mock_session, sample_reason_code):
        second_reason = EvaluationReasonCode(
            id=uuid.uuid4(),
            evaluation_id=sample_reason_code.evaluation_id,
            code="RC_VELOCITY",
            headline="High Velocity Burst",
            description="Rapid transaction burst",
            category="VELOCITY",
            source=ReasonSource.RULE,
            severity=ReasonSeverity.MEDIUM,
            rank=2,
        )
        items = [sample_reason_code, second_reason]

        repo = ReasonCodeRepository(mock_session)
        results = await repo.add_many(items)

        assert results == items
        mock_session.add_all.assert_called_once_with(items)
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_many_empty_list_no_db_operation(self, mock_session):
        repo = ReasonCodeRepository(mock_session)
        results = await repo.add_many([])

        assert results == []
        mock_session.add_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_failure_raises_persistence_error(self, mock_session):
        mock_session.execute.side_effect = SQLAlchemyError("DB error")

        repo = ReasonCodeRepository(mock_session)
        with pytest.raises(PersistenceError) as exc_info:
            await repo.get_by_id(uuid.uuid4())

        assert "Failed to retrieve reason code" in str(exc_info.value)


# ==============================================================================
# FeatureAttributionRepository Tests
# ==============================================================================

class TestFeatureAttributionRepository:
    """Validates EvaluationFeatureAttribution persistence operations."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session, sample_feature_attribution):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_feature_attribution
        mock_session.execute.return_value = mock_result

        repo = FeatureAttributionRepository(mock_session)
        result = await repo.get_by_id(sample_feature_attribution.id)

        assert result is sample_feature_attribution

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = FeatureAttributionRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_evaluation_id(self, mock_session, sample_feature_attribution):
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_feature_attribution]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = FeatureAttributionRepository(mock_session)
        results = await repo.list_by_evaluation_id(sample_feature_attribution.evaluation_id)

        assert results == [sample_feature_attribution]

    @pytest.mark.asyncio
    async def test_add_single(self, mock_session, sample_feature_attribution):
        repo = FeatureAttributionRepository(mock_session)
        result = await repo.add(sample_feature_attribution)

        assert result is sample_feature_attribution
        mock_session.add.assert_called_once_with(sample_feature_attribution)
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_many_multiple_items(self, mock_session, sample_feature_attribution):
        second_attr = EvaluationFeatureAttribution(
            id=uuid.uuid4(),
            evaluation_id=sample_feature_attribution.evaluation_id,
            feature_name="city_pop",
            display_name="City Population",
            raw_value=500000,
            shap_value=Decimal("-0.120000"),
            direction=AttributionDirection.MITIGATING,
            relative_contribution_pct=Decimal("0.2500"),
            rank=2,
        )
        items = [sample_feature_attribution, second_attr]

        repo = FeatureAttributionRepository(mock_session)
        results = await repo.add_many(items)

        assert results == items
        mock_session.add_all.assert_called_once_with(items)
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_many_empty_list_no_db_operation(self, mock_session):
        repo = FeatureAttributionRepository(mock_session)
        results = await repo.add_many([])

        assert results == []
        mock_session.add_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_failure_raises_persistence_error(self, mock_session):
        mock_session.execute.side_effect = SQLAlchemyError("Attribution read failure")

        repo = FeatureAttributionRepository(mock_session)
        with pytest.raises(PersistenceError) as exc_info:
            await repo.list_by_evaluation_id(uuid.uuid4())

        assert "Failed to list feature attributions" in str(exc_info.value)


# ==============================================================================
# AuditLogRepository Tests
# ==============================================================================

class TestAuditLogRepository:
    """Validates AuditLog append-oriented persistence operations."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session, sample_audit_log):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_audit_log
        mock_session.execute.return_value = mock_result

        repo = AuditLogRepository(mock_session)
        result = await repo.get_by_id(sample_audit_log.id)

        assert result is sample_audit_log

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = AuditLogRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_entity_with_enum(self, mock_session, sample_audit_log):
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_audit_log]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = AuditLogRepository(mock_session)
        results = await repo.list_by_entity(
            AuditEntityType.RISK_EVALUATION,
            entity_id=sample_audit_log.entity_id,
            limit=10,
        )

        assert results == [sample_audit_log]
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_by_entity_with_string(self, mock_session, sample_audit_log):
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_audit_log]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = AuditLogRepository(mock_session)
        results = await repo.list_by_entity("RISK_EVALUATION")

        assert results == [sample_audit_log]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_limit", [0, -1, -100])
    async def test_list_by_entity_invalid_limit_raises_value_error(self, mock_session, invalid_limit):
        repo = AuditLogRepository(mock_session)
        with pytest.raises(ValueError) as exc_info:
            await repo.list_by_entity(AuditEntityType.SYSTEM, limit=invalid_limit)

        assert "limit must be a positive integer" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_add_single(self, mock_session, sample_audit_log):
        repo = AuditLogRepository(mock_session)
        result = await repo.add(sample_audit_log)

        assert result is sample_audit_log
        mock_session.add.assert_called_once_with(sample_audit_log)
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_many_multiple_records(self, mock_session, sample_audit_log):
        second_log = AuditLog(
            id=uuid.uuid4(),
            event_type="RULE_MATCH_RECORDED",
            entity_type=AuditEntityType.RISK_EVALUATION,
            entity_id=sample_audit_log.entity_id,
            action="CREATE",
            actor_type=AuditActorType.SYSTEM,
            actor_id="rule_engine",
            event_timestamp=datetime.now(timezone.utc),
        )
        items = [sample_audit_log, second_log]

        repo = AuditLogRepository(mock_session)
        results = await repo.add_many(items)

        assert results == items
        mock_session.add_all.assert_called_once_with(items)
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_many_empty_list_no_db_operation(self, mock_session):
        repo = AuditLogRepository(mock_session)
        results = await repo.add_many([])

        assert results == []
        mock_session.add_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_update_or_delete_methods_exposed(self):
        """Validates that repository layer preserves append-oriented semantics."""
        repo = AuditLogRepository(MagicMock(spec=AsyncSession))
        assert not hasattr(repo, "update")
        assert not hasattr(repo, "delete")
        assert not hasattr(repo, "remove")
        assert not hasattr(repo, "delete_by_id")

    @pytest.mark.asyncio
    async def test_query_failure_raises_persistence_error(self, mock_session):
        mock_session.execute.side_effect = SQLAlchemyError("Audit log table locked")

        repo = AuditLogRepository(mock_session)
        with pytest.raises(PersistenceError) as exc_info:
            await repo.list_by_entity(AuditEntityType.SYSTEM)

        assert "Failed to list audit logs" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, SQLAlchemyError)
