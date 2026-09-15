"""
Integration Tests for Phase 9.5 Increment 3: Persistence Orchestration Layer.

Verifies the complete persistence workflow against a real PostgreSQL database instance:
1. Actual database inserts of full evaluation aggregate.
2. Cross-session visibility of committed aggregates.
3. Conflict rejection on duplicate external_transaction_id.
4. Transactional rollback integrity on child insert failures.
5. Foreign-key constraint enforcement and RESTRICT deletion rules.
6. Database check constraints on numeric bounds.
7. Fidelity of JSON/JSONB snapshots, Decimals, UTC datetimes, and Enums.
8. Unit of Work transaction boundaries and uncommitted state isolation.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch
import uuid
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

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

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ==============================================================================
# Sample Aggregate Helper
# ==============================================================================

def create_sample_persistence_command(
    external_id: str = "TX_INT_001",
    amount: Decimal = Decimal("249.99"),
) -> PersistRiskEvaluationCommand:
    """Helper constructing a complete, valid persistence command with all 55 features."""
    # 55-feature snapshot representation
    snapshot_55 = {
        "amount": float(amount),
        "cardholder_lat": 40.7128,
        "cardholder_long": -74.0060,
        "merchant_lat": 40.7130,
        "merchant_long": -74.0058,
        "city_pop": 500000,
        "merchant_category": "grocery_pos",
        "job_category": "engineer",
        "transaction_hour": 14,
        "day_of_week": 2,
        "day_of_month": 15,
        "month": 9,
        "week_of_year": 38,
        "is_weekend": 0,
        "is_night": 0,
        "hour_sin": 0.5,
        "hour_cos": -0.866,
        "day_of_week_sin": 0.9749,
        "day_of_week_cos": -0.2225,
        "txn_count_1h": 3.0,
        "txn_count_6h": 5.0,
        "txn_count_24h": 8.0,
        "txn_count_7d": 20.0,
        "txn_count_30d": 45.0,
        "time_since_prev_txn_seconds": 3600.0,
        "is_first_account_txn": 0,
        "amt_sum_1h": 450.0,
        "amt_sum_24h": 1200.0,
        "amt_sum_7d": 3500.0,
        "amt_sum_30d": 8500.0,
        "amt_mean_24h": 150.0,
        "amt_mean_7d": 175.0,
        "amt_max_24h": 500.0,
        "amt_median_30d": 140.0,
        "historical_amount_mean": 135.0,
        "historical_amount_std": 45.0,
        "historical_amount_median": 130.0,
        "amount_zscore": 2.55,
        "amount_ratio_to_historical_mean": 1.85,
        "account_txn_count_before": 120.0,
        "account_total_spend_before": 16200.0,
        "account_avg_amount_before": 135.0,
        "account_max_amount_before": 650.0,
        "account_unique_merchant_count_before": 42.0,
        "account_unique_category_count_before": 12.0,
        "account_merchant_txn_count_before": 15.0,
        "account_category_txn_count_before": 35.0,
        "account_merchant_spend_before": 1800.0,
        "account_category_spend_before": 4200.0,
        "merchant_txn_count_before": 1500.0,
        "category_txn_count_before": 12000.0,
        "cardholder_merchant_distance_km": 2.45,
        "distance_from_prev_merchant_km": 1.15,
        "implied_travel_speed_kmh": 15.5,
        "is_impossible_travel_speed": 0,
    }

    return PersistRiskEvaluationCommand(
        transaction=TransactionData(
            external_transaction_id=external_id,
            account_id="ACC_INT_98765",
            merchant_id="MERCH_INT_4321",
            merchant_category="grocery_pos",
            job_category="engineer",
            amount=amount,
            currency="USD",
            cardholder_lat=Decimal("40.712800"),
            cardholder_long=Decimal("-74.006000"),
            merchant_lat=Decimal("40.713000"),
            merchant_long=Decimal("-74.005800"),
            city_pop=500000,
            transaction_timestamp=datetime.now(timezone.utc),
            features_snapshot=snapshot_55,
        ),
        evaluation=RiskEvaluationData(
            model_version="1.0.0",
            policy_mode=PolicyMode.TRI_TIER,
            model_score=Decimal("0.725000"),
            risk_score=73,
            risk_tier=RiskTier.HIGH,
            decision_action=DecisionAction.REVIEW,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=False,
            rule_action=None,
            decision_reason="Model score in high-risk manual review tier",
            output_margin=Decimal("0.969400"),
            base_value=Decimal("-1.500000"),
            evaluation_latency_ms=Decimal("14.80"),
            correlation_id="corr-int-001",
            evaluated_at=datetime.now(timezone.utc),
        ),
        rule_matches=[
            RuleMatchData(
                rule_id="RULE_VELOCITY_BURST_REVIEW",
                description="Elevated transaction velocity within 1 hour",
                feature_name="txn_count_1h",
                operator=">",
                comparison_value="4.0",
                outcome=RuleOutcome.REVIEW,
                rule_type=RuleType.VELOCITY,
                priority=10,
            )
        ],
        reason_codes=[
            ReasonCodeData(
                code="RC_AMOUNT_ZSCORE_ELEVATED",
                headline="Unusual Spending Deviation",
                description="Transaction amount deviates from cardholder historical profile",
                category="AMOUNT",
                source=ReasonSource.MODEL,
                severity=ReasonSeverity.HIGH,
                rank=1,
            ),
            ReasonCodeData(
                code="RC_VELOCITY_1H",
                headline="High Frequency Burst",
                description="Multiple transactions in past 1 hour",
                category="VELOCITY",
                source=ReasonSource.RULE,
                severity=ReasonSeverity.MEDIUM,
                rank=2,
            ),
        ],
        feature_attributions=[
            FeatureAttributionData(
                feature_name="amount_zscore",
                display_name="Amount Z-Score",
                raw_value=2.55,
                shap_value=Decimal("0.550000"),
                direction=AttributionDirection.RISK_INCREASING,
                relative_contribution_pct=Decimal("0.6500"),
                rank=1,
            ),
            FeatureAttributionData(
                feature_name="time_since_prev_txn_seconds",
                display_name="Time Since Previous Transaction",
                raw_value=3600.0,
                shap_value=Decimal("-0.200000"),
                direction=AttributionDirection.MITIGATING,
                relative_contribution_pct=Decimal("0.3500"),
                rank=2,
            ),
        ],
        audit=AuditLogData(
            event_type="RISK_EVALUATION_PERSISTED",
            action="PERSIST_EVALUATION",
            actor_type=AuditActorType.SYSTEM,
            actor_id="integration_test_runner",
            correlation_id="corr-int-001",
            client_ip="127.0.0.1",
        ),
    )


# ==============================================================================
# Integration Test Cases
# ==============================================================================

class TestPersistenceOrchestrationIntegration:
    """Live PostgreSQL database integration tests for the persistence service."""

    async def test_1_persist_complete_risk_evaluation(
        self,
        persistence_service: FraudPersistenceService,
        uow: FraudPersistenceUnitOfWork,
    ):
        """Test 1: Persist complete evaluation aggregate and verify all rows in database."""
        cmd = create_sample_persistence_command(external_id="TX_INT_TEST_001")
        result = await persistence_service.persist_evaluation(cmd)

        assert isinstance(result, PersistedRiskEvaluationResult)
        assert result.external_transaction_id == "TX_INT_TEST_001"
        assert result.rule_matches_count == 1
        assert result.reason_codes_count == 2
        assert result.feature_attributions_count == 2

        # 1. Verify Transaction row
        tx = await uow.transactions.get_by_id(result.transaction_id)
        assert tx is not None
        assert tx.id == result.transaction_id
        assert tx.external_transaction_id == "TX_INT_TEST_001"
        assert tx.account_id == "ACC_INT_98765"
        assert tx.amount == Decimal("249.99")
        assert tx.currency == "USD"
        assert tx.cardholder_lat == Decimal("40.712800")
        assert tx.cardholder_long == Decimal("-74.006000")
        assert len(tx.features_snapshot) == 55
        assert tx.features_snapshot["txn_count_1h"] == 3.0

        # 2. Verify RiskEvaluation row
        eval_row = await uow.risk_evaluations.get_by_id(result.evaluation_id)
        assert eval_row is not None
        assert eval_row.id == result.evaluation_id
        assert eval_row.transaction_id == tx.id
        assert eval_row.model_score == Decimal("0.725000")
        assert eval_row.risk_score == 73
        assert eval_row.risk_tier == RiskTier.HIGH
        assert eval_row.decision_action == DecisionAction.REVIEW
        assert eval_row.output_margin == Decimal("0.969400")
        assert eval_row.base_value == Decimal("-1.500000")

        # 3. Verify RuleMatch rows
        rules = await uow.rule_matches.list_by_evaluation_id(result.evaluation_id)
        assert len(rules) == 1
        assert rules[0].evaluation_id == eval_row.id
        assert rules[0].rule_id == "RULE_VELOCITY_BURST_REVIEW"
        assert rules[0].outcome == RuleOutcome.REVIEW
        assert rules[0].priority == 10

        # 4. Verify ReasonCode rows
        reasons = await uow.reason_codes.list_by_evaluation_id(result.evaluation_id)
        assert len(reasons) == 2
        assert reasons[0].code == "RC_AMOUNT_ZSCORE_ELEVATED"
        assert reasons[0].rank == 1
        assert reasons[1].code == "RC_VELOCITY_1H"
        assert reasons[1].rank == 2

        # 5. Verify FeatureAttribution rows
        attrs = await uow.feature_attributions.list_by_evaluation_id(result.evaluation_id)
        assert len(attrs) == 2
        assert attrs[0].feature_name == "amount_zscore"
        assert attrs[0].shap_value == Decimal("0.550000")
        assert attrs[0].direction == AttributionDirection.RISK_INCREASING

        # 6. Verify AuditLog row
        assert result.audit_log_id is not None
        audit = await uow.audit_logs.get_by_id(result.audit_log_id)
        assert audit is not None
        assert audit.entity_type == AuditEntityType.RISK_EVALUATION
        assert audit.entity_id == eval_row.id
        assert audit.action == "PERSIST_EVALUATION"
        assert audit.actor_type == AuditActorType.SYSTEM
        assert audit.actor_id == "integration_test_runner"

    async def test_2_read_persisted_aggregate_across_separate_session(
        self,
        pg_engine: AsyncEngine,
        persistence_service: FraudPersistenceService,
    ):
        """Test 2: Prove that committed data is visible from a completely separate AsyncSession."""
        cmd = create_sample_persistence_command(external_id="TX_INT_CROSS_SESSION")
        result = await persistence_service.persist_evaluation(cmd)

        # Open completely fresh session
        session_factory = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as separate_session:
            new_uow = FraudPersistenceUnitOfWork(separate_session)

            # Read by external transaction ID
            tx = await new_uow.transactions.get_by_external_id("TX_INT_CROSS_SESSION")
            assert tx is not None
            assert tx.id == result.transaction_id

            # Read latest evaluation
            eval_row = await new_uow.risk_evaluations.get_by_transaction_id(tx.id)
            assert eval_row is not None
            assert eval_row.id == result.evaluation_id

            # Read all child attributions
            attrs = await new_uow.feature_attributions.list_by_evaluation_id(eval_row.id)
            assert len(attrs) == 2

            # Read audit log by entity
            audit_records = await new_uow.audit_logs.list_by_entity(
                AuditEntityType.RISK_EVALUATION,
                entity_id=eval_row.id,
            )
            assert len(audit_records) >= 1
            assert audit_records[0].id == result.audit_log_id

    async def test_3_verify_duplicate_external_transaction_handling(
        self,
        pg_engine: AsyncEngine,
        persistence_service: FraudPersistenceService,
    ):
        """Test 3: Verify PersistenceConflictError on duplicate external_transaction_id."""
        cmd = create_sample_persistence_command(external_id="TX_INT_DUP_001")

        # First persist succeeds
        result1 = await persistence_service.persist_evaluation(cmd)
        assert result1.is_new_transaction is True

        # Second persist with same external ID must raise PersistenceConflictError
        with pytest.raises(PersistenceConflictError) as exc_info:
            await persistence_service.persist_evaluation(cmd)

        assert "already exists" in str(exc_info.value)
        assert exc_info.value.details == {"external_transaction_id": "TX_INT_DUP_001"}

        # Verify in a separate session that only exactly 1 transaction exists
        session_factory = async_sessionmaker(pg_engine, class_=AsyncSession)
        async with session_factory() as session:
            stmt = select(Transaction).where(Transaction.external_transaction_id == "TX_INT_DUP_001")
            res = await session.execute(stmt)
            tx_rows = list(res.scalars().all())
            assert len(tx_rows) == 1

            stmt_eval = select(RiskEvaluation).where(RiskEvaluation.transaction_id == tx_rows[0].id)
            res_eval = await session.execute(stmt_eval)
            assert len(list(res_eval.scalars().all())) == 1

    async def test_4_verify_rollback_when_child_insert_fails(
        self,
        pg_engine: AsyncEngine,
    ):
        """Test 4: Verify atomic rollback when an error occurs during child persistence."""
        session_factory = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            uow = FraudPersistenceUnitOfWork(session)
            service = FraudPersistenceService(uow)

            cmd = create_sample_persistence_command(external_id="TX_INT_FAIL_ROLLBACK")

            # Patch reason_codes.add_many to fail with a simulated database error
            with patch.object(
                uow.reason_codes,
                "add_many",
                side_effect=PersistenceError("Simulated child batch persistence failure"),
            ):
                with pytest.raises(PersistenceError, match="Simulated child batch persistence failure"):
                    await service.persist_evaluation(cmd)

        # Open a new session to verify that NO partial data was committed
        async with session_factory() as verify_session:
            stmt_tx = select(Transaction).where(Transaction.external_transaction_id == "TX_INT_FAIL_ROLLBACK")
            res_tx = await verify_session.execute(stmt_tx)
            assert res_tx.scalar_one_or_none() is None

            stmt_eval = select(RiskEvaluation)
            res_eval = await verify_session.execute(stmt_eval)
            assert len(list(res_eval.scalars().all())) == 0

    async def test_5_verify_foreign_key_integrity_and_restrict_delete(
        self,
        pg_engine: AsyncEngine,
        persistence_service: FraudPersistenceService,
    ):
        """Test 5: Verify foreign-key constraints and RESTRICT deletion semantics."""
        cmd = create_sample_persistence_command(external_id="TX_INT_FK_TEST")
        result = await persistence_service.persist_evaluation(cmd)

        session_factory = async_sessionmaker(pg_engine, class_=AsyncSession)

        # 1. Attempt invalid foreign key insertion directly
        async with session_factory() as session:
            invalid_eval = RiskEvaluation(
                id=uuid.uuid4(),
                transaction_id=uuid.uuid4(),  # Non-existent foreign key
                model_version="1.0.0",
                policy_mode=PolicyMode.TRI_TIER,
                model_score=Decimal("0.500000"),
                risk_score=50,
                risk_tier=RiskTier.MEDIUM,
                decision_action=DecisionAction.REVIEW,
                baseline_action=DecisionAction.APPROVE,
                decision_reason="Test invalid FK",
                evaluated_at=datetime.now(timezone.utc),
            )
            session.add(invalid_eval)
            with pytest.raises(IntegrityError):
                await session.commit()

        # 2. Attempt deleting Transaction when evaluations exist (ON DELETE RESTRICT)
        async with session_factory() as session:
            tx = await session.get(Transaction, result.transaction_id)
            assert tx is not None
            await session.delete(tx)
            with pytest.raises(IntegrityError):
                await session.commit()

    async def test_6_verify_database_check_constraints(
        self,
        pg_engine: AsyncEngine,
    ):
        """Test 6: Verify PostgreSQL check constraints on model score, risk score, and amounts."""
        session_factory = async_sessionmaker(pg_engine, class_=AsyncSession)

        # 1. Negative transaction amount violates chk_transactions_amount_positive
        async with session_factory() as session:
            invalid_tx = Transaction(
                id=uuid.uuid4(),
                account_id="ACC_INVALID",
                merchant_category="test",
                job_category="test",
                amount=Decimal("-50.00"),  # Violates amount >= 0.00
                currency="USD",
                cardholder_lat=Decimal("0.0"),
                cardholder_long=Decimal("0.0"),
                merchant_lat=Decimal("0.0"),
                merchant_long=Decimal("0.0"),
                city_pop=1000,
                transaction_timestamp=datetime.now(timezone.utc),
                features_snapshot={},
            )
            session.add(invalid_tx)
            with pytest.raises(IntegrityError):
                await session.commit()

        # 2. Model score > 1.0 violates chk_risk_evaluations_model_score
        async with session_factory() as session:
            # First insert a valid transaction
            valid_tx = Transaction(
                id=uuid.uuid4(),
                account_id="ACC_VALID",
                merchant_category="test",
                job_category="test",
                amount=Decimal("10.00"),
                currency="USD",
                cardholder_lat=Decimal("0.0"),
                cardholder_long=Decimal("0.0"),
                merchant_lat=Decimal("0.0"),
                merchant_long=Decimal("0.0"),
                city_pop=1000,
                transaction_timestamp=datetime.now(timezone.utc),
                features_snapshot={},
            )
            session.add(valid_tx)
            await session.flush()

            invalid_eval = RiskEvaluation(
                id=uuid.uuid4(),
                transaction_id=valid_tx.id,
                model_version="1.0.0",
                policy_mode=PolicyMode.TRI_TIER,
                model_score=Decimal("1.500000"),  # Violates model_score <= 1.0
                risk_score=50,
                risk_tier=RiskTier.MEDIUM,
                decision_action=DecisionAction.REVIEW,
                baseline_action=DecisionAction.APPROVE,
                decision_reason="Test invalid score",
                evaluated_at=datetime.now(timezone.utc),
            )
            session.add(invalid_eval)
            with pytest.raises(IntegrityError):
                await session.commit()

    async def test_7_verify_json_numeric_timestamp_enum_fidelity(
        self,
        pg_engine: AsyncEngine,
        persistence_service: FraudPersistenceService,
    ):
        """Test 7: Verify accurate persistence and round-trip types of JSON, Numeric, Timestamps, and Enums."""
        cmd = create_sample_persistence_command(
            external_id="TX_INT_TYPE_FIDELITY",
            amount=Decimal("1234.56"),
        )
        result = await persistence_service.persist_evaluation(cmd)

        session_factory = async_sessionmaker(pg_engine, class_=AsyncSession)
        async with session_factory() as session:
            uow = FraudPersistenceUnitOfWork(session)
            tx = await uow.transactions.get_by_id(result.transaction_id)
            eval_row = await uow.risk_evaluations.get_by_id(result.evaluation_id)
            attrs = await uow.feature_attributions.list_by_evaluation_id(result.evaluation_id)

            # 1. Decimal & Numeric
            assert tx.amount == Decimal("1234.56")
            assert eval_row.model_score == Decimal("0.725000")
            assert attrs[0].shap_value == Decimal("0.550000")

            # 2. JSON Snapshot
            assert isinstance(tx.features_snapshot, dict)
            assert tx.features_snapshot["city_pop"] == 500000
            assert tx.features_snapshot["implied_travel_speed_kmh"] == 15.5

            # 3. Timestamps are timezone-aware
            assert tx.transaction_timestamp.tzinfo is not None
            assert eval_row.evaluated_at.tzinfo is not None

            # 4. Enums
            assert eval_row.policy_mode is PolicyMode.TRI_TIER
            assert eval_row.risk_tier is RiskTier.HIGH
            assert eval_row.decision_action is DecisionAction.REVIEW
            assert attrs[0].direction is AttributionDirection.RISK_INCREASING

    async def test_8_verify_unit_of_work_transaction_boundaries(
        self,
        pg_engine: AsyncEngine,
        uow: FraudPersistenceUnitOfWork,
    ):
        """Test 8: Verify Unit of Work explicit commit/rollback controls database visibility."""
        session_factory = async_sessionmaker(pg_engine, class_=AsyncSession)

        # 1. Add entity but explicitly rollback -> must not exist in database
        tx_rollback = Transaction(
            id=uuid.uuid4(),
            external_transaction_id="TX_INT_EXPLICIT_ROLLBACK",
            account_id="ACC_RB",
            merchant_category="test",
            job_category="test",
            amount=Decimal("50.00"),
            currency="USD",
            cardholder_lat=Decimal("0.0"),
            cardholder_long=Decimal("0.0"),
            merchant_lat=Decimal("0.0"),
            merchant_long=Decimal("0.0"),
            city_pop=1000,
            transaction_timestamp=datetime.now(timezone.utc),
            features_snapshot={},
        )
        await uow.transactions.add(tx_rollback)
        await uow.rollback()

        async with session_factory() as session:
            tx_check = await session.get(Transaction, tx_rollback.id)
            assert tx_check is None

        # 2. Add entity and explicitly commit -> must exist in database
        tx_commit = Transaction(
            id=uuid.uuid4(),
            external_transaction_id="TX_INT_EXPLICIT_COMMIT",
            account_id="ACC_COMMIT",
            merchant_category="test",
            job_category="test",
            amount=Decimal("75.00"),
            currency="USD",
            cardholder_lat=Decimal("0.0"),
            cardholder_long=Decimal("0.0"),
            merchant_lat=Decimal("0.0"),
            merchant_long=Decimal("0.0"),
            city_pop=1000,
            transaction_timestamp=datetime.now(timezone.utc),
            features_snapshot={},
        )
        await uow.transactions.add(tx_commit)
        await uow.commit()

        async with session_factory() as session:
            tx_saved = await session.get(Transaction, tx_commit.id)
            assert tx_saved is not None
            assert tx_saved.external_transaction_id == "TX_INT_EXPLICIT_COMMIT"
