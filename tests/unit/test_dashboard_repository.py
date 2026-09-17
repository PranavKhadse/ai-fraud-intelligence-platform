"""
Unit Tests for DashboardRepository (Increment 11.1).

Tests:
1. Empty database handling (returns safe zero-defaults).
2. Rate and average calculation accuracy from database rows.
3. Zero-division protection.
4. Database error handling and PersistenceError wrapping.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest
from sqlalchemy.exc import SQLAlchemyError

from backend.app.repositories.dashboard_repository import DashboardRepository
from backend.app.repositories.exceptions import PersistenceError
from backend.app.schemas.dashboard import DashboardOverviewResponse

pytestmark = pytest.mark.asyncio


class TestDashboardRepositoryUnit:
    """Unit tests for DashboardRepository using mocked AsyncSession."""

    async def test_get_overview_metrics_empty_database(self) -> None:
        """Verify get_overview_metrics returns empty defaults when count is 0."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_mappings = MagicMock()

        # Database returns row with total_transactions = 0
        mock_mappings.one_or_none.return_value = {
            "total_transactions": 0,
            "total_amount": 0,
            "avg_risk_score": 0.0,
            "avg_latency_ms": 0.0,
            "approval_count": 0,
            "review_count": 0,
            "block_count": 0,
        }
        mock_result.mappings.return_value = mock_mappings
        mock_session.execute.return_value = mock_result

        repo = DashboardRepository(mock_session)
        overview = await repo.get_overview_metrics()

        assert overview == DashboardOverviewResponse.empty()
        assert overview.total_transactions == 0
        assert overview.approval_rate == 0.0
        assert overview.review_rate == 0.0
        assert overview.block_rate == 0.0

    async def test_get_overview_metrics_none_row(self) -> None:
        """Verify get_overview_metrics returns empty defaults when query returns None."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_mappings = MagicMock()

        mock_mappings.one_or_none.return_value = None
        mock_result.mappings.return_value = mock_mappings
        mock_session.execute.return_value = mock_result

        repo = DashboardRepository(mock_session)
        overview = await repo.get_overview_metrics()

        assert overview == DashboardOverviewResponse.empty()

    async def test_get_overview_metrics_populated_data(self) -> None:
        """Verify get_overview_metrics correctly aggregates metrics and rounds values."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_mappings = MagicMock()

        mock_mappings.one_or_none.return_value = {
            "total_transactions": 200,
            "total_amount": 15450.758,
            "avg_risk_score": 34.5678,
            "avg_latency_ms": 18.234,
            "approval_count": 160,
            "review_count": 30,
            "block_count": 10,
        }
        mock_result.mappings.return_value = mock_mappings
        mock_session.execute.return_value = mock_result

        repo = DashboardRepository(mock_session)
        overview = await repo.get_overview_metrics()

        assert overview.total_transactions == 200
        assert overview.total_amount == 15450.76
        assert overview.approval_count == 160
        assert overview.approval_rate == 80.0
        assert overview.review_count == 30
        assert overview.review_rate == 15.0
        assert overview.block_count == 10
        assert overview.block_rate == 5.0
        assert overview.average_risk_score == 34.57
        assert overview.average_latency_ms == 18.23

    async def test_get_overview_metrics_sqlalchemy_error(self) -> None:
        """Verify SQLAlchemyError is caught, logged, and wrapped as PersistenceError."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = SQLAlchemyError("Database connection failed")

        repo = DashboardRepository(mock_session)
        with pytest.raises(PersistenceError) as exc_info:
            await repo.get_overview_metrics()

        assert "Failed to retrieve dashboard overview metrics" in str(exc_info.value)

    async def test_get_transactions_empty_database(self) -> None:
        """Verify get_transactions returns empty list when count is 0."""
        mock_session = AsyncMock()
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_count_result

        repo = DashboardRepository(mock_session)
        resp = await repo.get_transactions(limit=20, offset=0)

        assert resp.total_count == 0
        assert resp.items == []
        assert resp.limit == 20
        assert resp.offset == 0

    async def test_get_transactions_populated_results(self) -> None:
        """Verify get_transactions returns mapped TransactionListResponse on data."""
        mock_session = AsyncMock()

        # 1st call for count
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 2

        # 2nd call for items
        mock_data_result = MagicMock()
        mock_mappings = MagicMock()
        mock_mappings.all.return_value = [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "external_transaction_id": "TX_001",
                "transaction_timestamp": "2026-09-17T14:00:00Z",
                "amount": 150.00,
                "currency": "USD",
                "merchant_category": "retail",
                "risk_score": 25,
                "risk_tier": "LOW",
                "decision_action": "APPROVE",
                "is_overridden": False,
                "evaluation_latency_ms": 15.2,
            },
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "external_transaction_id": "TX_002",
                "transaction_timestamp": "2026-09-17T13:50:00Z",
                "amount": 450.00,
                "currency": "USD",
                "merchant_category": "electronics",
                "risk_score": 85,
                "risk_tier": "CRITICAL",
                "decision_action": "BLOCK",
                "is_overridden": True,
                "evaluation_latency_ms": 22.1,
            },
        ]
        mock_data_result.mappings.return_value = mock_mappings

        mock_session.execute.side_effect = [mock_count_result, mock_data_result]

        repo = DashboardRepository(mock_session)
        resp = await repo.get_transactions(limit=10, offset=0)

        assert resp.total_count == 2
        assert len(resp.items) == 2
        assert resp.items[0].id == "11111111-1111-1111-1111-111111111111"
        assert resp.items[0].external_transaction_id == "TX_001"
        assert resp.items[0].decision_action == "APPROVE"
        assert resp.items[1].id == "22222222-2222-2222-2222-222222222222"
        assert resp.items[1].is_overridden is True
        assert resp.items[1].decision_action == "BLOCK"

    async def test_get_transactions_limit_bounding(self) -> None:
        """Verify safe limit and offset bounding."""
        mock_session = AsyncMock()
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_count_result

        repo = DashboardRepository(mock_session)

        # Excessively high limit clamped to 100
        resp1 = await repo.get_transactions(limit=500, offset=-10)
        assert resp1.limit == 100
        assert resp1.offset == 0

        # Negative limit clamped to 1
        resp2 = await repo.get_transactions(limit=-5, offset=5)
        assert resp2.limit == 1
        assert resp2.offset == 5

    async def test_get_transactions_sqlalchemy_error(self) -> None:
        """Verify database error raises PersistenceError in get_transactions."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = SQLAlchemyError("Query execution error")

        repo = DashboardRepository(mock_session)
        with pytest.raises(PersistenceError) as exc_info:
            await repo.get_transactions()

        assert "Failed to retrieve dashboard transactions" in str(exc_info.value)


import uuid
from datetime import datetime, timezone
from decimal import Decimal
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


class TestDashboardRepositoryDetailUnit:
    """Unit tests for DashboardRepository.get_transaction_detail."""

    async def test_get_transaction_detail_not_found(self) -> None:
        """Verify get_transaction_detail returns None when transaction does not exist."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = DashboardRepository(mock_session)
        res = await repo.get_transaction_detail("11111111-1111-1111-1111-111111111111")
        assert res is None

    async def test_get_transaction_detail_success_with_eager_loaded_children(self) -> None:
        """Verify get_transaction_detail properly maps all nested child entities deterministically."""
        mock_session = AsyncMock()

        tx_id = uuid.uuid4()
        eval_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        # Create mock ORM objects
        tx = Transaction(
            id=tx_id,
            external_transaction_id="TX_DET_001",
            account_id="ACC_001",
            merchant_id="MERCH_01",
            merchant_category="electronics",
            job_category="engineer",
            amount=Decimal("499.99"),
            currency="USD",
            cardholder_lat=Decimal("37.7749"),
            cardholder_long=Decimal("-122.4194"),
            merchant_lat=Decimal("37.7833"),
            merchant_long=Decimal("-122.4167"),
            city_pop=800000,
            transaction_timestamp=now,
            created_at=now,
            features_snapshot={"txn_count_1h": 5, "amount": 499.99},
        )

        eval_rec = RiskEvaluation(
            id=eval_id,
            transaction_id=tx_id,
            model_version="1.0.0",
            policy_mode=PolicyMode.TRI_TIER,
            model_score=Decimal("0.850000"),
            risk_score=85,
            risk_tier=RiskTier.CRITICAL,
            decision_action=DecisionAction.BLOCK,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=True,
            rule_action=RuleOutcome.BLOCK,
            decision_reason="Exceeded risk threshold",
            output_margin=Decimal("1.735000"),
            base_value=Decimal("-3.500000"),
            evaluation_latency_ms=Decimal("14.50"),
            correlation_id="corr_999",
            evaluated_at=now,
        )

        # Unordered feature attributions to test deterministic sorting (rank ASC, abs(shap) DESC)
        fa1 = EvaluationFeatureAttribution(
            id=uuid.uuid4(),
            evaluation_id=eval_id,
            feature_name="amt_sum_24h",
            display_name="24h Spending",
            raw_value=1200.0,
            shap_value=Decimal("0.850000"),
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=Decimal("35.0000"),
            rank=2,
        )
        fa2 = EvaluationFeatureAttribution(
            id=uuid.uuid4(),
            evaluation_id=eval_id,
            feature_name="txn_count_1h",
            display_name="1h Velocity",
            raw_value=5,
            shap_value=Decimal("1.450000"),
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=Decimal("65.0000"),
            rank=1,
        )
        eval_rec.feature_attributions = [fa1, fa2]

        # Unordered reason codes (rank 2, rank 1)
        rc1 = EvaluationReasonCode(
            id=uuid.uuid4(),
            evaluation_id=eval_id,
            code="HIGH_AMOUNT",
            headline="High Spend",
            description="Amount elevated",
            category="AMOUNT",
            source=ReasonSource.MODEL,
            severity=ReasonSeverity.MEDIUM,
            rank=2,
        )
        rc2 = EvaluationReasonCode(
            id=uuid.uuid4(),
            evaluation_id=eval_id,
            code="BURST_VELOCITY",
            headline="Rapid Velocity",
            description="5 txns in 1 hour",
            category="VELOCITY",
            source=ReasonSource.RULE,
            severity=ReasonSeverity.HIGH,
            rank=1,
        )
        eval_rec.reason_codes = [rc1, rc2]

        # Unordered rule matches (priority 20, priority 10)
        rm1 = EvaluationRuleMatch(
            id=uuid.uuid4(),
            evaluation_id=eval_id,
            rule_id="RULE_AMOUNT_HIGH",
            description="Review high amount",
            feature_name="amount",
            operator=">",
            comparison_value="300",
            outcome=RuleOutcome.REVIEW,
            rule_type=RuleType.AMOUNT,
            priority=20,
        )
        rm2 = EvaluationRuleMatch(
            id=uuid.uuid4(),
            evaluation_id=eval_id,
            rule_id="RULE_VELOCITY_BURST",
            description="Block rapid velocity",
            feature_name="txn_count_1h",
            operator=">",
            comparison_value="4",
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.VELOCITY,
            priority=10,
        )
        eval_rec.rule_matches = [rm1, rm2]

        tx.evaluations = [eval_rec]

        audit_log = AuditLog(
            id=uuid.uuid4(),
            event_type="RISK_EVALUATION_PERSISTED",
            entity_type=AuditEntityType.RISK_EVALUATION,
            entity_id=eval_id,
            action="PERSIST_EVALUATION",
            actor_type=AuditActorType.SYSTEM,
            actor_id="worker_01",
            correlation_id="corr_999",
            client_ip="127.0.0.1",
            event_timestamp=now,
        )

        # 1st execute for tx query
        mock_tx_result = MagicMock()
        mock_tx_result.scalar_one_or_none.return_value = tx

        # 2nd execute for audit query
        mock_audit_result = MagicMock()
        mock_audit_result.scalars.return_value.all.return_value = [audit_log]

        mock_session.execute.side_effect = [mock_tx_result, mock_audit_result]

        repo = DashboardRepository(mock_session)
        detail = await repo.get_transaction_detail(str(tx_id))

        assert detail is not None
        assert detail.transaction.id == str(tx_id)
        assert detail.transaction.external_transaction_id == "TX_DET_001"
        assert detail.transaction.amount == 499.99
        assert detail.evaluation is not None
        assert detail.evaluation.risk_score == 85
        assert detail.evaluation.decision_action == DecisionAction.BLOCK
        assert detail.evaluation.is_overridden is True

        # Check deterministic feature attribution ordering (rank 1 before rank 2)
        assert len(detail.feature_attributions) == 2
        assert detail.feature_attributions[0].rank == 1
        assert detail.feature_attributions[0].feature_name == "txn_count_1h"
        assert detail.feature_attributions[1].rank == 2
        assert detail.feature_attributions[1].feature_name == "amt_sum_24h"

        # Check deterministic reason code ordering (rank 1 before rank 2)
        assert len(detail.reason_codes) == 2
        assert detail.reason_codes[0].rank == 1
        assert detail.reason_codes[0].code == "BURST_VELOCITY"
        assert detail.reason_codes[1].rank == 2
        assert detail.reason_codes[1].code == "HIGH_AMOUNT"

        # Check deterministic rule match ordering (priority 10 before priority 20)
        assert len(detail.rule_matches) == 2
        assert detail.rule_matches[0].priority == 10
        assert detail.rule_matches[0].rule_id == "RULE_VELOCITY_BURST"
        assert detail.rule_matches[1].priority == 20
        assert detail.rule_matches[1].rule_id == "RULE_AMOUNT_HIGH"

        # Check audit trail mapping
        assert len(detail.audit_trail) == 1
        assert detail.audit_trail[0].action == "PERSIST_EVALUATION"
        assert detail.audit_trail[0].actor_type == "SYSTEM"

    async def test_get_transaction_detail_db_error(self) -> None:
        """Verify database error raises PersistenceError in get_transaction_detail."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = SQLAlchemyError("DB error")

        repo = DashboardRepository(mock_session)
        with pytest.raises(PersistenceError):
            await repo.get_transaction_detail("11111111-1111-1111-1111-111111111111")


class TestDashboardRepositoryAnalyticsUnit:
    """Unit tests for Increment 11.4 analytics repository aggregation methods."""

    async def test_get_analytics_distributions_empty_database(self) -> None:
        """Verify get_analytics_distributions returns compliant zero-value 10-bucket structures."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_mappings = MagicMock()

        # Database returns 0 rows
        mock_mappings.one_or_none.return_value = {
            "total_evaluated": 0,
            "rs_0_9": 0, "rs_10_19": 0, "rs_20_29": 0, "rs_30_39": 0, "rs_40_49": 0,
            "rs_50_59": 0, "rs_60_69": 0, "rs_70_79": 0, "rs_80_89": 0, "rs_90_100": 0,
            "ms_0_1": 0, "ms_1_2": 0, "ms_2_3": 0, "ms_3_4": 0, "ms_4_5": 0,
            "ms_5_6": 0, "ms_6_7": 0, "ms_7_8": 0, "ms_8_9": 0, "ms_9_10": 0,
            "tier_low": 0, "tier_med": 0, "tier_high": 0, "tier_crit": 0,
            "act_appr": 0, "act_rev": 0, "act_blk": 0,
        }
        mock_result.mappings.return_value = mock_mappings
        mock_session.execute.return_value = mock_result

        repo = DashboardRepository(mock_session)
        resp = await repo.get_analytics_distributions()

        assert resp.total_evaluated == 0
        assert len(resp.risk_score_distribution) == 10
        assert resp.risk_score_distribution[0].bucket_label == "0–9"
        assert resp.risk_score_distribution[9].bucket_label == "90–100"
        assert all(b.count == 0 for b in resp.risk_score_distribution)
        assert all(b.percentage == 0.0 for b in resp.risk_score_distribution)

        assert len(resp.model_score_distribution) == 10
        assert len(resp.risk_tier_distribution) == 4
        assert len(resp.decision_distribution) == 3

    async def test_get_analytics_distributions_populated(self) -> None:
        """Verify get_analytics_distributions correctly maps counts and percentages across 10 buckets."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_mappings = MagicMock()

        mock_mappings.one_or_none.return_value = {
            "total_evaluated": 100,
            "rs_0_9": 30, "rs_10_19": 20, "rs_20_29": 10, "rs_30_39": 5, "rs_40_49": 5,
            "rs_50_59": 5, "rs_60_69": 5, "rs_70_79": 5, "rs_80_89": 5, "rs_90_100": 10,
            "ms_0_1": 40, "ms_1_2": 15, "ms_2_3": 5, "ms_3_4": 5, "ms_4_5": 5,
            "ms_5_6": 5, "ms_6_7": 5, "ms_7_8": 5, "ms_8_9": 5, "ms_9_10": 10,
            "tier_low": 60, "tier_med": 15, "tier_high": 15, "tier_crit": 10,
            "act_appr": 60, "act_rev": 25, "act_blk": 15,
        }
        mock_result.mappings.return_value = mock_mappings
        mock_session.execute.return_value = mock_result

        repo = DashboardRepository(mock_session)
        resp = await repo.get_analytics_distributions()

        assert resp.total_evaluated == 100
        # Check risk score buckets
        assert resp.risk_score_distribution[0].bucket_label == "0–9"
        assert resp.risk_score_distribution[0].count == 30
        assert resp.risk_score_distribution[0].percentage == 30.0
        assert resp.risk_score_distribution[9].bucket_label == "90–100"
        assert resp.risk_score_distribution[9].count == 10
        assert resp.risk_score_distribution[9].percentage == 10.0

        # Check tiers
        low_tier = next(t for t in resp.risk_tier_distribution if t.category == "LOW")
        assert low_tier.count == 60
        assert low_tier.percentage == 60.0

        # Check decisions
        blk_dec = next(d for d in resp.decision_distribution if d.category == "BLOCK")
        assert blk_dec.count == 15
        assert blk_dec.percentage == 15.0

    async def test_get_analytics_trends_contiguous_zero_fill(self) -> None:
        """Verify get_analytics_trends creates contiguous time series points for empty windows."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_mappings = MagicMock()

        t_start = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
        t_end = datetime(2026, 9, 17, 13, 0, 0, tzinfo=timezone.utc)

        # Database returns 1 point at 11:00, points at 10:00, 12:00, 13:00 must be zero-filled
        t_mid = datetime(2026, 9, 17, 11, 0, 0, tzinfo=timezone.utc)
        mock_mappings.all.return_value = [
            {
                "bucket_time": t_mid,
                "total_count": 25,
                "total_amount": 3500.0,
                "avg_risk_score": 35.0,
                "approval_count": 20,
                "review_count": 3,
                "block_count": 2,
                "high_critical_count": 5,
            }
        ]
        mock_result.mappings.return_value = mock_mappings
        mock_session.execute.return_value = mock_result

        repo = DashboardRepository(mock_session)
        resp = await repo.get_analytics_trends(
            start_date=t_start,
            end_date=t_end,
            interval=None,  # 3 hour delta -> auto-selected to HOURLY
        )

        assert resp.interval.value == "hourly"
        # Points generated: 10:00, 11:00, 12:00, 13:00 (4 points)
        assert len(resp.data_points) == 4
        assert resp.data_points[0].total_count == 0
        assert resp.data_points[0].total_amount == 0.0

        # Point at 11:00 has actual data
        assert resp.data_points[1].total_count == 25
        assert resp.data_points[1].total_amount == 3500.0
        assert resp.data_points[1].block_count == 2

        # Point at 12:00 is zero-filled
        assert resp.data_points[2].total_count == 0

    async def test_get_analytics_rules_aggregation(self) -> None:
        """Verify get_analytics_rules correctly computes trigger rates and outcome breakdowns."""
        mock_session = AsyncMock()

        # 1st call for total evaluations count
        mock_evals_res = MagicMock()
        mock_evals_res.scalar.return_value = 200

        # 2nd call for rules aggregation rows
        mock_rules_res = MagicMock()
        mock_mappings = MagicMock()
        mock_mappings.all.return_value = [
            {
                "rule_id": "RULE_VELOCITY_BURST",
                "description": "Exceeded 5 transactions in hour",
                "rule_type": "VELOCITY",
                "priority": 10,
                "trigger_count": 20,
                "affected_transactions": 20,
                "override_count": 15,
                "cnt_block": 18,
                "cnt_review": 2,
                "cnt_approve": 0,
                "cnt_monitor": 0,
            }
        ]
        mock_rules_res.mappings.return_value = mock_mappings
        mock_session.execute.side_effect = [mock_evals_res, mock_rules_res]

        repo = DashboardRepository(mock_session)
        resp = await repo.get_analytics_rules()

        assert resp.total_rules_active == 1
        assert resp.total_evaluations_analyzed == 200
        assert len(resp.rules) == 1

        r = resp.rules[0]
        assert r.rule_id == "RULE_VELOCITY_BURST"
        assert r.trigger_count == 20
        assert r.affected_transactions == 20
        # trigger_rate = (20 / 200) * 100 = 10.0%
        assert r.trigger_rate == 10.0
        assert r.override_count == 15
        assert len(r.outcomes) == 2  # BLOCK (18), REVIEW (2)
        assert r.outcomes[0].outcome == "BLOCK"
        assert r.outcomes[0].count == 18
        assert r.outcomes[0].percentage == 90.0

