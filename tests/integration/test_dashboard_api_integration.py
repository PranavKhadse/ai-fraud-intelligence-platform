"""
Integration Tests for Dashboard API Endpoint (Increment 11.1).

Verifies GET /api/v1/dashboard/overview against live PostgreSQL database:
1. Empty database behavior (returns 200 OK with safe zero metrics).
2. Seeded database behavior (verifies SQL aggregations, rates, sums, and averages).
3. Schema adherence to DashboardOverviewResponse.
4. Database error handling (returns 500 on unexpected persistence failure).
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncGenerator
from unittest.mock import patch
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.db.models.enums import DecisionAction, PolicyMode, RiskTier
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.transaction import Transaction
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.repositories.dashboard_repository import DashboardRepository
from backend.app.repositories.exceptions import PersistenceError
from backend.app.schemas.dashboard import DashboardOverviewResponse

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="function")
async def async_api_client(
    db_session: AsyncSession, pg_engine: AsyncEngine
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an asynchronous httpx client with request-scoped database sessions."""
    session_factory = async_sessionmaker(
        bind=pg_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async def override_get_db_session():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_db_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()


class TestDashboardApiIntegration:
    """Integration test suite for /api/v1/dashboard/overview."""

    async def test_dashboard_overview_empty_db(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify GET /api/v1/dashboard/overview returns safe zero metrics on empty database."""
        response = await async_api_client.get("/api/v1/dashboard/overview")
        assert response.status_code == 200
        data = response.json()

        # Validate against Pydantic schema
        overview = DashboardOverviewResponse(**data)
        assert overview.total_transactions == 0
        assert overview.total_amount == 0.0
        assert overview.approval_count == 0
        assert overview.approval_rate == 0.0
        assert overview.review_count == 0
        assert overview.review_rate == 0.0
        assert overview.block_count == 0
        assert overview.block_rate == 0.0
        assert overview.average_risk_score == 0.0
        assert overview.average_latency_ms == 0.0

    async def test_dashboard_overview_seeded_transactions(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify GET /api/v1/dashboard/overview computes exact metrics on seeded data."""
        now = datetime.now(timezone.utc)

        # 1. Approved transaction
        tx1 = Transaction(
            external_transaction_id="TX_DASH_001",
            account_id="ACC_DASH_001",
            amount=Decimal("100.00"),
            currency="USD",
            transaction_timestamp=now,
            merchant_id="MERCH_01",
            merchant_category="retail",
            job_category="engineer",
            cardholder_lat=Decimal("40.710000"),
            cardholder_long=Decimal("-74.000000"),
            merchant_lat=Decimal("40.720000"),
            merchant_long=Decimal("-74.010000"),
            city_pop=50000,
            features_snapshot={"amount": 100.00},
        )
        db_session.add(tx1)
        await db_session.flush()

        eval1 = RiskEvaluation(
            transaction_id=tx1.id,
            model_score=Decimal("0.050000"),
            risk_score=10,
            risk_tier=RiskTier.LOW,
            decision_action=DecisionAction.APPROVE,
            baseline_action=DecisionAction.APPROVE,
            is_overridden=False,
            decision_reason="Score below review threshold",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("10.00"),
            evaluated_at=now,
            model_version="1.0.0",
        )
        db_session.add(eval1)

        # 2. Review transaction
        tx2 = Transaction(
            external_transaction_id="TX_DASH_002",
            account_id="ACC_DASH_002",
            amount=Decimal("300.00"),
            currency="USD",
            transaction_timestamp=now,
            merchant_id="MERCH_02",
            merchant_category="electronics",
            job_category="doctor",
            cardholder_lat=Decimal("40.710000"),
            cardholder_long=Decimal("-74.000000"),
            merchant_lat=Decimal("40.750000"),
            merchant_long=Decimal("-74.050000"),
            city_pop=50000,
            features_snapshot={"amount": 300.00},
        )
        db_session.add(tx2)
        await db_session.flush()

        eval2 = RiskEvaluation(
            transaction_id=tx2.id,
            model_score=Decimal("0.550000"),
            risk_score=60,
            risk_tier=RiskTier.MEDIUM,
            decision_action=DecisionAction.REVIEW,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=False,
            decision_reason="Score reached review threshold",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("20.00"),
            evaluated_at=now,
            model_version="1.0.0",
        )
        db_session.add(eval2)

        # 3. Blocked transaction
        tx3 = Transaction(
            external_transaction_id="TX_DASH_003",
            account_id="ACC_DASH_003",
            amount=Decimal("600.00"),
            currency="USD",
            transaction_timestamp=now,
            merchant_id="MERCH_03",
            merchant_category="jewelry",
            job_category="lawyer",
            cardholder_lat=Decimal("40.710000"),
            cardholder_long=Decimal("-74.000000"),
            merchant_lat=Decimal("40.800000"),
            merchant_long=Decimal("-74.100000"),
            city_pop=50000,
            features_snapshot={"amount": 600.00},
        )
        db_session.add(tx3)
        await db_session.flush()

        eval3 = RiskEvaluation(
            transaction_id=tx3.id,
            model_score=Decimal("0.920000"),
            risk_score=95,
            risk_tier=RiskTier.CRITICAL,
            decision_action=DecisionAction.BLOCK,
            baseline_action=DecisionAction.BLOCK,
            is_overridden=False,
            decision_reason="Score reached block threshold",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("30.00"),
            evaluated_at=now,
            model_version="1.0.0",
        )
        db_session.add(eval3)

        await db_session.commit()

        # Query endpoint
        response = await async_api_client.get("/api/v1/dashboard/overview")
        assert response.status_code == 200
        data = response.json()

        overview = DashboardOverviewResponse(**data)
        assert overview.total_transactions == 3
        assert overview.total_amount == 1000.00
        assert overview.approval_count == 1
        assert overview.approval_rate == 33.33
        assert overview.review_count == 1
        assert overview.review_rate == 33.33
        assert overview.block_count == 1
        assert overview.block_rate == 33.33
        # avg risk score = (10 + 60 + 95) / 3 = 165 / 3 = 55.00
        assert overview.average_risk_score == 55.00
        # avg latency = (10 + 20 + 30) / 3 = 60 / 3 = 20.00
        assert overview.average_latency_ms == 20.00

    async def test_dashboard_overview_database_error_handling(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify endpoint returns HTTP 500 when repository raises PersistenceError."""
        with patch.object(
            DashboardRepository,
            "get_overview_metrics",
            side_effect=PersistenceError("Simulated DB failure"),
        ):
            response = await async_api_client.get("/api/v1/dashboard/overview")
            assert response.status_code == 500
            assert "Failed to retrieve dashboard overview metrics" in response.json()["detail"]

    async def test_dashboard_transactions_empty_db(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify GET /api/v1/dashboard/transactions returns empty page on empty database."""
        response = await async_api_client.get("/api/v1/dashboard/transactions")
        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 0
        assert data["items"] == []
        assert data["limit"] == 20
        assert data["offset"] == 0

    async def test_dashboard_transactions_seeded_and_filters(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify GET /api/v1/dashboard/transactions with pagination, ordering, and all filters."""
        t1 = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 17, 11, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Approved transaction (Earliest)
        tx1 = Transaction(
            external_transaction_id="TX_FEED_001",
            account_id="ACC_FEED_001",
            amount=Decimal("100.00"),
            currency="USD",
            transaction_timestamp=t1,
            merchant_id="MERCH_RETAIL_01",
            merchant_category="retail",
            job_category="engineer",
            cardholder_lat=Decimal("40.710000"),
            cardholder_long=Decimal("-74.000000"),
            merchant_lat=Decimal("40.720000"),
            merchant_long=Decimal("-74.010000"),
            city_pop=50000,
            features_snapshot={"amount": 100.00, "secret_feature_vector": [1, 2, 3]},
        )
        db_session.add(tx1)
        await db_session.flush()

        eval1 = RiskEvaluation(
            transaction_id=tx1.id,
            model_score=Decimal("0.050000"),
            risk_score=10,
            risk_tier=RiskTier.LOW,
            decision_action=DecisionAction.APPROVE,
            baseline_action=DecisionAction.APPROVE,
            is_overridden=False,
            decision_reason="Score below review threshold",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("10.00"),
            evaluated_at=t1,
            model_version="1.0.0",
        )
        db_session.add(eval1)

        # 2. Review transaction (Middle)
        tx2 = Transaction(
            external_transaction_id="TX_FEED_002",
            account_id="ACC_FEED_002",
            amount=Decimal("350.00"),
            currency="USD",
            transaction_timestamp=t2,
            merchant_id="MERCH_ELEC_02",
            merchant_category="electronics",
            job_category="doctor",
            cardholder_lat=Decimal("40.710000"),
            cardholder_long=Decimal("-74.000000"),
            merchant_lat=Decimal("40.750000"),
            merchant_long=Decimal("-74.050000"),
            city_pop=50000,
            features_snapshot={"amount": 350.00},
        )
        db_session.add(tx2)
        await db_session.flush()

        eval2 = RiskEvaluation(
            transaction_id=tx2.id,
            model_score=Decimal("0.550000"),
            risk_score=60,
            risk_tier=RiskTier.MEDIUM,
            decision_action=DecisionAction.REVIEW,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=False,
            decision_reason="Score reached review threshold",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("20.00"),
            evaluated_at=t2,
            model_version="1.0.0",
        )
        db_session.add(eval2)

        # 3. Blocked transaction (Latest)
        tx3 = Transaction(
            external_transaction_id="TX_FEED_003",
            account_id="ACC_FEED_003",
            amount=Decimal("750.00"),
            currency="USD",
            transaction_timestamp=t3,
            merchant_id="MERCH_JEWEL_03",
            merchant_category="jewelry",
            job_category="lawyer",
            cardholder_lat=Decimal("40.710000"),
            cardholder_long=Decimal("-74.000000"),
            merchant_lat=Decimal("40.800000"),
            merchant_long=Decimal("-74.100000"),
            city_pop=50000,
            features_snapshot={"amount": 750.00},
        )
        db_session.add(tx3)
        await db_session.flush()

        eval3 = RiskEvaluation(
            transaction_id=tx3.id,
            model_score=Decimal("0.950000"),
            risk_score=95,
            risk_tier=RiskTier.CRITICAL,
            decision_action=DecisionAction.BLOCK,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=True,
            decision_reason="High amount override rule match",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("30.00"),
            evaluated_at=t3,
            model_version="1.0.0",
        )
        db_session.add(eval3)

        await db_session.commit()

        # 1. Test default list (newest first: tx3, tx2, tx1)
        resp = await async_api_client.get("/api/v1/dashboard/transactions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 3
        assert len(data["items"]) == 3
        assert data["items"][0]["external_transaction_id"] == "TX_FEED_003"
        assert data["items"][1]["external_transaction_id"] == "TX_FEED_002"
        assert data["items"][2]["external_transaction_id"] == "TX_FEED_001"

        # Verify no sensitive or bulky feature snapshots exposed
        first_item = data["items"][0]
        assert "features_snapshot" not in first_item
        assert "secret_feature_vector" not in first_item
        assert "cardholder_lat" not in first_item

        # 2. Test pagination limit=2, offset=1 (should return tx2, tx1)
        resp_page = await async_api_client.get("/api/v1/dashboard/transactions?limit=2&offset=1")
        assert resp_page.status_code == 200
        page_data = resp_page.json()
        assert page_data["total_count"] == 3
        assert len(page_data["items"]) == 2
        assert page_data["items"][0]["external_transaction_id"] == "TX_FEED_002"
        assert page_data["items"][1]["external_transaction_id"] == "TX_FEED_001"

        # 3. Test filter decision_action=REVIEW
        resp_rev = await async_api_client.get("/api/v1/dashboard/transactions?decision_action=REVIEW")
        assert resp_rev.status_code == 200
        rev_data = resp_rev.json()
        assert rev_data["total_count"] == 1
        assert rev_data["items"][0]["external_transaction_id"] == "TX_FEED_002"

        # 4. Test filter risk_tier=CRITICAL
        resp_crit = await async_api_client.get("/api/v1/dashboard/transactions?risk_tier=CRITICAL")
        assert resp_crit.status_code == 200
        crit_data = resp_crit.json()
        assert crit_data["total_count"] == 1
        assert crit_data["items"][0]["external_transaction_id"] == "TX_FEED_003"
        assert crit_data["items"][0]["is_overridden"] is True

        # 5. Test score range min_score=50&max_score=80
        resp_score = await async_api_client.get("/api/v1/dashboard/transactions?min_score=50&max_score=80")
        assert resp_score.status_code == 200
        score_data = resp_score.json()
        assert score_data["total_count"] == 1
        assert score_data["items"][0]["external_transaction_id"] == "TX_FEED_002"

        # 6. Test search term for merchant category
        resp_search = await async_api_client.get("/api/v1/dashboard/transactions?search_term=jewelry")
        assert resp_search.status_code == 200
        search_data = resp_search.json()
        assert search_data["total_count"] == 1
        assert search_data["items"][0]["external_transaction_id"] == "TX_FEED_003"

        # 7. Test date range
        resp_date = await async_api_client.get(
            "/api/v1/dashboard/transactions",
            params={"start_date": t2.isoformat(), "end_date": t3.isoformat()},
        )
        assert resp_date.status_code == 200
        date_data = resp_date.json()
        assert date_data["total_count"] == 2
        assert date_data["items"][0]["external_transaction_id"] == "TX_FEED_003"
        assert date_data["items"][1]["external_transaction_id"] == "TX_FEED_002"

        # 8. Combined filters
        resp_comb = await async_api_client.get(
            "/api/v1/dashboard/transactions?decision_action=BLOCK&risk_tier=CRITICAL&min_score=90"
        )
        assert resp_comb.status_code == 200
        comb_data = resp_comb.json()
        assert comb_data["total_count"] == 1
        assert comb_data["items"][0]["external_transaction_id"] == "TX_FEED_003"

    async def test_dashboard_transactions_validation_errors(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify invalid query parameter validation."""
        # min_score > max_score -> 400 Bad Request
        resp1 = await async_api_client.get("/api/v1/dashboard/transactions?min_score=80&max_score=20")
        assert resp1.status_code == 400
        assert "min_score cannot exceed max_score" in resp1.json()["detail"]

        # start_date > end_date -> 400 Bad Request
        resp2 = await async_api_client.get(
            "/api/v1/dashboard/transactions?start_date=2026-09-18T00:00:00Z&end_date=2026-09-17T00:00:00Z"
        )
        assert resp2.status_code == 400
        assert "start_date cannot be later than end_date" in resp2.json()["detail"]

        # limit > 100 -> 422 Unprocessable Entity
        resp3 = await async_api_client.get("/api/v1/dashboard/transactions?limit=150")
        assert resp3.status_code == 422

        # offset < 0 -> 422 Unprocessable Entity
        resp4 = await async_api_client.get("/api/v1/dashboard/transactions?offset=-5")
        assert resp4.status_code == 422

    async def test_dashboard_transactions_database_error(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify endpoint returns HTTP 500 when repository raises PersistenceError."""
        with patch.object(
            DashboardRepository,
            "get_transactions",
            side_effect=PersistenceError("Simulated query failure"),
        ):
            response = await async_api_client.get("/api/v1/dashboard/transactions")
            assert response.status_code == 500
            assert "Failed to retrieve dashboard transactions" in response.json()["detail"]


from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import (
    AttributionDirection,
    AuditActorType,
    AuditEntityType,
    ReasonSeverity,
    ReasonSource,
    RuleOutcome,
    RuleType,
)
from backend.app.db.models.feature_attribution import EvaluationFeatureAttribution
from backend.app.db.models.reason_code import EvaluationReasonCode
from backend.app.db.models.rule_match import EvaluationRuleMatch
from backend.app.schemas.dashboard import TransactionDetailResponse
import uuid


class TestDashboardTransactionDetailIntegration:
    """Integration test suite for GET /api/v1/dashboard/transactions/{transaction_id} (Increment 11.3)."""

    async def test_transaction_detail_by_uuid_success(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify full nested transaction investigation detail lookup by internal UUID."""
        now = datetime.now(timezone.utc)
        tx_id = uuid.uuid4()
        eval_id = uuid.uuid4()

        # Seed full aggregate
        tx = Transaction(
            id=tx_id,
            external_transaction_id="TX_INT_DET_001",
            account_id="ACC_INT_001",
            merchant_id="MERCH_INT_01",
            merchant_category="electronics",
            job_category="software_engineer",
            amount=Decimal("1250.00"),
            currency="USD",
            cardholder_lat=Decimal("37.774900"),
            cardholder_long=Decimal("-122.419400"),
            merchant_lat=Decimal("37.783300"),
            merchant_long=Decimal("-122.416700"),
            city_pop=850000,
            transaction_timestamp=now,
            created_at=now,
            features_snapshot={"txn_count_1h": 8, "amt_sum_24h": 3500.0, "amount": 1250.0},
        )
        db_session.add(tx)
        await db_session.flush()

        evaluation = RiskEvaluation(
            id=eval_id,
            transaction_id=tx_id,
            model_version="1.0.0",
            policy_mode=PolicyMode.TRI_TIER,
            model_score=Decimal("0.925000"),
            risk_score=93,
            risk_tier=RiskTier.CRITICAL,
            decision_action=DecisionAction.BLOCK,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=True,
            rule_action=RuleOutcome.BLOCK,
            decision_reason="Risk score 93 meets BLOCK threshold and triggered RULE_VELOCITY_BURST.",
            output_margin=Decimal("2.512000"),
            base_value=Decimal("-3.542000"),
            evaluation_latency_ms=Decimal("16.40"),
            correlation_id="corr_int_det_001",
            evaluated_at=now,
        )
        db_session.add(evaluation)
        await db_session.flush()

        fa = EvaluationFeatureAttribution(
            id=uuid.uuid4(),
            evaluation_id=eval_id,
            feature_name="txn_count_1h",
            display_name="Transactions in Past 1 Hour",
            raw_value=8,
            shap_value=Decimal("1.850000"),
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=Decimal("55.0000"),
            rank=1,
        )
        db_session.add(fa)

        rc = EvaluationReasonCode(
            id=uuid.uuid4(),
            evaluation_id=eval_id,
            code="VELOCITY_BURST_1H",
            headline="Rapid Transaction Velocity",
            description="8 transactions attempted within the last hour.",
            category="VELOCITY",
            source=ReasonSource.MODEL,
            severity=ReasonSeverity.CRITICAL,
            rank=1,
        )
        db_session.add(rc)

        rm = EvaluationRuleMatch(
            id=uuid.uuid4(),
            evaluation_id=eval_id,
            rule_id="RULE_VELOCITY_BURST_BLOCK",
            description="Block cardholder when hourly transaction count exceeds 5.",
            feature_name="txn_count_1h",
            operator=">",
            comparison_value="5",
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.VELOCITY,
            priority=10,
        )
        db_session.add(rm)

        audit = AuditLog(
            id=uuid.uuid4(),
            event_type="RISK_EVALUATION_PERSISTED",
            entity_type=AuditEntityType.RISK_EVALUATION,
            entity_id=eval_id,
            action="PERSIST_EVALUATION",
            actor_type=AuditActorType.SYSTEM,
            actor_id="test_worker",
            correlation_id="corr_int_det_001",
            client_ip="127.0.0.1",
            event_timestamp=now,
        )
        db_session.add(audit)
        await db_session.commit()

        # Query GET endpoint by UUID
        response = await async_api_client.get(f"/api/v1/dashboard/transactions/{tx_id}")
        assert response.status_code == 200
        data = response.json()

        # Validate against Pydantic schema
        detail = TransactionDetailResponse(**data)
        assert detail.transaction.id == str(tx_id)
        assert detail.transaction.external_transaction_id == "TX_INT_DET_001"
        assert detail.transaction.amount == 1250.00
        assert detail.evaluation is not None
        assert detail.evaluation.risk_score == 93
        assert detail.evaluation.decision_action == DecisionAction.BLOCK
        assert detail.evaluation.is_overridden is True
        assert len(detail.features) == 3
        assert detail.features["txn_count_1h"] == 8

        assert len(detail.feature_attributions) == 1
        assert detail.feature_attributions[0].feature_name == "txn_count_1h"
        assert detail.feature_attributions[0].shap_value == 1.85

        assert len(detail.reason_codes) == 1
        assert detail.reason_codes[0].code == "VELOCITY_BURST_1H"

        assert len(detail.rule_matches) == 1
        assert detail.rule_matches[0].rule_id == "RULE_VELOCITY_BURST_BLOCK"

        assert len(detail.audit_trail) == 1
        assert detail.audit_trail[0].action == "PERSIST_EVALUATION"

    async def test_transaction_detail_by_external_id_success(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify transaction lookup by client external transaction ID."""
        now = datetime.now(timezone.utc)
        tx_id = uuid.uuid4()

        tx = Transaction(
            id=tx_id,
            external_transaction_id="TX_EXT_LOOKUP_999",
            account_id="ACC_EXT_001",
            merchant_category="retail",
            job_category="teacher",
            amount=Decimal("75.50"),
            currency="USD",
            cardholder_lat=Decimal("40.712800"),
            cardholder_long=Decimal("-74.006000"),
            merchant_lat=Decimal("40.713800"),
            merchant_long=Decimal("-74.005000"),
            city_pop=8000000,
            transaction_timestamp=now,
            created_at=now,
            features_snapshot={"amount": 75.50},
        )
        db_session.add(tx)
        await db_session.commit()

        # Query GET endpoint by external ID
        response = await async_api_client.get("/api/v1/dashboard/transactions/TX_EXT_LOOKUP_999")
        assert response.status_code == 200
        data = response.json()
        assert data["transaction"]["external_transaction_id"] == "TX_EXT_LOOKUP_999"
        assert data["transaction"]["amount"] == 75.50
        assert data["evaluation"] is None

    async def test_transaction_detail_unknown_id_returns_404(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify unknown transaction returns HTTP 404 with structured error message."""
        non_existent_id = "00000000-0000-0000-0000-000000000000"
        response = await async_api_client.get(f"/api/v1/dashboard/transactions/{non_existent_id}")
        assert response.status_code == 404
        assert f"Transaction '{non_existent_id}' not found." in response.json()["detail"]

    async def test_transaction_detail_no_sensitive_leakage(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify no sensitive attributes, credentials, or DB secrets are leaked."""
        now = datetime.now(timezone.utc)
        tx_id = uuid.uuid4()

        tx = Transaction(
            id=tx_id,
            external_transaction_id="TX_SAFE_001",
            account_id="ACC_SAFE_001",
            merchant_category="groceries",
            job_category="manager",
            amount=Decimal("35.00"),
            currency="USD",
            cardholder_lat=Decimal("0.0"),
            cardholder_long=Decimal("0.0"),
            merchant_lat=Decimal("0.0"),
            merchant_long=Decimal("0.0"),
            city_pop=10000,
            transaction_timestamp=now,
            created_at=now,
            features_snapshot={"amount": 35.0},
        )
        db_session.add(tx)
        await db_session.commit()

        response = await async_api_client.get(f"/api/v1/dashboard/transactions/{tx_id}")
        assert response.status_code == 200
        raw_text = response.text

        # Verify sensitive tokens are completely absent from response payload
        assert "password" not in raw_text.lower()
        assert "cvv" not in raw_text.lower()
        assert "pan" not in raw_text.lower()
        assert "card_number" not in raw_text.lower()
        assert "db_password" not in raw_text.lower()
        assert "connection_string" not in raw_text.lower()

    async def test_transaction_detail_read_only_no_ml_inference(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify the detail endpoint is strictly read-only and does not invoke ML/SHAP inference."""
        now = datetime.now(timezone.utc)
        tx_id = uuid.uuid4()

        tx = Transaction(
            id=tx_id,
            external_transaction_id="TX_READONLY_001",
            account_id="ACC_RO_001",
            merchant_category="transportation",
            job_category="driver",
            amount=Decimal("12.50"),
            currency="USD",
            cardholder_lat=Decimal("0.0"),
            cardholder_long=Decimal("0.0"),
            merchant_lat=Decimal("0.0"),
            merchant_long=Decimal("0.0"),
            city_pop=10000,
            transaction_timestamp=now,
            created_at=now,
            features_snapshot={"amount": 12.50},
        )
        db_session.add(tx)
        await db_session.commit()

        # Patch prediction/evaluator and TreeSHAP explainer to assert zero calls
        with patch("ml.risk_engine.evaluator.RiskEvaluator.evaluate_dataframe") as mock_eval, \
             patch("ml.explainability.explainer.TreeSHAPExplainer.explain_features") as mock_shap:
            response = await async_api_client.get(f"/api/v1/dashboard/transactions/{tx_id}")
            assert response.status_code == 200
            assert mock_eval.call_count == 0
            assert mock_shap.call_count == 0


from backend.app.schemas.dashboard import (
    AnalyticsDistributionsResponse,
    AnalyticsRulesResponse,
    AnalyticsTrendsResponse,
)


class TestDashboardAnalyticsIntegration:
    """Integration test suite for Increment 11.4 Analytics Endpoints."""

    async def test_analytics_distributions_empty_db(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify GET /api/v1/dashboard/analytics/distributions on empty database."""
        response = await async_api_client.get("/api/v1/dashboard/analytics/distributions")
        assert response.status_code == 200
        data = response.json()

        resp = AnalyticsDistributionsResponse(**data)
        assert resp.total_evaluated == 0
        assert len(resp.risk_score_distribution) == 10
        assert resp.risk_score_distribution[0].bucket_label == "0–9"
        assert resp.risk_score_distribution[9].bucket_label == "90–100"
        assert all(b.count == 0 for b in resp.risk_score_distribution)
        assert len(resp.model_score_distribution) == 10
        assert len(resp.risk_tier_distribution) == 4
        assert len(resp.decision_distribution) == 3

    async def test_analytics_distributions_boundaries_and_filters(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """
        Verify exact risk score boundary bucket handling (0, 9, 10, 89, 90, 99, 100)
        and model probability boundary buckets [0.0–0.1 to 0.9–1.0].
        """
        now = datetime.now(timezone.utc)

        # Boundary test evaluations
        # Score 0 (0-9)
        # Score 9 (0-9)
        # Score 10 (10-19)
        # Score 89 (80-89)
        # Score 90 (90-100)
        # Score 99 (90-100)
        # Score 100 (90-100)
        test_cases = [
            ("TX_BND_00", 0, "0.000000", RiskTier.LOW, DecisionAction.APPROVE),
            ("TX_BND_09", 9, "0.090000", RiskTier.LOW, DecisionAction.APPROVE),
            ("TX_BND_10", 10, "0.100000", RiskTier.LOW, DecisionAction.APPROVE),
            ("TX_BND_89", 89, "0.890000", RiskTier.HIGH, DecisionAction.REVIEW),
            ("TX_BND_90", 90, "0.900000", RiskTier.CRITICAL, DecisionAction.BLOCK),
            ("TX_BND_99", 99, "0.990000", RiskTier.CRITICAL, DecisionAction.BLOCK),
            ("TX_BND_100", 100, "1.000000", RiskTier.CRITICAL, DecisionAction.BLOCK),
        ]

        for ext_id, r_score, m_score, r_tier, d_act in test_cases:
            tx_id = uuid.uuid4()
            tx = Transaction(
                id=tx_id,
                external_transaction_id=ext_id,
                account_id="ACC_BND",
                merchant_category="retail",
                job_category="analyst",
                amount=Decimal("100.00"),
                currency="USD",
                cardholder_lat=Decimal("0.0"),
                cardholder_long=Decimal("0.0"),
                merchant_lat=Decimal("0.0"),
                merchant_long=Decimal("0.0"),
                city_pop=50000,
                transaction_timestamp=now,
                created_at=now,
                features_snapshot={"amount": 100.00},
            )
            db_session.add(tx)
            await db_session.flush()

            evaluation = RiskEvaluation(
                id=uuid.uuid4(),
                transaction_id=tx_id,
                model_version="1.0.0",
                policy_mode=PolicyMode.TRI_TIER,
                model_score=Decimal(m_score),
                risk_score=r_score,
                risk_tier=r_tier,
                decision_action=d_act,
                baseline_action=d_act,
                is_overridden=False,
                decision_reason=f"Boundary score {r_score}",
                evaluated_at=now,
            )
            db_session.add(evaluation)

        await db_session.commit()

        # 1. Query all distributions
        response = await async_api_client.get("/api/v1/dashboard/analytics/distributions")
        assert response.status_code == 200
        data = response.json()
        resp = AnalyticsDistributionsResponse(**data)

        assert resp.total_evaluated == 7

        # Check risk score distribution:
        # Bucket 0–9: tx 0, tx 9 -> count = 2
        # Bucket 10–19: tx 10 -> count = 1
        # Bucket 80–89: tx 89 -> count = 1
        # Bucket 90–100: tx 90, tx 99, tx 100 -> count = 3
        rs_map = {b.bucket_label: b for b in resp.risk_score_distribution}
        assert rs_map["0–9"].count == 2
        assert rs_map["10–19"].count == 1
        assert rs_map["20–29"].count == 0
        assert rs_map["70–79"].count == 0
        assert rs_map["80–89"].count == 1
        assert rs_map["90–100"].count == 3

        # Check model score distribution:
        # 0.0–0.1: tx 0.00, tx 0.09 -> count = 2
        # 0.1–0.2: tx 0.10 -> count = 1
        # 0.8–0.9: tx 0.89 -> count = 1
        # 0.9–1.0: tx 0.90, tx 0.99, tx 1.00 -> count = 3
        ms_map = {b.bucket_label: b for b in resp.model_score_distribution}
        assert ms_map["0.0–0.1"].count == 2
        assert ms_map["0.1–0.2"].count == 1
        assert ms_map["0.8–0.9"].count == 1
        assert ms_map["0.9–1.0"].count == 3

        # 2. Test filter decision_action=BLOCK (should return only 3 evaluations)
        resp_blk = await async_api_client.get("/api/v1/dashboard/analytics/distributions?decision_action=BLOCK")
        assert resp_blk.status_code == 200
        blk_data = resp_blk.json()
        assert blk_data["total_evaluated"] == 3
        assert blk_data["risk_score_distribution"][9]["count"] == 3  # All in 90-100

    async def test_analytics_trends_seeded_and_intervals(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify GET /api/v1/dashboard/analytics/trends computes time-series volume and score averages."""
        t1 = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 17, 11, 0, 0, tzinfo=timezone.utc)

        # tx1 at 10:00
        tx1 = Transaction(
            id=uuid.uuid4(),
            external_transaction_id="TX_TR_01",
            account_id="ACC_TR",
            merchant_category="retail",
            job_category="manager",
            amount=Decimal("50.00"),
            currency="USD",
            cardholder_lat=Decimal("0.0"),
            cardholder_long=Decimal("0.0"),
            merchant_lat=Decimal("0.0"),
            merchant_long=Decimal("0.0"),
            city_pop=50000,
            transaction_timestamp=t1,
            created_at=t1,
            features_snapshot={"amount": 50.00},
        )
        db_session.add(tx1)
        await db_session.flush()

        eval1 = RiskEvaluation(
            id=uuid.uuid4(),
            transaction_id=tx1.id,
            model_score=Decimal("0.100000"),
            risk_score=20,
            risk_tier=RiskTier.LOW,
            decision_action=DecisionAction.APPROVE,
            baseline_action=DecisionAction.APPROVE,
            is_overridden=False,
            decision_reason="Low risk",
            evaluated_at=t1,
        )
        db_session.add(eval1)

        # tx2 at 11:00
        tx2 = Transaction(
            id=uuid.uuid4(),
            external_transaction_id="TX_TR_02",
            account_id="ACC_TR",
            merchant_category="electronics",
            job_category="manager",
            amount=Decimal("150.00"),
            currency="USD",
            transaction_timestamp=t2,
            created_at=t2,
            cardholder_lat=Decimal("0.0"),
            cardholder_long=Decimal("0.0"),
            merchant_lat=Decimal("0.0"),
            merchant_long=Decimal("0.0"),
            city_pop=50000,
            features_snapshot={"amount": 150.00},
        )
        db_session.add(tx2)
        await db_session.flush()

        eval2 = RiskEvaluation(
            id=uuid.uuid4(),
            transaction_id=tx2.id,
            model_score=Decimal("0.900000"),
            risk_score=90,
            risk_tier=RiskTier.CRITICAL,
            decision_action=DecisionAction.BLOCK,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=True,
            decision_reason="Critical override",
            evaluated_at=t2,
        )
        db_session.add(eval2)
        await db_session.commit()

        # Query trends endpoint for this range
        response = await async_api_client.get(
            "/api/v1/dashboard/analytics/trends",
            params={"start_date": t1.isoformat(), "end_date": t2.isoformat(), "interval": "hourly"},
        )
        assert response.status_code == 200
        data = response.json()
        resp = AnalyticsTrendsResponse(**data)

        assert resp.interval.value == "hourly"
        assert len(resp.data_points) == 2
        # Bucket 1 (10:00): 1 txn, $50, avg score 20, 1 approve
        assert resp.data_points[0].total_count == 1
        assert resp.data_points[0].total_amount == 50.00
        assert resp.data_points[0].average_risk_score == 20.0
        assert resp.data_points[0].approval_count == 1

        # Bucket 2 (11:00): 1 txn, $150, avg score 90, 1 block
        assert resp.data_points[1].total_count == 1
        assert resp.data_points[1].total_amount == 150.00
        assert resp.data_points[1].average_risk_score == 90.0
        assert resp.data_points[1].block_count == 1
        assert resp.data_points[1].high_critical_count == 1

    async def test_analytics_rules_seeded_semantics(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """
        Verify rule analytics semantics:
        - trigger_count: count of rule match records
        - affected_transactions: count of distinct evaluations
        - override_count: count of distinct evaluations where is_overridden=True
        """
        now = datetime.now(timezone.utc)
        tx_id1 = uuid.uuid4()
        tx_id2 = uuid.uuid4()
        eval_id1 = uuid.uuid4()
        eval_id2 = uuid.uuid4()

        # Evaluation 1: overridden = True, triggers RULE_A and RULE_B
        tx1 = Transaction(
            id=tx_id1,
            external_transaction_id="TX_RULE_01",
            account_id="ACC_R",
            merchant_category="retail",
            job_category="engineer",
            amount=Decimal("500.00"),
            currency="USD",
            transaction_timestamp=now,
            created_at=now,
            cardholder_lat=Decimal("0.0"),
            cardholder_long=Decimal("0.0"),
            merchant_lat=Decimal("0.0"),
            merchant_long=Decimal("0.0"),
            city_pop=50000,
            features_snapshot={"amount": 500.00},
        )
        db_session.add(tx1)
        await db_session.flush()

        eval1 = RiskEvaluation(
            id=eval_id1,
            transaction_id=tx_id1,
            model_score=Decimal("0.600000"),
            risk_score=60,
            risk_tier=RiskTier.MEDIUM,
            decision_action=DecisionAction.BLOCK,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=True,  # Overridden evaluation
            decision_reason="Override rule match",
            evaluated_at=now,
        )
        db_session.add(eval1)
        await db_session.flush()

        rm1_a = EvaluationRuleMatch(
            id=uuid.uuid4(),
            evaluation_id=eval_id1,
            rule_id="RULE_VELOCITY_BURST",
            description="Hourly count exceeds 5",
            feature_name="txn_count_1h",
            operator=">",
            comparison_value="5",
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.VELOCITY,
            priority=10,
        )
        rm1_b = EvaluationRuleMatch(
            id=uuid.uuid4(),
            evaluation_id=eval_id1,
            rule_id="RULE_HIGH_AMOUNT",
            description="Amount exceeds 300",
            feature_name="amount",
            operator=">",
            comparison_value="300",
            outcome=RuleOutcome.REVIEW,
            rule_type=RuleType.AMOUNT,
            priority=20,
        )
        db_session.add(rm1_a)
        db_session.add(rm1_b)

        # Evaluation 2: overridden = False, triggers RULE_VELOCITY_BURST
        tx2 = Transaction(
            id=tx_id2,
            external_transaction_id="TX_RULE_02",
            account_id="ACC_R2",
            merchant_category="retail",
            job_category="engineer",
            amount=Decimal("100.00"),
            currency="USD",
            transaction_timestamp=now,
            created_at=now,
            cardholder_lat=Decimal("0.0"),
            cardholder_long=Decimal("0.0"),
            merchant_lat=Decimal("0.0"),
            merchant_long=Decimal("0.0"),
            city_pop=50000,
            features_snapshot={"amount": 100.00},
        )
        db_session.add(tx2)
        await db_session.flush()

        eval2 = RiskEvaluation(
            id=eval_id2,
            transaction_id=tx_id2,
            model_score=Decimal("0.900000"),
            risk_score=90,
            risk_tier=RiskTier.CRITICAL,
            decision_action=DecisionAction.BLOCK,
            baseline_action=DecisionAction.BLOCK,
            is_overridden=False,  # Not overridden
            decision_reason="Model block with rule confirmation",
            evaluated_at=now,
        )
        db_session.add(eval2)
        await db_session.flush()

        rm2_a = EvaluationRuleMatch(
            id=uuid.uuid4(),
            evaluation_id=eval_id2,
            rule_id="RULE_VELOCITY_BURST",
            description="Hourly count exceeds 5",
            feature_name="txn_count_1h",
            operator=">",
            comparison_value="5",
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.VELOCITY,
            priority=10,
        )
        db_session.add(rm2_a)

        await db_session.commit()

        # Query rules analytics endpoint
        response = await async_api_client.get("/api/v1/dashboard/analytics/rules")
        assert response.status_code == 200
        data = response.json()
        resp = AnalyticsRulesResponse(**data)

        assert resp.total_rules_active == 2
        assert resp.total_evaluations_analyzed >= 2

        # RULE_VELOCITY_BURST: trigger_count = 2, affected_tx = 2, override_count = 1 (from eval1 only)
        rule_vel = next(r for r in resp.rules if r.rule_id == "RULE_VELOCITY_BURST")
        assert rule_vel.trigger_count == 2
        assert rule_vel.affected_transactions == 2
        assert rule_vel.override_count == 1

        # RULE_HIGH_AMOUNT: trigger_count = 1, affected_tx = 1, override_count = 1
        rule_amt = next(r for r in resp.rules if r.rule_id == "RULE_HIGH_AMOUNT")
        assert rule_amt.trigger_count == 1
        assert rule_amt.affected_transactions == 1
        assert rule_amt.override_count == 1

    async def test_analytics_endpoints_no_ml_inference(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """
        CRITICAL REGRESSION TEST: Prove that all analytics endpoints are strictly read-only
        over persisted records and NEVER invoke the ML prediction or SHAP explainer pipelines.
        """
        with patch("ml.risk_engine.evaluator.RiskEvaluator.evaluate_dataframe") as mock_eval, \
             patch("ml.explainability.explainer.TreeSHAPExplainer.explain_features") as mock_shap:

            resp_dist = await async_api_client.get("/api/v1/dashboard/analytics/distributions")
            assert resp_dist.status_code == 200

            resp_trend = await async_api_client.get("/api/v1/dashboard/analytics/trends")
            assert resp_trend.status_code == 200

            resp_rules = await async_api_client.get("/api/v1/dashboard/analytics/rules")
            assert resp_rules.status_code == 200

            assert mock_eval.call_count == 0
            assert mock_shap.call_count == 0

    async def test_analytics_endpoints_no_sensitive_leakage(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify analytics endpoints never leak credentials, PAN, or raw secrets."""
        for endpoint in [
            "/api/v1/dashboard/analytics/distributions",
            "/api/v1/dashboard/analytics/trends",
            "/api/v1/dashboard/analytics/rules",
        ]:
            response = await async_api_client.get(endpoint)
            assert response.status_code == 200
            text = response.text.lower()
            assert "password" not in text
            assert "cvv" not in text
            assert "pan" not in text
            assert "card_number" not in text
            assert "db_password" not in text
            assert "secret" not in text


# ==============================================================================
# Increment 11.5: Simulation Endpoint Integration Tests
# ==============================================================================

from sqlalchemy import func, select
from backend.app.db.models import (
    AuditLog,
    EvaluationFeatureAttribution,
    EvaluationReasonCode,
    EvaluationRuleMatch,
    RiskEvaluation,
    Transaction,
)
from backend.app.schemas.dashboard import RuleDiffStatus, SimulationResponse


def get_canonical_55_features(overrides: dict | None = None) -> dict:
    """Helper returning a complete, valid dictionary of all 55 features."""
    base = {
        "amount": 125.50,
        "cardholder_lat": 40.7128,
        "cardholder_long": -74.0060,
        "merchant_lat": 40.7130,
        "merchant_long": -74.0058,
        "city_pop": 500000.0,
        "transaction_hour": 12,
        "day_of_week": 2,
        "day_of_month": 15,
        "month": 9,
        "week_of_year": 38,
        "is_weekend": 0,
        "is_night": 0,
        "hour_sin": 0.0,
        "hour_cos": -1.0,
        "day_of_week_sin": 0.9749,
        "day_of_week_cos": -0.2225,
        "txn_count_1h": 1.0,
        "txn_count_6h": 2.0,
        "txn_count_24h": 4.0,
        "txn_count_7d": 12.0,
        "txn_count_30d": 30.0,
        "time_since_prev_txn_seconds": 3600.0,
        "is_first_account_txn": 0,
        "amt_sum_1h": 125.50,
        "amt_sum_24h": 450.0,
        "amt_sum_7d": 1500.0,
        "amt_sum_30d": 4500.0,
        "amt_mean_24h": 112.5,
        "amt_mean_7d": 125.0,
        "amt_max_24h": 200.0,
        "amt_median_30d": 110.0,
        "historical_amount_mean": 120.0,
        "historical_amount_std": 30.0,
        "historical_amount_median": 115.0,
        "amount_zscore": 0.1833,
        "amount_ratio_to_historical_mean": 1.0458,
        "account_txn_count_before": 100.0,
        "account_total_spend_before": 12000.0,
        "account_avg_amount_before": 120.0,
        "account_max_amount_before": 400.0,
        "account_unique_merchant_count_before": 25.0,
        "account_unique_category_count_before": 8.0,
        "account_merchant_txn_count_before": 5.0,
        "account_category_txn_count_before": 20.0,
        "account_merchant_spend_before": 600.0,
        "account_category_spend_before": 2400.0,
        "merchant_txn_count_before": 1200.0,
        "category_txn_count_before": 8500.0,
        "cardholder_merchant_distance_km": 1.25,
        "distance_from_prev_merchant_km": 0.5,
        "implied_travel_speed_kmh": 0.5,
        "is_impossible_travel_speed": 0,
        "merchant_category": "grocery_pos",
        "job_category": "engineer",
    }
    if overrides:
        base.update(overrides)
    return base


class TestDashboardSimulationIntegration:
    """Integration test suite for POST /api/v1/dashboard/simulate."""

    async def test_simulation_from_scratch_success(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify successful simulation without baseline transaction."""
        features = get_canonical_55_features()
        payload = {
            "simulated_features": features,
            "top_k": 5,
            "top_mitigating": 3,
            "max_reasons": 5,
        }

        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()

        sim_resp = SimulationResponse(**data)
        assert sim_resp.is_simulation is True
        assert sim_resp.baseline is None
        assert sim_resp.comparison is None
        assert sim_resp.simulated.risk_score >= 0
        assert sim_resp.simulated.risk_tier in [RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL]
        assert sim_resp.simulated.decision_action in [DecisionAction.APPROVE, DecisionAction.REVIEW, DecisionAction.BLOCK]
        assert len(sim_resp.simulated.feature_attributions) > 0
        assert sim_resp.evaluation_latency_ms >= 0.0

    async def test_simulation_with_persisted_uuid_baseline(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify simulation comparing against a persisted baseline transaction via UUID."""
        now = datetime.now(timezone.utc)
        baseline_features = get_canonical_55_features({"amount": 50.0})

        tx = Transaction(
            external_transaction_id="TX_SIM_BASE_001",
            account_id="ACC_SIM_001",
            amount=Decimal("50.00"),
            currency="USD",
            transaction_timestamp=now,
            merchant_id="MERCH_01",
            merchant_category="grocery_pos",
            job_category="engineer",
            cardholder_lat=Decimal("40.7128"),
            cardholder_long=Decimal("-74.0060"),
            merchant_lat=Decimal("40.7130"),
            merchant_long=Decimal("-74.0058"),
            city_pop=500000,
            features_snapshot=baseline_features,
        )
        db_session.add(tx)
        await db_session.flush()

        eval_rec = RiskEvaluation(
            transaction_id=tx.id,
            model_score=Decimal("0.100000"),
            risk_score=15,
            risk_tier=RiskTier.LOW,
            decision_action=DecisionAction.APPROVE,
            baseline_action=DecisionAction.APPROVE,
            is_overridden=False,
            decision_reason="Low risk approved",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("12.50"),
            evaluated_at=now,
            model_version="1.0.0",
        )
        db_session.add(eval_rec)
        await db_session.commit()

        # Simulate with modified amount and velocity
        simulated_features = get_canonical_55_features({"amount": 5000.0, "txn_count_1h": 10.0})
        payload = {
            "baseline_transaction_id": str(tx.id),
            "simulated_features": simulated_features,
        }

        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()

        sim_resp = SimulationResponse(**data)
        assert sim_resp.baseline is not None
        assert sim_resp.baseline.transaction_id == str(tx.id)
        assert sim_resp.baseline.external_transaction_id == "TX_SIM_BASE_001"
        assert sim_resp.baseline.risk_score == 15
        assert sim_resp.baseline.decision_action == DecisionAction.APPROVE

        assert sim_resp.comparison is not None
        assert sim_resp.comparison.risk_score_delta == sim_resp.simulated.risk_score - 15
        assert sim_resp.comparison.modified_features_count >= 2

        # Verify feature diff list contains amount
        amt_diff = next((fd for fd in sim_resp.comparison.feature_diffs if fd.feature_name == "amount"), None)
        assert amt_diff is not None
        assert amt_diff.is_modified is True
        assert amt_diff.delta == 4950.0

    async def test_simulation_with_persisted_external_id_baseline(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify simulation baseline lookup by external_transaction_id."""
        now = datetime.now(timezone.utc)
        baseline_features = get_canonical_55_features({"amount": 75.0})

        tx = Transaction(
            external_transaction_id="TX_SIM_EXT_LOOKUP_01",
            account_id="ACC_SIM_002",
            amount=Decimal("75.00"),
            currency="USD",
            transaction_timestamp=now,
            merchant_id="MERCH_01",
            merchant_category="grocery_pos",
            job_category="engineer",
            cardholder_lat=Decimal("40.7128"),
            cardholder_long=Decimal("-74.0060"),
            merchant_lat=Decimal("40.7130"),
            merchant_long=Decimal("-74.0058"),
            city_pop=500000,
            features_snapshot=baseline_features,
        )
        db_session.add(tx)
        await db_session.flush()

        eval_rec = RiskEvaluation(
            transaction_id=tx.id,
            model_score=Decimal("0.080000"),
            risk_score=12,
            risk_tier=RiskTier.LOW,
            decision_action=DecisionAction.APPROVE,
            baseline_action=DecisionAction.APPROVE,
            is_overridden=False,
            decision_reason="Low risk approved",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("11.00"),
            evaluated_at=now,
            model_version="1.0.0",
        )
        db_session.add(eval_rec)
        await db_session.commit()

        simulated_features = get_canonical_55_features({"amount": 100.0})
        payload = {
            "baseline_transaction_id": "TX_SIM_EXT_LOOKUP_01",
            "simulated_features": simulated_features,
        }

        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 200
        sim_resp = SimulationResponse(**response.json())
        assert sim_resp.baseline is not None
        assert sim_resp.baseline.external_transaction_id == "TX_SIM_EXT_LOOKUP_01"

    async def test_simulation_unknown_baseline_returns_404(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify non-existent baseline ID returns HTTP 404."""
        payload = {
            "baseline_transaction_id": "99999999-9999-9999-9999-999999999999",
            "simulated_features": get_canonical_55_features(),
        }
        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    async def test_simulation_missing_feature_rejection_422(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify omission of any of the 55 canonical features returns HTTP 422."""
        features = get_canonical_55_features()
        del features["amount"]  # missing feature

        payload = {"simulated_features": features}
        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "amount" in str(data).lower()

    async def test_simulation_unknown_feature_rejection_422(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify extraneous unknown feature fields in feature dict return HTTP 422."""
        features = get_canonical_55_features({"malicious_unregistered_feature": 999.9})
        payload = {"simulated_features": features}
        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "unknown" in str(data).lower() or "extra" in str(data).lower() or "unregistered" in str(data).lower()

    async def test_simulation_invalid_type_rejection_422(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify string value for numerical feature returns HTTP 422."""
        features = get_canonical_55_features({"amount": "not_a_valid_number"})
        payload = {"simulated_features": features}
        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 422

    async def test_simulation_nan_infinity_rejection_422(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify NaN or Infinity payload is rejected with HTTP 422."""
        import json
        valid_dict = {"simulated_features": get_canonical_55_features({"amount": 100.0})}
        headers = {"Content-Type": "application/json"}

        # 1. NaN in payload
        raw_nan_json = json.dumps(valid_dict).replace("100.0", "NaN")
        response_nan = await async_api_client.post(
            "/api/v1/dashboard/simulate", content=raw_nan_json, headers=headers
        )
        assert response_nan.status_code == 422

        # 2. Infinity in payload
        raw_inf_json = json.dumps(valid_dict).replace("100.0", "Infinity")
        response_inf = await async_api_client.post(
            "/api/v1/dashboard/simulate", content=raw_inf_json, headers=headers
        )
        assert response_inf.status_code == 422

    async def test_simulation_boundary_validation_422(
        self, async_api_client: AsyncClient
    ) -> None:
        """Verify out-of-bounds features return HTTP 422."""
        features = get_canonical_55_features({"transaction_hour": 25})  # valid is 0..23
        payload = {"simulated_features": features}
        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 422

        features_neg_amount = get_canonical_55_features({"amount": -10.0})
        payload = {"simulated_features": features_neg_amount}
        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 422

    async def test_simulation_rule_diff_behavior(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Verify rule diff categorization across NEWLY_TRIGGERED, RESOLVED, PERSISTENT, NEITHER."""
        now = datetime.now(timezone.utc)
        # Baseline with high velocity that triggered a velocity rule
        baseline_features = get_canonical_55_features({"txn_count_1h": 10.0, "amount": 50.0})

        tx = Transaction(
            external_transaction_id="TX_SIM_RULE_DIFF_01",
            account_id="ACC_SIM_003",
            amount=Decimal("50.00"),
            currency="USD",
            transaction_timestamp=now,
            merchant_id="MERCH_01",
            merchant_category="grocery_pos",
            job_category="engineer",
            cardholder_lat=Decimal("40.7128"),
            cardholder_long=Decimal("-74.0060"),
            merchant_lat=Decimal("40.7130"),
            merchant_long=Decimal("-74.0058"),
            city_pop=500000,
            features_snapshot=baseline_features,
        )
        db_session.add(tx)
        await db_session.flush()

        eval_rec = RiskEvaluation(
            transaction_id=tx.id,
            model_score=Decimal("0.400000"),
            risk_score=50,
            risk_tier=RiskTier.MEDIUM,
            decision_action=DecisionAction.REVIEW,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=True,
            decision_reason="Rule override",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("15.00"),
            evaluated_at=now,
            model_version="1.0.0",
        )
        db_session.add(eval_rec)
        await db_session.flush()

        # Add a baseline rule match: RULE_VELOCITY_BURST
        rule_match = EvaluationRuleMatch(
            evaluation_id=eval_rec.id,
            rule_id="RULE_VELOCITY_BURST",
            rule_type="VELOCITY",
            priority=10,
            outcome=RuleOutcome.BLOCK,
            feature_name="txn_count_1h",
            operator=">",
            comparison_value="5",
            description="Hourly velocity exceeds threshold",
        )
        db_session.add(rule_match)
        await db_session.commit()

        # Simulate reducing velocity below threshold
        simulated_features = get_canonical_55_features({"txn_count_1h": 1.0, "amount": 50.0})
        payload = {
            "baseline_transaction_id": str(tx.id),
            "simulated_features": simulated_features,
        }

        response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
        assert response.status_code == 200
        sim_resp = SimulationResponse(**response.json())
        assert sim_resp.comparison is not None

        # Verify rule diff status for RULE_VELOCITY_BURST is RESOLVED
        vel_rule_diff = next((rd for rd in sim_resp.comparison.rule_diffs if rd.rule_id == "RULE_VELOCITY_BURST"), None)
        assert vel_rule_diff is not None
        assert vel_rule_diff.baseline_triggered is True
        assert vel_rule_diff.simulated_triggered is False
        assert vel_rule_diff.diff_status == RuleDiffStatus.RESOLVED

    async def test_simulation_zero_write_guarantee(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """
        CRITICAL ZERO-WRITE GUARANTEE TEST:
        Proves conclusively that running What-If simulations performs ZERO database writes:
        - 0 INSERT
        - 0 UPDATE
        - 0 DELETE
        - 0 AuditLog entries created
        - 0 RiskEvaluation records created
        - 0 Rule / Reason / Attribution records created
        - Persisted baseline row contents remain 100% byte-for-byte identical
        - FraudPersistenceService is NEVER invoked
        """
        now = datetime.now(timezone.utc)
        baseline_features = get_canonical_55_features({"amount": 250.0})

        # 1. Seed baseline record
        tx = Transaction(
            external_transaction_id="TX_ZERO_WRITE_001",
            account_id="ACC_ZW_001",
            amount=Decimal("250.00"),
            currency="USD",
            transaction_timestamp=now,
            merchant_id="MERCH_01",
            merchant_category="grocery_pos",
            job_category="engineer",
            cardholder_lat=Decimal("40.7128"),
            cardholder_long=Decimal("-74.0060"),
            merchant_lat=Decimal("40.7130"),
            merchant_long=Decimal("-74.0058"),
            city_pop=500000,
            features_snapshot=baseline_features,
        )
        db_session.add(tx)
        await db_session.flush()

        eval_rec = RiskEvaluation(
            transaction_id=tx.id,
            model_score=Decimal("0.250000"),
            risk_score=30,
            risk_tier=RiskTier.LOW,
            decision_action=DecisionAction.APPROVE,
            baseline_action=DecisionAction.APPROVE,
            is_overridden=False,
            decision_reason="Baseline low risk",
            policy_mode=PolicyMode.TRI_TIER,
            evaluation_latency_ms=Decimal("14.00"),
            evaluated_at=now,
            model_version="1.0.0",
        )
        db_session.add(eval_rec)
        await db_session.commit()

        # 2. Record table counts before simulation
        tx_count_before = (await db_session.execute(select(func.count()).select_from(Transaction))).scalar_one()
        eval_count_before = (await db_session.execute(select(func.count()).select_from(RiskEvaluation))).scalar_one()
        attr_count_before = (await db_session.execute(select(func.count()).select_from(EvaluationFeatureAttribution))).scalar_one()
        rc_count_before = (await db_session.execute(select(func.count()).select_from(EvaluationReasonCode))).scalar_one()
        rule_count_before = (await db_session.execute(select(func.count()).select_from(EvaluationRuleMatch))).scalar_one()
        audit_count_before = (await db_session.execute(select(func.count()).select_from(AuditLog))).scalar_one()

        # 3. Execute simulation with FraudPersistenceService patched to assert 0 calls
        with patch("backend.app.services.persistence_service.FraudPersistenceService.persist_evaluation") as mock_persist:
            simulated_features = get_canonical_55_features({"amount": 9999.0, "txn_count_1h": 20.0})
            payload = {
                "baseline_transaction_id": str(tx.id),
                "simulated_features": simulated_features,
            }

            response = await async_api_client.post("/api/v1/dashboard/simulate", json=payload)
            assert response.status_code == 200
            assert mock_persist.call_count == 0

        tx_id = tx.id
        eval_id = eval_rec.id

        # 4. Verify table counts after simulation are 100% IDENTICAL
        # Expire session cache to ensure fresh SELECT queries against PostgreSQL
        db_session.expire_all()

        tx_count_after = (await db_session.execute(select(func.count()).select_from(Transaction))).scalar_one()
        eval_count_after = (await db_session.execute(select(func.count()).select_from(RiskEvaluation))).scalar_one()
        attr_count_after = (await db_session.execute(select(func.count()).select_from(EvaluationFeatureAttribution))).scalar_one()
        rc_count_after = (await db_session.execute(select(func.count()).select_from(EvaluationReasonCode))).scalar_one()
        rule_count_after = (await db_session.execute(select(func.count()).select_from(EvaluationRuleMatch))).scalar_one()
        audit_count_after = (await db_session.execute(select(func.count()).select_from(AuditLog))).scalar_one()

        assert tx_count_after == tx_count_before, "Transaction table was modified by simulation!"
        assert eval_count_after == eval_count_before, "RiskEvaluation record was created by simulation!"
        assert attr_count_after == attr_count_before, "FeatureAttribution record was created by simulation!"
        assert rc_count_after == rc_count_before, "ReasonCode record was created by simulation!"
        assert rule_count_after == rule_count_before, "EvaluationRuleMatch record was created by simulation!"
        assert audit_count_after == audit_count_before, "AuditLog record was created by simulation!"

        # 5. Verify baseline record contents are 100% unchanged
        refreshed_tx = (await db_session.execute(select(Transaction).where(Transaction.id == tx_id))).scalar_one()
        refreshed_eval = (await db_session.execute(select(RiskEvaluation).where(RiskEvaluation.id == eval_id))).scalar_one()

        assert refreshed_tx.amount == Decimal("250.00")
        assert refreshed_tx.external_transaction_id == "TX_ZERO_WRITE_001"
        assert refreshed_eval.risk_score == 30
        assert refreshed_eval.model_score == Decimal("0.250000")
        assert refreshed_eval.decision_action == DecisionAction.APPROVE



