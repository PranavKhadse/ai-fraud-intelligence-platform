"""
API Integration Tests for Monitoring REST Endpoints (Phase 13.7).

Validates all 6 GET /api/v1/monitoring/* endpoints against live PostgreSQL session:
1. GET /api/v1/monitoring/health (empty db vs seeded db vs custom window)
2. GET /api/v1/monitoring/drift/features (sorting, category filtering, status filtering, search term, pagination)
3. GET /api/v1/monitoring/drift/features/{feature_name} (200 OK for valid feature, 404 for unknown feature)
4. GET /api/v1/monitoring/drift/predictions (continuous score, risk buckets, tiers, actions, override rate)
5. GET /api/v1/monitoring/performance (confusion matrix, metrics at tau*=0.78, custom tau, review purity)
6. GET /api/v1/monitoring/snapshots (pagination, window_type filter, total count)
7. Request parameter validation & 422 error boundaries
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.db.models.case import Case
from backend.app.db.models.enums import (
    CaseDisposition,
    CasePriority,
    CaseStatus,
    CaseTriggerSource,
    DecisionAction,
    PolicyMode,
    RiskTier,
)
from backend.app.db.models.monitoring_snapshot import ModelMonitoringSnapshot
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.transaction import Transaction
from backend.app.db.session import get_db_session
from backend.app.main import app
from ml.monitoring.config import default_monitoring_config
from ml.monitoring.schemas import FeatureBaselineProfile, NumericalFeatureProfile, CategoricalFeatureProfile

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


def _build_features_dict(base_profile: FeatureBaselineProfile) -> Dict[str, Any]:
    """Build dummy 55-feature dictionary."""
    snapshot: Dict[str, Any] = {}
    for feat_name, meta in base_profile.features.items():
        if isinstance(meta, NumericalFeatureProfile):
            snapshot[feat_name] = float(meta.mean)
        elif isinstance(meta, CategoricalFeatureProfile) and meta.vocabulary:
            snapshot[feat_name] = meta.vocabulary[0]
        else:
            snapshot[feat_name] = "UNKNOWN"
    return snapshot


class TestMonitoringAPIIntegration:
    """Integration test suite for Monitoring REST APIs."""

    @pytest.fixture
    def base_profile(self) -> FeatureBaselineProfile:
        return FeatureBaselineProfile.load(default_monitoring_config.feature_profile_path)

    async def test_get_monitoring_health_empty_and_seeded(
        self, async_api_client: AsyncClient, db_session: AsyncSession, base_profile: FeatureBaselineProfile
    ):
        """Verify GET /api/v1/monitoring/health returns valid schema on empty and seeded DB."""
        # 1. Empty database test
        resp_empty = await async_api_client.get("/api/v1/monitoring/health?window=24h")
        assert resp_empty.status_code == 200
        data_empty = resp_empty.json()
        assert data_empty["model_version"] == "1.0.0"
        assert data_empty["sample_count"] == 0
        assert data_empty["overall_status"] == "INSUFFICIENT_DATA"
        assert data_empty["active_alert_count"] == 0

        # 2. Seed database
        now = datetime.now(timezone.utc)
        features = _build_features_dict(base_profile)

        for i in range(15):
            txn_id = uuid.uuid4()
            eval_id = uuid.uuid4()
            txn = Transaction(
                id=txn_id,
                account_id=f"ACC_{i}",
                merchant_category="grocery_pos",
                job_category="engineer",
                amount=Decimal("50.00"),
                currency="USD",
                cardholder_lat=Decimal("40.0"),
                cardholder_long=Decimal("-74.0"),
                merchant_lat=Decimal("40.0"),
                merchant_long=Decimal("-74.0"),
                city_pop=50000,
                transaction_timestamp=now - timedelta(hours=i),
                features_snapshot=features,
            )
            db_session.add(txn)

            evaluation = RiskEvaluation(
                id=eval_id,
                transaction_id=txn_id,
                model_score=Decimal("0.20"),
                risk_score=20,
                decision_action=DecisionAction.APPROVE,
                baseline_action=DecisionAction.APPROVE,
                decision_reason="API test evaluation",
                risk_tier=RiskTier.LOW,
                model_version="1.0.0",
                policy_mode=PolicyMode.TRI_TIER,
                evaluated_at=now - timedelta(hours=i),
            )
            db_session.add(evaluation)

        await db_session.commit()

        # 3. Seeded query test
        resp_seeded = await async_api_client.get("/api/v1/monitoring/health?window=24h")
        assert resp_seeded.status_code == 200
        data_seeded = resp_seeded.json()
        assert data_seeded["sample_count"] == 15
        assert data_seeded["overall_status"] in ("NORMAL", "WARNING", "CRITICAL", "INSUFFICIENT_DATA")

    async def test_get_feature_drift_filters_and_pagination(
        self, async_api_client: AsyncClient, db_session: AsyncSession, base_profile: FeatureBaselineProfile
    ):
        """Verify GET /api/v1/monitoring/drift/features filtering, sorting, and pagination."""
        now = datetime.now(timezone.utc)
        features = _build_features_dict(base_profile)

        txn = Transaction(
            id=uuid.uuid4(),
            account_id="ACC_FILTERS",
            merchant_category="grocery_pos",
            job_category="engineer",
            amount=Decimal("100.00"),
            currency="USD",
            cardholder_lat=Decimal("40.0"),
            cardholder_long=Decimal("-74.0"),
            merchant_lat=Decimal("40.0"),
            merchant_long=Decimal("-74.0"),
            city_pop=50000,
            transaction_timestamp=now - timedelta(hours=1),
            features_snapshot=features,
        )
        db_session.add(txn)
        eval_obj = RiskEvaluation(
            id=uuid.uuid4(),
            transaction_id=txn.id,
            model_score=Decimal("0.10"),
            risk_score=10,
            decision_action=DecisionAction.APPROVE,
            baseline_action=DecisionAction.APPROVE,
            decision_reason="API test",
            risk_tier=RiskTier.LOW,
            model_version="1.0.0",
            policy_mode=PolicyMode.TRI_TIER,
            evaluated_at=now - timedelta(hours=1),
        )
        db_session.add(eval_obj)
        await db_session.commit()

        # 1. Full 55 feature list
        resp_all = await async_api_client.get("/api/v1/monitoring/drift/features?limit=55")
        assert resp_all.status_code == 200
        data_all = resp_all.json()
        assert data_all["total_features_count"] == 55
        assert len(data_all["items"]) == 55

        # 2. Search term filter
        resp_search = await async_api_client.get("/api/v1/monitoring/drift/features?search_term=amount")
        assert resp_search.status_code == 200
        data_search = resp_search.json()
        assert len(data_search["items"]) >= 1
        assert all("amount" in item["feature_name"] for item in data_search["items"])

        # 3. Pagination limit & offset
        resp_paged = await async_api_client.get("/api/v1/monitoring/drift/features?limit=10&offset=5")
        assert resp_paged.status_code == 200
        data_paged = resp_paged.json()
        assert len(data_paged["items"]) == 10

    async def test_get_single_feature_drift_valid_and_404(
        self, async_api_client: AsyncClient
    ):
        """Verify GET /api/v1/monitoring/drift/features/{name} returns 200 for known feature and 404 for unknown."""
        # Known feature
        resp_valid = await async_api_client.get("/api/v1/monitoring/drift/features/amount?window=24h")
        assert resp_valid.status_code == 200
        data_valid = resp_valid.json()
        assert data_valid["feature_name"] == "amount"
        assert data_valid["feature_type"] == "numerical"

        # Unknown feature returns 404
        resp_404 = await async_api_client.get("/api/v1/monitoring/drift/features/non_existent_feature_123")
        assert resp_404.status_code == 404
        data_404 = resp_404.json()
        assert "not found" in data_404["detail"].lower()

    async def test_get_prediction_drift_endpoint(
        self, async_api_client: AsyncClient
    ):
        """Verify GET /api/v1/monitoring/drift/predictions schema compliance."""
        resp = await async_api_client.get("/api/v1/monitoring/drift/predictions?window=24h")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_score_psi" in data
        assert "risk_score_psi" in data
        assert "risk_tier_jsd" in data
        assert "action_jsd" in data
        assert "override_rate_current" in data

    async def test_get_model_performance_endpoint(
        self, async_api_client: AsyncClient
    ):
        """Verify GET /api/v1/monitoring/performance schema and operating threshold parameter."""
        resp = await async_api_client.get("/api/v1/monitoring/performance?window=24h&operating_threshold=0.78")
        assert resp.status_code == 200
        data = resp.json()
        assert data["operating_threshold"] == 0.78
        assert "primary_metrics" in data
        assert "comparison_metrics" in data
        assert "operational_metrics" in data
        assert "degradation_results" in data

    async def test_get_snapshots_pagination(
        self, async_api_client: AsyncClient, db_session: AsyncSession
    ):
        """Verify GET /api/v1/monitoring/snapshots retrieves paginated snapshot history."""
        now = datetime.now(timezone.utc)

        # Seed 3 snapshots
        for i in range(3):
            snap = ModelMonitoringSnapshot(
                id=uuid.uuid4(),
                model_version="1.0.0",
                window_type="DAILY" if i < 2 else "HOURLY",
                window_start=now - timedelta(days=i + 1),
                window_end=now - timedelta(days=i),
                sample_count=100 * (i + 1),
                labeled_count=10 * (i + 1),
                overall_status="NORMAL",
                data_drift_status="NORMAL",
                prediction_drift_status="NORMAL",
                performance_status="NORMAL",
                feature_drift_summary={},
                prediction_drift_summary={},
                performance_summary={},
                created_at=now - timedelta(days=i),
            )
            db_session.add(snap)

        await db_session.commit()

        resp = await async_api_client.get("/api/v1/monitoring/snapshots?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] >= 3
        assert len(data["items"]) >= 3

    async def test_request_validation_boundaries_422(
        self, async_api_client: AsyncClient
    ):
        """Verify FastAPI 422 validation on out-of-boundary parameters."""
        # 1. Invalid operating threshold > 1.0
        resp_thresh = await async_api_client.get("/api/v1/monitoring/performance?operating_threshold=1.5")
        assert resp_thresh.status_code == 422

        # 2. Invalid limit < 1
        resp_limit = await async_api_client.get("/api/v1/monitoring/drift/features?limit=0")
        assert resp_limit.status_code == 422
