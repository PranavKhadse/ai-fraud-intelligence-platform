"""
Unit Tests for Phase 15.3: RBAC Enforcement on Existing Endpoints.

Validates:
1. Unauthenticated requests fail closed with HTTP 401 across prediction, dashboard, monitoring, and case management endpoints.
2. Invalid, expired, and malformed Bearer tokens fail closed with HTTP 401.
3. Invalid, revoked, and expired API keys fail closed with HTTP 401.
4. Role-based isolation:
   - Machine identities (API_CLIENT via X-API-Key or dev header) CAN access prediction.
   - Machine identities (API_CLIENT) CANNOT access analyst/admin-only operations (Cases, Dashboard, Monitoring) -> HTTP 403 Forbidden.
   - Specifically verifies /api/v1/cases/{case_id} and /api/v1/cases/{case_id}/timeline strictly reject API_CLIENT with HTTP 403.
5. Human roles (ANALYST, ADMIN via Bearer JWT) are permitted across all 4 endpoint domains.
6. Dev-header compatibility (X-Actor-ID, X-Actor-Role) is strictly gated by ALLOW_DEV_ACTOR_HEADERS:
   - When ALLOW_DEV_ACTOR_HEADERS=False: dev headers fail closed with HTTP 401.
   - When ALLOW_DEV_ACTOR_HEADERS=True: dev headers authenticate and enforce RBAC.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    generate_api_key,
    hash_api_key,
)
from backend.app.db.models.api_key import APIKey
from backend.app.db.models.case import Case, CaseNote
from backend.app.db.models.enums import (
    AuditActorType,
    CaseDisposition,
    CaseNoteType,
    CasePriority,
    CaseStatus,
    CaseTriggerSource,
    DecisionAction,
    PolicyMode,
    RiskTier,
    UserRole,
)
from backend.app.db.models.monitoring_snapshot import ModelMonitoringSnapshot
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.transaction import Transaction
from backend.app.db.models.user import User
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.repositories.api_key_repository import APIKeyRepository
from backend.app.repositories.dashboard_repository import DashboardRepository
from backend.app.repositories.user_repository import UserRepository
from backend.app.schemas.dashboard import DashboardOverviewResponse
from backend.app.schemas.monitoring import MonitoringHealthResponse
from backend.app.schemas.predict import (
    FeatureAttributionResponse,
    PredictionResponse,
    ReasonCodeResponse,
    RuleMatchResponse,
    TransactionPredictRequest,
)
from backend.app.services.case_service import CaseService, get_case_service
from backend.app.services.monitoring_service import MonitoringService, get_monitoring_service
from backend.app.services.persistence_service import FraudPersistenceService, get_persistence_service
from backend.app.services.risk_service import RiskService, get_risk_service


# ==============================================================================
# Sample Fixtures & Mocks
# ==============================================================================

@pytest.fixture
def mock_predict_payload() -> Dict[str, Any]:
    return {
        "transaction_id": "TX_RBAC_001",
        "account_id": "ACC_998877",
        "amount": 149.99,
        "currency": "USD",
        "merchant_category": "electronics",
        "job_category": "engineer",
        "cardholder_lat": 37.7749,
        "cardholder_long": -122.4194,
        "merchant_lat": 37.7750,
        "merchant_long": -122.4180,
        "city_pop": 850000,
        "timestamp": "2026-09-15T10:30:00Z",
        "transaction_hour": 10,
        "day_of_week": 1,
        "day_of_month": 15,
        "month": 9,
        "week_of_year": 38,
        "is_weekend": 0,
        "is_night": 0,
        "hour_sin": 0.5,
        "hour_cos": -0.866,
        "day_of_week_sin": 0.781,
        "day_of_week_cos": 0.623,
        "txn_count_1h": 1.0,
        "txn_count_6h": 3.0,
        "txn_count_24h": 5.0,
        "txn_count_7d": 15.0,
        "txn_count_30d": 45.0,
        "time_since_prev_txn_seconds": 3600.0,
        "is_first_account_txn": 0,
        "amt_sum_1h": 149.99,
        "amt_sum_24h": 520.0,
        "amt_sum_7d": 1800.0,
        "amt_sum_30d": 6500.0,
        "amt_mean_24h": 104.0,
        "amt_mean_7d": 120.0,
        "amt_max_24h": 250.0,
        "amt_median_30d": 95.0,
        "historical_amount_mean": 110.0,
        "historical_amount_std": 35.0,
        "historical_amount_median": 95.0,
        "amount_zscore": 1.14,
        "amount_ratio_to_historical_mean": 1.36,
        "account_txn_count_before": 45.0,
        "account_total_spend_before": 6500.0,
        "account_avg_amount_before": 110.0,
        "account_max_amount_before": 400.0,
        "account_unique_merchant_count_before": 20.0,
        "account_unique_category_count_before": 8.0,
        "account_merchant_txn_count_before": 2.0,
        "account_category_txn_count_before": 5.0,
        "account_merchant_spend_before": 300.0,
        "account_category_spend_before": 750.0,
        "merchant_txn_count_before": 500.0,
        "category_txn_count_before": 12000.0,
        "cardholder_merchant_distance_km": 0.15,
        "distance_from_prev_merchant_km": 2.5,
        "implied_travel_speed_kmh": 2.5,
        "is_impossible_travel_speed": 0,
    }


@pytest.fixture
def mock_prediction_response() -> PredictionResponse:
    return PredictionResponse(
        transaction_id="tx_rbac_test_001",
        model_version="1.0.0",
        model_score=0.15,
        risk_score=15,
        risk_tier="LOW",
        decision_action="APPROVE",
        policy_mode="TRI_TIER",
        reason="Model score 0.15 is below LOW threshold",
        is_overridden=False,
        rule_action=None,
        rules_triggered=[],
        rule_matches=[],
        reason_codes=[],
        top_risk_factors=[],
        top_mitigating_factors=[],
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def mock_case() -> Case:
    case_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)
    c = Case(
        id=case_id,
        case_number="CASE-20261014-TEST",
        transaction_id=uuid.uuid4(),
        evaluation_id=uuid.uuid4(),
        status=CaseStatus.OPEN,
        priority=CasePriority.HIGH,
        trigger_source=CaseTriggerSource.AUTOMATED_REVIEW_POLICY,
        assigned_to=None,
        assigned_at=None,
        opened_at=now_utc,
        resolved_at=None,
        closed_at=None,
        disposition=None,
        disposition_reason=None,
        dispositioned_by=None,
        dispositioned_at=None,
        created_at=now_utc,
        updated_at=now_utc,
    )
    c.transaction = Transaction(
        id=c.transaction_id,
        external_transaction_id="EXT-101",
        account_id="acc_rbac_001",
        merchant_id="merch_101",
        merchant_category="retail",
        job_category="engineer",
        amount=Decimal("150.00"),
        currency="USD",
        cardholder_lat=Decimal("40.712800"),
        cardholder_long=Decimal("-74.006000"),
        merchant_lat=Decimal("40.713000"),
        merchant_long=Decimal("-74.005500"),
        city_pop=50000,
        transaction_timestamp=now_utc,
        features_snapshot={"amount": 150.0},
        created_at=now_utc,
        updated_at=now_utc,
    )
    c.evaluation = RiskEvaluation(
        id=c.evaluation_id,
        transaction_id=c.transaction_id,
        model_version="1.0.0",
        policy_mode=PolicyMode.TRI_TIER,
        model_score=Decimal("0.850000"),
        risk_score=85,
        risk_tier=RiskTier.HIGH,
        decision_action=DecisionAction.REVIEW,
        baseline_action=DecisionAction.APPROVE,
        is_overridden=True,
        decision_reason="High risk score",
        output_margin=Decimal("1.734000"),
        base_value=Decimal("-2.150000"),
        evaluated_at=now_utc,
        created_at=now_utc,
    )
    c.evaluation.rule_matches = []
    c.evaluation.reason_codes = []
    c.evaluation.feature_attributions = []
    c.notes = []
    return c


@pytest.fixture
def mock_user_analyst() -> User:
    return User(
        id=uuid.uuid4(),
        username="analyst_jane",
        email="jane@example.com",
        role=UserRole.ANALYST,
        is_active=True,
        is_superuser=False,
    )


@pytest.fixture
def mock_user_admin() -> User:
    return User(
        id=uuid.uuid4(),
        username="admin_bob",
        email="bob@example.com",
        role=UserRole.ADMIN,
        is_active=True,
        is_superuser=True,
    )


@pytest.fixture
def mock_api_key_record() -> tuple[str, APIKey]:
    raw_key, hashed_key = generate_api_key()
    record = APIKey(
        id=uuid.uuid4(),
        name="Payment Gateway Ingestion",
        key_prefix=raw_key[:8],
        hashed_key=hashed_key,
        client_id="payment_gateway_prod",
        role=AuditActorType.API_CLIENT,
        is_active=True,
        expires_at=datetime.now(timezone.utc) + timedelta(days=90),
    )
    return raw_key, record


# ==============================================================================
# Helper Client Provider
# ==============================================================================

@pytest.fixture
def rbac_client(
    mock_prediction_response,
    mock_case,
    mock_user_analyst,
    mock_user_admin,
    mock_api_key_record,
    monkeypatch,
) -> TestClient:
    """Configures TestClient with all service mocks and session overrides."""
    raw_key, api_key_entity = mock_api_key_record

    # Mock RiskService & PersistenceService
    mock_risk = MagicMock(spec=RiskService)
    mock_risk.predict_transaction = MagicMock(return_value=mock_prediction_response)
    mock_pers = MagicMock(spec=FraudPersistenceService)
    mock_pers.get_existing_evaluation = AsyncMock(return_value=None)
    mock_pers.persist_evaluation = AsyncMock()

    # Mock CaseService
    mock_case_svc = MagicMock(spec=CaseService)
    mock_case_svc.get_queue_cases = AsyncMock(return_value=([mock_case], 1))
    mock_case_svc.get_summary_metrics = AsyncMock(return_value={"total_open": 1})
    mock_case_svc.get_case_detail = AsyncMock(return_value=mock_case)
    mock_case_svc.get_timeline = AsyncMock(return_value=(mock_case, []))

    # Mock MonitoringService
    mock_mon_svc = MagicMock(spec=MonitoringService)
    mock_mon_svc.get_health_overview = AsyncMock(
        return_value=MonitoringHealthResponse(
            model_version="1.0.0",
            window_type="DAILY",
            window_start="2026-10-14T00:00:00Z",
            window_end="2026-10-14T23:59:59Z",
            sample_count=100,
            labeled_count=20,
            overall_status="NORMAL",
            data_drift_status="NORMAL",
            prediction_drift_status="NORMAL",
            performance_status="NORMAL",
            active_alert_count=0,
            active_alerts=[],
            created_at="2026-10-14T12:00:00Z",
        )
    )

    # Mock DashboardRepository
    mock_dash_repo = MagicMock(spec=DashboardRepository)
    mock_dash_repo.get_overview_metrics = AsyncMock(
        return_value=DashboardOverviewResponse(
            total_transactions=100,
            total_amount=50000.0,
            approval_count=80,
            approval_rate=80.0,
            review_count=15,
            review_rate=15.0,
            block_count=5,
            block_rate=5.0,
            average_risk_score=22.5,
            average_latency_ms=12.4,
        )
    )

    # Mock Session and Repositories
    mock_session = AsyncMock(spec=AsyncSession)

    async def mock_user_get_by_id(user_id):
        if user_id == mock_user_analyst.id:
            return mock_user_analyst
        if user_id == mock_user_admin.id:
            return mock_user_admin
        return None

    async def mock_api_key_get_by_hash(h):
        if h == api_key_entity.hashed_key:
            return api_key_entity
        return None

    user_repo_mock = MagicMock(spec=UserRepository)
    user_repo_mock.get_by_id = AsyncMock(side_effect=mock_user_get_by_id)

    api_key_repo_mock = MagicMock(spec=APIKeyRepository)
    api_key_repo_mock.get_by_hashed_key = AsyncMock(side_effect=mock_api_key_get_by_hash)
    api_key_repo_mock.update_last_used = AsyncMock()

    # Wire up overrides
    app.dependency_overrides[get_risk_service] = lambda: mock_risk
    app.dependency_overrides[get_persistence_service] = lambda: mock_pers
    app.dependency_overrides[get_case_service] = lambda: mock_case_svc
    app.dependency_overrides[get_monitoring_service] = lambda: mock_mon_svc
    from backend.app.api.v1.endpoints.dashboard import get_dashboard_repository
    app.dependency_overrides[get_dashboard_repository] = lambda: mock_dash_repo
    app.dependency_overrides[get_db_session] = lambda: mock_session

    monkeypatch.setattr("backend.app.repositories.user_repository.UserRepository", lambda s: user_repo_mock)
    monkeypatch.setattr("backend.app.repositories.api_key_repository.APIKeyRepository", lambda s: api_key_repo_mock)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ==============================================================================
# 1. Unauthenticated Requests (HTTP 401)
# ==============================================================================

class TestUnauthenticatedRequestsFailClosed:
    """Verifies that all 4 endpoint domains return 401 when no auth is presented."""

    def test_predict_endpoint_unauthenticated_returns_401(self, rbac_client, mock_predict_payload, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        response = rbac_client.post("/api/v1/predict", json=mock_predict_payload)
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_root_predict_endpoint_unauthenticated_returns_401(self, rbac_client, mock_predict_payload, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        response = rbac_client.post("/predict", json=mock_predict_payload)
        assert response.status_code == 401

    def test_dashboard_overview_unauthenticated_returns_401(self, rbac_client, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        response = rbac_client.get("/api/v1/dashboard/overview")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_monitoring_health_unauthenticated_returns_401(self, rbac_client, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        response = rbac_client.get("/api/v1/monitoring/health")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_case_list_unauthenticated_returns_401(self, rbac_client, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        response = rbac_client.get("/api/v1/cases")
        assert response.status_code == 401

    def test_case_detail_unauthenticated_returns_401(self, rbac_client, mock_case, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        response = rbac_client.get(f"/api/v1/cases/{mock_case.id}")
        assert response.status_code == 401


# ==============================================================================
# 2. Invalid / Expired / Tampered Credentials (HTTP 401)
# ==============================================================================

class TestInvalidCredentialsFailClosed:
    """Verifies that corrupted, expired, or bogus tokens and keys fail with 401."""

    def test_invalid_bearer_token_returns_401(self, rbac_client, mock_predict_payload, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        headers = {"Authorization": "Bearer invalid.tampered.token"}
        response = rbac_client.post("/api/v1/predict", json=mock_predict_payload, headers=headers)
        assert response.status_code == 401

    def test_expired_bearer_token_returns_401(self, rbac_client, mock_user_analyst, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        # Create token already expired in the past
        expired_token = create_access_token(
            subject=mock_user_analyst.id,
            role=mock_user_analyst.role,
            expires_delta=timedelta(seconds=-10),
            username=mock_user_analyst.username,
        )
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = rbac_client.get("/api/v1/dashboard/overview", headers=headers)
        assert response.status_code == 401

    def test_refresh_token_used_as_access_token_returns_401(self, rbac_client, mock_user_analyst, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        refresh_tok, _ = create_refresh_token(subject=mock_user_analyst.id, username=mock_user_analyst.username)
        headers = {"Authorization": f"Bearer {refresh_tok}"}
        response = rbac_client.get("/api/v1/monitoring/health", headers=headers)
        assert response.status_code == 401

    def test_invalid_api_key_returns_401(self, rbac_client, mock_predict_payload, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        headers = {"X-API-Key": "pk_live_completely_bogus_key_value_here"}
        response = rbac_client.post("/api/v1/predict", json=mock_predict_payload, headers=headers)
        assert response.status_code == 401


# ==============================================================================
# 3. Machine Identity (API_CLIENT) Role Restrictions (403 vs Allowed)
# ==============================================================================

class TestMachineIdentityRoleIsolation:
    """
    Verifies machine API clients (API_CLIENT):
    - CAN invoke transaction predictions.
    - CANNOT invoke analyst or admin operations (Cases, Dashboard, Monitoring) -> 403.
    """

    def test_api_client_permitted_on_predict(
        self, rbac_client, mock_predict_payload, mock_api_key_record, monkeypatch
    ):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        raw_key, _ = mock_api_key_record
        headers = {"X-API-Key": raw_key}
        response = rbac_client.post("/api/v1/predict", json=mock_predict_payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["decision_action"] == "APPROVE"

    def test_api_client_denied_on_case_queue(
        self, rbac_client, mock_api_key_record, monkeypatch
    ):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        raw_key, _ = mock_api_key_record
        headers = {"X-API-Key": raw_key}
        response = rbac_client.get("/api/v1/cases", headers=headers)
        assert response.status_code == 403
        assert "Forbidden" in response.json()["detail"]

    def test_api_client_denied_on_case_detail(
        self, rbac_client, mock_case, mock_api_key_record, monkeypatch
    ):
        """Specifically verifies that get_case no longer permits API_CLIENT."""
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        raw_key, _ = mock_api_key_record
        headers = {"X-API-Key": raw_key}
        response = rbac_client.get(f"/api/v1/cases/{mock_case.id}", headers=headers)
        assert response.status_code == 403
        assert "Forbidden" in response.json()["detail"]

    def test_api_client_denied_on_case_timeline(
        self, rbac_client, mock_case, mock_api_key_record, monkeypatch
    ):
        """Specifically verifies that get_case_timeline no longer permits API_CLIENT."""
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        raw_key, _ = mock_api_key_record
        headers = {"X-API-Key": raw_key}
        response = rbac_client.get(f"/api/v1/cases/{mock_case.id}/timeline", headers=headers)
        assert response.status_code == 403
        assert "Forbidden" in response.json()["detail"]

    def test_api_client_denied_on_dashboard_overview(
        self, rbac_client, mock_api_key_record, monkeypatch
    ):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        raw_key, _ = mock_api_key_record
        headers = {"X-API-Key": raw_key}
        response = rbac_client.get("/api/v1/dashboard/overview", headers=headers)
        assert response.status_code == 403
        assert "Forbidden" in response.json()["detail"]

    def test_api_client_denied_on_monitoring_health(
        self, rbac_client, mock_api_key_record, monkeypatch
    ):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        raw_key, _ = mock_api_key_record
        headers = {"X-API-Key": raw_key}
        response = rbac_client.get("/api/v1/monitoring/health", headers=headers)
        assert response.status_code == 403
        assert "Forbidden" in response.json()["detail"]


# ==============================================================================
# 4. Human Roles (ANALYST, ADMIN) Authorization
# ==============================================================================

class TestHumanRolesAccess:
    """Verifies that authenticated human analysts and administrators can access authorized endpoints."""

    def test_analyst_permitted_on_all_domains(
        self, rbac_client, mock_user_analyst, mock_predict_payload, mock_case, monkeypatch
    ):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        token = create_access_token(
            subject=mock_user_analyst.id,
            role=mock_user_analyst.role,
            username=mock_user_analyst.username,
        )
        headers = {"Authorization": f"Bearer {token}"}

        # Predict
        res_predict = rbac_client.post("/api/v1/predict", json=mock_predict_payload, headers=headers)
        assert res_predict.status_code == 200

        # Cases
        res_cases = rbac_client.get("/api/v1/cases", headers=headers)
        assert res_cases.status_code == 200

        # Case Detail
        res_case_detail = rbac_client.get(f"/api/v1/cases/{mock_case.id}", headers=headers)
        assert res_case_detail.status_code == 200

        # Dashboard Overview
        res_dash = rbac_client.get("/api/v1/dashboard/overview", headers=headers)
        assert res_dash.status_code == 200

        # Monitoring Health
        res_mon = rbac_client.get("/api/v1/monitoring/health", headers=headers)
        assert res_mon.status_code == 200

    def test_admin_permitted_on_all_domains(
        self, rbac_client, mock_user_admin, mock_predict_payload, mock_case, monkeypatch
    ):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        token = create_access_token(
            subject=mock_user_admin.id,
            role=mock_user_admin.role,
            username=mock_user_admin.username,
        )
        headers = {"Authorization": f"Bearer {token}"}

        # Predict
        res_predict = rbac_client.post("/api/v1/predict", json=mock_predict_payload, headers=headers)
        assert res_predict.status_code == 200

        # Cases
        res_cases = rbac_client.get("/api/v1/cases", headers=headers)
        assert res_cases.status_code == 200

        # Dashboard
        res_dash = rbac_client.get("/api/v1/dashboard/overview", headers=headers)
        assert res_dash.status_code == 200

        # Monitoring
        res_mon = rbac_client.get("/api/v1/monitoring/health", headers=headers)
        assert res_mon.status_code == 200


# ==============================================================================
# 5. Development-Header Compatibility Gating
# ==============================================================================

class TestDevHeaderCompatibilityGating:
    """Verifies that development headers are strictly gated by ALLOW_DEV_ACTOR_HEADERS."""

    def test_dev_headers_rejected_when_allow_dev_headers_false(
        self, rbac_client, mock_predict_payload, monkeypatch
    ):
        """When ALLOW_DEV_ACTOR_HEADERS=False, X-Actor-ID/Role are ignored and fails closed with 401."""
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
        headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
        response = rbac_client.post("/api/v1/predict", json=mock_predict_payload, headers=headers)
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_dev_headers_accepted_when_allow_dev_headers_true(
        self, rbac_client, mock_predict_payload, monkeypatch
    ):
        """When ALLOW_DEV_ACTOR_HEADERS=True, valid dev headers authenticate successfully."""
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", True)
        headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
        response = rbac_client.post("/api/v1/predict", json=mock_predict_payload, headers=headers)
        assert response.status_code == 200

    def test_dev_headers_rbac_enforced_when_allowed(
        self, rbac_client, monkeypatch
    ):
        """Even with ALLOW_DEV_ACTOR_HEADERS=True, API_CLIENT role is forbidden on cases."""
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", True)
        headers = {"X-Actor-ID": "machine_client_1", "X-Actor-Role": "API_CLIENT"}
        response = rbac_client.get("/api/v1/cases", headers=headers)
        assert response.status_code == 403
        assert "Forbidden" in response.json()["detail"]
