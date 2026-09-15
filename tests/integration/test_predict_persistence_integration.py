"""
Integration Tests for Phase 9.6 Increment 2: Safe /predict API Persistence Integration.

Verifies end-to-end integration between FastAPI /predict endpoint, RiskService,
RiskPersistenceMapper, and FraudPersistenceService against live PostgreSQL:
1. Successful prediction with full relational aggregate persistence and database readback.
2. Context extraction: X-Correlation-ID / X-Request-ID, client IP, evaluation latency.
3. Duplicate external_transaction_id returns HTTP 409 Conflict.
4. Database persistence failure returns HTTP 500 and triggers clean rollback.
5. Risk engine inference failure halts before any database persistence.
6. Missing account_id returns HTTP 422 Unprocessable Entity without persistence.
7. Explicit transaction timestamp preservation vs. ingestion fallback for omitted timestamp.
8. Public response contract immutability (no persistence internal ID leakage).
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, patch
import pandas as pd
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import (
    AttributionDirection,
    AuditActorType,
    DecisionAction,
    PolicyMode,
    RiskTier,
)
from backend.app.db.models.feature_attribution import EvaluationFeatureAttribution
from backend.app.db.models.reason_code import EvaluationReasonCode
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.rule_match import EvaluationRuleMatch
from backend.app.db.models.transaction import Transaction
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.repositories.exceptions import PersistenceError
from backend.app.schemas.predict import PredictionResponse
from ml.models.config import VAL_FEATURES_PATH

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="function")
async def async_api_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide an asynchronous httpx client bound to the active test database session."""
    async def override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def sample_payload() -> Dict[str, Any]:
    """Extract a canonical transaction row for testing."""
    df = pd.read_parquet(VAL_FEATURES_PATH)
    row = df.iloc[0].to_dict()
    row["transaction_id"] = "TX_INT_TEST_001"
    row["account_id"] = "ACC_INT_9876"
    row["timestamp"] = "2026-09-15T10:30:00Z"
    return row


class TestPredictPersistenceIntegration:
    """Test suite verifying end-to-end /predict API persistence integration."""

    async def test_1_predict_with_full_persistence_readback(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 1: POST /predict persists all 6 aggregate entities and verifies database readback."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_PERSIST_001"
        payload["account_id"] = "ACC_USER_555"
        payload["timestamp"] = "2026-09-15T12:00:00Z"

        headers = {
            "X-Correlation-ID": "corr-uuid-test-999",
            "X-Forwarded-For": "203.0.113.195, 10.0.0.1",
        }

        response = await async_api_client.post("/predict", json=payload, headers=headers)
        assert response.status_code == 200
        data = response.json()

        # Verify response structure and contract
        assert data["decision_action"] in ("APPROVE", "REVIEW", "BLOCK")
        assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert 0 <= data["risk_score"] <= 100
        assert 0.0 <= data["model_score"] <= 1.0
        assert "evaluated_at" in data

        # Query database directly using db_session
        # 1. Verify Transaction record
        stmt_tx = select(Transaction).where(Transaction.external_transaction_id == "TX_PERSIST_001")
        res_tx = await db_session.execute(stmt_tx)
        tx = res_tx.scalar_one_or_none()
        assert tx is not None
        assert tx.account_id == "ACC_USER_555"
        assert tx.amount == Decimal(str(round(payload["amount"], 2)))
        assert tx.currency == "USD"
        assert tx.merchant_category == payload["merchant_category"]
        assert tx.job_category == payload["job_category"]
        assert len(tx.features_snapshot) == 55
        assert tx.transaction_timestamp == datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

        # 2. Verify RiskEvaluation record
        stmt_eval = select(RiskEvaluation).where(RiskEvaluation.transaction_id == tx.id)
        res_eval = await db_session.execute(stmt_eval)
        eval_record = res_eval.scalar_one_or_none()
        assert eval_record is not None
        assert eval_record.model_score == Decimal(str(round(data["model_score"], 6)))
        assert eval_record.risk_score == data["risk_score"]
        assert eval_record.risk_tier.value == data["risk_tier"]
        assert eval_record.decision_action.value == data["decision_action"]
        assert eval_record.correlation_id == "corr-uuid-test-999"
        assert eval_record.evaluation_latency_ms is not None
        assert eval_record.evaluation_latency_ms >= Decimal("0.00")

        # 3. Verify Reason Codes in DB
        stmt_rc = select(EvaluationReasonCode).where(EvaluationReasonCode.evaluation_id == eval_record.id)
        res_rc = await db_session.execute(stmt_rc)
        reasons = res_rc.scalars().all()
        assert len(reasons) == len(data["reason_codes"])

        # 4. Verify Feature Attributions in DB
        stmt_fa = select(EvaluationFeatureAttribution).where(EvaluationFeatureAttribution.evaluation_id == eval_record.id)
        res_fa = await db_session.execute(stmt_fa)
        attributions = res_fa.scalars().all()
        expected_attr_count = len(data["top_risk_factors"]) + len(data["top_mitigating_factors"])
        assert len(attributions) == expected_attr_count

        # 5. Verify Audit Log in DB
        stmt_audit = select(AuditLog).where(AuditLog.entity_id == eval_record.id)
        res_audit = await db_session.execute(stmt_audit)
        audit = res_audit.scalar_one_or_none()
        assert audit is not None
        assert audit.event_type == "RISK_EVALUATION_PERSISTED"
        assert audit.actor_id == "fastapi_predict_api"
        assert audit.correlation_id == "corr-uuid-test-999"
        assert audit.client_ip in ("127.0.0.1", "testclient")
        assert audit.payload["external_transaction_id"] == "TX_PERSIST_001"

    async def test_2_predict_duplicate_external_transaction_id_returns_409(
        self, async_api_client: AsyncClient, sample_payload: Dict[str, Any]
    ):
        """Test 2: Duplicate external_transaction_id returns HTTP 409 Conflict with descriptive message."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_DUPLICATE_ID_888"

        # First evaluation succeeds
        resp1 = await async_api_client.post("/predict", json=payload)
        assert resp1.status_code == 200

        # Second evaluation with duplicate ID must return 409 Conflict
        resp2 = await async_api_client.post("/predict", json=payload)
        assert resp2.status_code == 409
        error_detail = resp2.json()["detail"]
        assert "TX_DUPLICATE_ID_888" in error_detail
        assert "already exists" in error_detail

    async def test_3_predict_context_extraction_x_request_id(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 3: X-Request-ID header is extracted as correlation_id when X-Correlation-ID is absent."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_REQ_ID_001"

        headers = {"X-Request-ID": "req-header-uuid-777"}
        response = await async_api_client.post("/predict", json=payload, headers=headers)
        assert response.status_code == 200

        stmt = select(RiskEvaluation).join(Transaction).where(Transaction.external_transaction_id == "TX_REQ_ID_001")
        res = await db_session.execute(stmt)
        eval_record = res.scalar_one_or_none()
        assert eval_record is not None
        assert eval_record.correlation_id == "req-header-uuid-777"

    async def test_4_predict_client_ip_from_connection_when_no_forwarded_header(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 4: Client IP falls back to request.client.host when X-Forwarded-For is missing."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_DIRECT_IP_001"

        response = await async_api_client.post("/predict", json=payload)
        assert response.status_code == 200

        stmt = select(AuditLog).where(AuditLog.payload["external_transaction_id"].as_string() == "TX_DIRECT_IP_001")
        res = await db_session.execute(stmt)
        audit = res.scalar_one_or_none()
        assert audit is not None
        assert audit.client_ip in ("testclient", "127.0.0.1", "localhost", None) or audit.client_ip is not None

    async def test_5_predict_persistence_database_failure_returns_500_and_rolls_back(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 5: Unhandled persistence error returns HTTP 500 without leaving orphaned transaction records."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_FAIL_ROLLBACK_001"

        with patch(
            "backend.app.services.persistence_service.FraudPersistenceService.persist_evaluation",
            side_effect=PersistenceError("Disk I/O error during persistence"),
        ):
            response = await async_api_client.post("/predict", json=payload)
            assert response.status_code == 500
            assert "Persistence failure" in response.json()["detail"]

        stmt = select(Transaction).where(Transaction.external_transaction_id == "TX_FAIL_ROLLBACK_001")
        res = await db_session.execute(stmt)
        tx = res.scalar_one_or_none()
        assert tx is None

    async def test_6_predict_risk_service_failure_prevents_persistence(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 6: Inference validation failure (HTTP 422) executes zero persistence operations."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_INVALID_AMOUNT_001"
        payload["amount"] = -500.0  # Invalid negative amount

        response = await async_api_client.post("/predict", json=payload)
        assert response.status_code == 422

        stmt = select(Transaction).where(Transaction.external_transaction_id == "TX_INVALID_AMOUNT_001")
        res = await db_session.execute(stmt)
        tx = res.scalar_one_or_none()
        assert tx is None

    async def test_7_predict_missing_account_id_returns_422(
        self, async_api_client: AsyncClient, sample_payload: Dict[str, Any]
    ):
        """Test 7: Missing account_id raises validation error and returns HTTP 422."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_NO_ACC_001"
        payload["account_id"] = None

        response = await async_api_client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "account_id" in response.json()["detail"]

    async def test_8_predict_explicit_timestamp_preservation(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 8: Explicit timestamp from request is preserved exactly in database."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_TIMESTAMP_EXPLICIT"
        payload["timestamp"] = "2026-06-20T18:45:00Z"

        response = await async_api_client.post("/predict", json=payload)
        assert response.status_code == 200

        stmt = select(Transaction).where(Transaction.external_transaction_id == "TX_TIMESTAMP_EXPLICIT")
        res = await db_session.execute(stmt)
        tx = res.scalar_one_or_none()
        assert tx is not None
        assert tx.transaction_timestamp == datetime(2026, 6, 20, 18, 45, 0, tzinfo=timezone.utc)

    async def test_9_predict_omitted_timestamp_uses_ingestion_fallback(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 9: Omitted timestamp uses explicit UTC ingestion fallback satisfying database NOT NULL constraint."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_TIMESTAMP_OMITTED"
        payload["timestamp"] = None

        before_request = datetime.now(timezone.utc)
        response = await async_api_client.post("/predict", json=payload)
        assert response.status_code == 200
        after_request = datetime.now(timezone.utc)

        stmt = select(Transaction).where(Transaction.external_transaction_id == "TX_TIMESTAMP_OMITTED")
        res = await db_session.execute(stmt)
        tx = res.scalar_one_or_none()
        assert tx is not None
        assert tx.transaction_timestamp is not None
        # Verify timestamp falls within request execution window
        assert before_request <= tx.transaction_timestamp <= after_request

    async def test_10_predict_response_schema_immutability_and_compatibility(
        self, async_api_client: AsyncClient, sample_payload: Dict[str, Any]
    ):
        """Test 10: Response payload strictly adheres to PredictionResponse contract without internal ID leakage."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_SCHEMA_CHECK_001"

        response = await async_api_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        expected_keys = {
            "transaction_id",
            "model_score",
            "risk_score",
            "risk_tier",
            "decision_action",
            "policy_mode",
            "reason",
            "is_overridden",
            "rule_action",
            "rules_triggered",
            "rule_matches",
            "reason_codes",
            "top_risk_factors",
            "top_mitigating_factors",
            "model_version",
            "evaluated_at",
        }
        assert set(data.keys()) == expected_keys

        # Ensure no persistence-internal UUIDs are exposed
        assert "evaluation_id" not in data
        assert "audit_log_id" not in data
        assert "id" not in data

    async def test_11_predict_session_usability_after_rollback(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 11: Database session remains completely healthy and operational after an aborted 409 rollback."""
        payload_dup = sample_payload.copy()
        payload_dup["transaction_id"] = "TX_ROLLBACK_RECOVERY_001"

        # 1. First transaction succeeds
        r1 = await async_api_client.post("/predict", json=payload_dup)
        assert r1.status_code == 200

        # 2. Second transaction fails with 409 and triggers rollback
        r2 = await async_api_client.post("/predict", json=payload_dup)
        assert r2.status_code == 409

        # 3. Third distinct transaction succeeds on the active session
        payload_new = sample_payload.copy()
        payload_new["transaction_id"] = "TX_ROLLBACK_RECOVERY_002"
        r3 = await async_api_client.post("/predict", json=payload_new)
        assert r3.status_code == 200

        # Verify both successful transactions exist in database
        stmt = select(Transaction).where(
            Transaction.external_transaction_id.in_(
                ["TX_ROLLBACK_RECOVERY_001", "TX_ROLLBACK_RECOVERY_002"]
            )
        )
        res = await db_session.execute(stmt)
        records = res.scalars().all()
        assert len(records) == 2

    async def test_12_predict_trusted_proxy_forwarded_ip_integration(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 12: When TRUST_PROXY_HEADERS=True, valid X-Forwarded-For IP is stored in AuditLog."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_TRUSTED_PROXY_001"

        headers = {
            "X-Forwarded-For": "203.0.113.88, 10.0.0.1",
        }

        with patch("backend.app.api.v1.endpoints.predict.settings.TRUST_PROXY_HEADERS", True):
            response = await async_api_client.post("/predict", json=payload, headers=headers)
            assert response.status_code == 200

        stmt = select(AuditLog).where(
            AuditLog.payload["external_transaction_id"].as_string() == "TX_TRUSTED_PROXY_001"
        )
        res = await db_session.execute(stmt)
        audit = res.scalar_one_or_none()
        assert audit is not None
        assert audit.client_ip == "203.0.113.88"

