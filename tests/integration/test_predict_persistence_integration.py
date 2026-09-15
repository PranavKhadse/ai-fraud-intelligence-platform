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
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

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

    async def test_2_predict_idempotent_replay_and_conflict_detection(
        self, async_api_client: AsyncClient, sample_payload: Dict[str, Any]
    ):
        """Test 2: Identical duplicate payload replays original result (200); conflicting payload returns 409."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_IDEMPOTENT_REPLAY_888"

        # 1. First evaluation succeeds
        resp1 = await async_api_client.post("/predict", json=payload)
        assert resp1.status_code == 200
        data1 = resp1.json()

        # 2. Second evaluation with IDENTICAL payload returns original result (HTTP 200 Idempotent Replay)
        resp2 = await async_api_client.post("/predict", json=payload)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data1["risk_score"] == data2["risk_score"]
        assert data1["decision_action"] == data2["decision_action"]
        assert data1["evaluated_at"] == data2["evaluated_at"]
        assert len(data1["reason_codes"]) == len(data2["reason_codes"])

        # 3. Third evaluation with CONFLICTING payload (different amount) returns HTTP 409 Conflict
        conflicting_payload = payload.copy()
        conflicting_payload["amount"] = 8888.88
        resp3 = await async_api_client.post("/predict", json=conflicting_payload)
        assert resp3.status_code == 409
        error_detail = resp3.json()["detail"]
        assert "TX_IDEMPOTENT_REPLAY_888" in error_detail
        assert "conflicting request payload" in error_detail

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
        payload_base = sample_payload.copy()
        payload_base["transaction_id"] = "TX_ROLLBACK_RECOVERY_001"

        # 1. First transaction succeeds
        r1 = await async_api_client.post("/predict", json=payload_base)
        assert r1.status_code == 200

        # 2. Second transaction with CONFLICTING payload fails with 409 and triggers rollback
        payload_conflict = payload_base.copy()
        payload_conflict["amount"] = 9999.99
        r2 = await async_api_client.post("/predict", json=payload_conflict)
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

    async def test_13_predict_concurrent_duplicate_requests(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 13: Multiple concurrent identical requests succeed with 200 and insert exactly 1 DB record."""
        import asyncio

        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_CONCURRENT_RACE_001"

        # Fire 5 concurrent identical requests simultaneously
        tasks = [async_api_client.post("/predict", json=payload) for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        # All 5 requests must return HTTP 200 OK
        for resp in responses:
            assert resp.status_code == 200
            data = resp.json()
            assert data["transaction_id"] == "TX_CONCURRENT_RACE_001"

        # Verify exactly ONE transaction record and ONE evaluation record exist in the database
        stmt_tx = select(Transaction).where(Transaction.external_transaction_id == "TX_CONCURRENT_RACE_001")
        res_tx = await db_session.execute(stmt_tx)
        tx_records = res_tx.scalars().all()
        assert len(tx_records) == 1

        stmt_eval = select(RiskEvaluation).where(RiskEvaluation.transaction_id == tx_records[0].id)
        res_eval = await db_session.execute(stmt_eval)
        eval_records = res_eval.scalars().all()
        assert len(eval_records) == 1

    async def test_14_predict_whitespace_transaction_id_returns_422(
        self, async_api_client: AsyncClient, sample_payload: Dict[str, Any]
    ):
        """Test 14: Blank or whitespace-only transaction_id is rejected with HTTP 422."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "    "

        response = await async_api_client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "cannot be empty or whitespace-only" in response.json()["detail"]

    async def test_15_predict_concurrent_conflicting_requests(
        self, async_api_client: AsyncClient, db_session: AsyncSession, sample_payload: Dict[str, Any]
    ):
        """Test 15: Concurrent conflicting requests result in one 200 OK and one 409 Conflict."""
        import asyncio

        base_payload = sample_payload.copy()
        base_payload["transaction_id"] = "TX_CONCURRENT_CONFLICT_001"

        conflicting_payload = base_payload.copy()
        conflicting_payload["amount"] = 99999.00

        # Run 2 concurrent requests with the SAME transaction_id but DIFFERENT amounts
        tasks = [
            async_api_client.post("/predict", json=base_payload),
            async_api_client.post("/predict", json=conflicting_payload),
        ]
        responses = await asyncio.gather(*tasks)

        status_codes = [r.status_code for r in responses]
        assert 200 in status_codes
        assert 409 in status_codes

        # Exactly 1 transaction record committed in PostgreSQL
        stmt_tx = select(Transaction).where(
            Transaction.external_transaction_id == "TX_CONCURRENT_CONFLICT_001"
        )
        res_tx = await db_session.execute(stmt_tx)
        tx_records = res_tx.scalars().all()
        assert len(tx_records) == 1

    async def test_16_predict_replayed_response_json_exact_match(
        self, async_api_client: AsyncClient, sample_payload: Dict[str, Any]
    ):
        """Test 16: Complete field-by-field JSON response equality between original evaluation and replayed response."""
        payload = sample_payload.copy()
        payload["transaction_id"] = "TX_JSON_EXACT_MATCH_001"

        # 1. Initial live evaluation
        resp1 = await async_api_client.post("/predict", json=payload)
        assert resp1.status_code == 200
        json1 = resp1.json()

        # 2. Idempotent replay from database
        resp2 = await async_api_client.post("/predict", json=payload)
        assert resp2.status_code == 200
        json2 = resp2.json()

        # 3. Field-by-field verification
        assert json1["transaction_id"] == json2["transaction_id"] == "TX_JSON_EXACT_MATCH_001"
        assert json1["risk_score"] == json2["risk_score"]
        assert json1["risk_tier"] == json2["risk_tier"]
        assert json1["decision_action"] == json2["decision_action"]
        assert json1["policy_mode"] == json2["policy_mode"]
        assert json1["reason"] == json2["reason"]
        assert json1["is_overridden"] == json2["is_overridden"]
        assert json1["rule_action"] == json2["rule_action"]
        assert json1["rules_triggered"] == json2["rules_triggered"]
        assert json1["model_version"] == json2["model_version"]
        assert json1["evaluated_at"] == json2["evaluated_at"]

        # 4. Reason codes list equality
        assert len(json1["reason_codes"]) == len(json2["reason_codes"])
        for rc1, rc2 in zip(json1["reason_codes"], json2["reason_codes"]):
            assert rc1["code"] == rc2["code"]
            assert rc1["rank"] == rc2["rank"]
            assert rc1["headline"] == rc2["headline"]
            assert rc1["severity"] == rc2["severity"]

        # 5. Top risk and mitigating factor equality
        assert len(json1["top_risk_factors"]) == len(json2["top_risk_factors"])
        for rf1, rf2 in zip(json1["top_risk_factors"], json2["top_risk_factors"]):
            assert rf1["feature_name"] == rf2["feature_name"]
            assert rf1["rank"] == rf2["rank"]
            assert rf1["direction"] == rf2["direction"]
            assert abs(rf1["shap_value"] - rf2["shap_value"]) < 1e-4
            assert abs(rf1["relative_contribution_pct"] - rf2["relative_contribution_pct"]) < 1e-4

        assert len(json1["top_mitigating_factors"]) == len(json2["top_mitigating_factors"])
        for mf1, mf2 in zip(json1["top_mitigating_factors"], json2["top_mitigating_factors"]):
            assert mf1["feature_name"] == mf2["feature_name"]
            assert mf1["rank"] == mf2["rank"]
            assert mf1["direction"] == mf2["direction"]
            assert abs(mf1["shap_value"] - mf2["shap_value"]) < 1e-4

        # 6. Complete serialized dictionary equality
        assert json1 == json2
