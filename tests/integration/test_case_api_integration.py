"""
Integration Tests for Case Management REST API & Lifecycle State Machine.

Validates complete end-to-end human review workflow against PostgreSQL:
1. Automated REVIEW prediction generates OPEN Case in DB.
2. GET /api/v1/cases reflects the case in review queue.
3. GET /api/v1/cases/summary reflects queue KPI counts.
4. Analyst claims case (CLAIM) -> IN_REVIEW.
5. Analyst appends investigation note -> CaseNote persisted, audit log staged.
6. Analyst escalates case (ESCALATE) -> ESCALATED.
7. Admin reassigns case (ASSIGN) -> assigned to senior analyst.
8. Admin submits disposition (CONFIRMED_FRAUD) -> RESOLVED.
9. Admin archives case (CLOSE) -> CLOSED.
10. GET /api/v1/cases/{id}/timeline returns authoritative AuditEntityType.CASE timeline.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncGenerator
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.core.config import settings
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
)
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.transaction import Transaction
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.services.persistence_service import (
    AuditLogData,
    FraudPersistenceService,
    PersistRiskEvaluationCommand,
    RiskEvaluationData,
    TransactionData,
)
from backend.app.services.unit_of_work import FraudPersistenceUnitOfWork

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="function")
async def async_case_client(
    db_session: AsyncSession,
    pg_engine: AsyncEngine,
    monkeypatch,
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an asynchronous httpx client with request-scoped database sessions."""
    monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", True)

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


async def test_full_case_management_lifecycle_integration(
    async_case_client: AsyncClient,
    db_session: AsyncSession,
):
    """
    Execute full lifecycle E2E:
    Auto REVIEW -> Claim -> Note -> Escalate -> Reassign -> Disposition -> Close -> Timeline.
    """
    # 1. Seed Transaction and RiskEvaluation with REVIEW action using FraudPersistenceService
    tx_data = TransactionData(
        account_id="ACC_INTEG_001",
        merchant_category="jewelry",
        job_category="doctor",
        amount=Decimal("1250.00"),
        currency="USD",
        cardholder_lat=Decimal("40.712800"),
        cardholder_long=Decimal("-74.006000"),
        merchant_lat=Decimal("40.722800"),
        merchant_long=Decimal("-74.016000"),
        city_pop=8000000,
        transaction_timestamp=datetime.now(timezone.utc),
        features_snapshot={"amount": 1250.0, "txn_count_1h": 5},
    )

    eval_data = RiskEvaluationData(
        model_version="1.0.0",
        policy_mode=PolicyMode.TRI_TIER,
        model_score=Decimal("0.650000"),
        risk_score=65,
        risk_tier=RiskTier.HIGH,
        decision_action=DecisionAction.REVIEW,
        baseline_action=DecisionAction.REVIEW,
        is_overridden=False,
        decision_reason="Evaluation scored in review threshold.",
        output_margin=Decimal("0.620000"),
        base_value=Decimal("-2.150000"),
        evaluated_at=datetime.now(timezone.utc),
    )

    cmd = PersistRiskEvaluationCommand(
        transaction=tx_data,
        evaluation=eval_data,
        audit=AuditLogData(
            event_type="PREDICTION_EVALUATED",
            action="EVALUATE",
            actor_type=AuditActorType.SYSTEM,
            actor_id="RISK_ENGINE",
            payload={"risk_score": 65},
        ),
    )

    uow = FraudPersistenceUnitOfWork(db_session)
    persistence = FraudPersistenceService(uow)
    record = await persistence.persist_evaluation(cmd)
    assert record.case_id is not None
    case_id = record.case_id

    # Common Auth Headers
    analyst_headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    admin_headers = {"X-Actor-ID": "admin_supervisor", "X-Actor-Role": "ADMIN"}

    # 2. Query Queue: GET /api/v1/cases
    queue_resp = await async_case_client.get("/api/v1/cases?status=OPEN", headers=analyst_headers)
    assert queue_resp.status_code == 200
    queue_json = queue_resp.json()
    assert queue_json["total"] >= 1
    found_case = next((c for c in queue_json["items"] if c["id"] == str(case_id)), None)
    assert found_case is not None
    assert found_case["account_id"] == "ACC_INTEG_001"
    assert found_case["risk_score"] == 65

    # 3. Query KPIs: GET /api/v1/cases/summary
    summary_resp = await async_case_client.get("/api/v1/cases/summary", headers=analyst_headers)
    assert summary_resp.status_code == 200
    summary_json = summary_resp.json()
    assert summary_json["total_open"] >= 1
    assert summary_json["unassigned_count"] >= 1

    # 4. Analyst Claims Case: PATCH /api/v1/cases/{case_id}/assignment
    claim_resp = await async_case_client.patch(
        f"/api/v1/cases/{case_id}/assignment",
        json={"action": "CLAIM"},
        headers=analyst_headers,
    )
    assert claim_resp.status_code == 200
    claim_json = claim_resp.json()
    assert claim_json["status"] == "IN_REVIEW"
    assert claim_json["assigned_to"] == "analyst_1"

    # 5. Analyst Appends Note: POST /api/v1/cases/{case_id}/notes
    note_resp = await async_case_client.post(
        f"/api/v1/cases/{case_id}/notes",
        json={
            "content": "Attempted cardholder contact via registered phone. No answer, left voicemail.",
            "note_type": "INVESTIGATION",
        },
        headers=analyst_headers,
    )
    assert note_resp.status_code == 201
    note_json = note_resp.json()
    assert note_json["author_id"] == "analyst_1"

    # 6. Analyst Escalates Case: PATCH /api/v1/cases/{case_id}/status
    escalate_resp = await async_case_client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={
            "target_status": "ESCALATED",
            "reason": "Merchant flagged matching velocity pattern on sibling cards.",
        },
        headers=analyst_headers,
    )
    assert escalate_resp.status_code == 200
    assert escalate_resp.json()["status"] == "ESCALATED"

    # 7. Admin Reassigns Case: PATCH /api/v1/cases/{case_id}/assignment
    assign_resp = await async_case_client.patch(
        f"/api/v1/cases/{case_id}/assignment",
        json={"action": "ASSIGN", "assignee_id": "senior_analyst_2"},
        headers=admin_headers,
    )
    assert assign_resp.status_code == 200
    assert assign_resp.json()["assigned_to"] == "senior_analyst_2"

    # 8. Admin Submits Disposition: POST /api/v1/cases/{case_id}/disposition
    disp_resp = await async_case_client.post(
        f"/api/v1/cases/{case_id}/disposition",
        json={
            "disposition": "CONFIRMED_FRAUD",
            "reason": "Cardholder confirmed fraudulent transaction and reported card stolen.",
        },
        headers=admin_headers,
    )
    assert disp_resp.status_code == 200
    disp_json = disp_resp.json()
    assert disp_json["status"] == "RESOLVED"
    assert disp_json["disposition"] == "CONFIRMED_FRAUD"
    assert disp_json["dispositioned_by"] == "admin_supervisor"

    # 9. Admin Archives Case: PATCH /api/v1/cases/{case_id}/status
    close_resp = await async_case_client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={
            "target_status": "CLOSED",
            "reason": "Investigation completed, chargeback initiated, and case archived.",
        },
        headers=admin_headers,
    )
    assert close_resp.status_code == 200
    close_json = close_resp.json()
    assert close_json["status"] == "CLOSED"
    assert close_json["closed_at"] is not None

    # 10. Verify Case Timeline: GET /api/v1/cases/{case_id}/timeline
    timeline_resp = await async_case_client.get(
        f"/api/v1/cases/{case_id}/timeline",
        headers=analyst_headers,
    )
    assert timeline_resp.status_code == 200
    timeline_json = timeline_resp.json()
    assert timeline_json["case_id"] == str(case_id)
    assert timeline_json["total_events"] >= 6

    actions = [e["action"] for e in timeline_json["events"]]
    assert "CREATE_CASE" in actions
    assert "CLAIM_CASE" in actions
    assert "ADD_NOTE" in actions
    assert "ESCALATE_CASE" in actions
    assert "ASSIGN_CASE" in actions
    assert "RECORD_DISPOSITION" in actions
    assert "CLOSE_CASE" in actions
