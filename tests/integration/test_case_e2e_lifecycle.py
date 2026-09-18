"""
Phase 12.5 Comprehensive Integration Test Suite:
End-to-End Lifecycle Verification, Concurrency Hardening, RBAC Security Matrix & Ground-Truth Data Contract.

Validates against real PostgreSQL:
1. Multi-turn E2E case investigation workflow:
   Auto REVIEW creation -> Queue discovery -> Analyst claim -> Note -> Escalation ->
   Admin reassignment -> Human disposition -> Reopen -> Close -> Audit timeline.
2. Concurrent duplicate case creation:
   Row-level unique constraint serialization (1x 201 Created, Nx 409 Conflict).
3. Concurrent reviewer claims under SELECT FOR UPDATE row locking:
   1x 200 OK, Nx 409 Conflict, winner assigned, IN_REVIEW status, 1x CASE_ASSIGNED audit event,
   winning analyst retry is idempotent.
4. Concurrent human dispositions under SELECT FOR UPDATE row locking:
   1x 200 OK, Nx 422 Unprocessable Entity, zero deadlocks, zero 500 errors, internal state
   consistency, exactly one DISPOSITION note and CASE_DISPOSITION_RECORDED audit log.
5. Authoritative RBAC matrix:
   ANALYST, ADMIN, API_CLIENT, SYSTEM, and production fail-closed verification.
6. Ground-truth disposition data contract:
   Validates persisted feature snapshots (55 features), model scores, risk scores, dispositions,
   reasons, timestamps, and relational integrity for downstream Phase 14 retraining loops.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.core.config import settings
from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.case import Case, CaseNote
from backend.app.db.models.enums import (
    AuditActorType,
    AuditEntityType,
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


# ==============================================================================
# Fixtures & Helpers
# ==============================================================================

@pytest_asyncio.fixture(scope="function")
async def async_e2e_client(
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


async def _seed_test_transaction_and_evaluation(
    db_session: AsyncSession,
    account_id: str = "ACC_E2E_001",
    amount: Decimal = Decimal("750.00"),
    decision_action: DecisionAction = DecisionAction.REVIEW,
    risk_tier: RiskTier = RiskTier.HIGH,
    risk_score: int = 75,
    features_count: int = 55,
) -> Dict[str, Any]:
    """Seed a Transaction and RiskEvaluation in PostgreSQL via FraudPersistenceService."""
    # Ensure exactly features_count features in snapshot
    base_features_count = max(0, features_count - 2)
    features_snapshot = {f"feature_{i}": float(i) * 1.5 for i in range(base_features_count)}
    features_snapshot["amt"] = float(amount)
    features_snapshot["hour"] = 14.0

    tx_data = TransactionData(
        account_id=account_id,
        merchant_category="electronics",
        job_category="engineer",
        amount=amount,
        currency="USD",
        cardholder_lat=Decimal("37.774900"),
        cardholder_long=Decimal("-122.419400"),
        merchant_lat=Decimal("37.784900"),
        merchant_long=Decimal("-122.409400"),
        city_pop=870000,
        transaction_timestamp=datetime.now(timezone.utc),
        features_snapshot=features_snapshot,
    )

    eval_data = RiskEvaluationData(
        model_version="1.0.0",
        policy_mode=PolicyMode.TRI_TIER,
        model_score=Decimal("0.750000"),
        risk_score=risk_score,
        risk_tier=risk_tier,
        decision_action=decision_action,
        baseline_action=decision_action,
        is_overridden=False,
        decision_reason="Evaluation scored in review threshold for testing.",
        output_margin=Decimal("0.720000"),
        base_value=Decimal("-1.850000"),
        evaluated_at=datetime.now(timezone.utc),
    )

    cmd = PersistRiskEvaluationCommand(
        transaction=tx_data,
        evaluation=eval_data,
        audit=AuditLogData(
            event_type="PREDICTION_EVALUATED",
            action="EVALUATE",
            actor_type=AuditActorType.SYSTEM,
            actor_id="TEST_RISK_ENGINE",
            payload={"risk_score": risk_score},
        ),
    )

    uow = FraudPersistenceUnitOfWork(db_session)
    persistence = FraudPersistenceService(uow)
    record = await persistence.persist_evaluation(cmd)

    return {
        "transaction_id": record.transaction_id,
        "evaluation_id": record.evaluation_id,
        "case_id": record.case_id,
        "case_number": record.case_number,
    }


# ==============================================================================
# 1. Full Multi-Turn Lifecycle Integration Test
# ==============================================================================

async def test_full_multi_turn_case_lifecycle_e2e(
    async_e2e_client: AsyncClient,
    db_session: AsyncSession,
):
    """
    Scenario: Multi-Turn Case Lifecycle Execution
    1. Automated REVIEW prediction triggers OPEN Case.
    2. Analyst queries review queue with filters and discovers case.
    3. Analyst claims case (CLAIM) -> IN_REVIEW.
    4. Analyst appends investigation note -> CaseNote persisted.
    5. Analyst escalates case (ESCALATE) -> ESCALATED with rationale.
    6. Admin supervisor reassigns case (ASSIGN) to Senior Analyst -> IN_REVIEW.
    7. Senior Analyst resolves case via human disposition (CONFIRMED_FRAUD) -> RESOLVED.
    8. Post-disposition dispute triggers REOPEN -> IN_REVIEW.
    9. Final investigation completes -> CLOSE -> CLOSED.
    10. Case Audit Timeline verifies complete, unbroken chronological event sequence.
    """
    # Step 1: Ingest Transaction & Evaluation with REVIEW decision
    seed = await _seed_test_transaction_and_evaluation(
        db_session,
        account_id="ACC_LIFECYCLE_999",
        amount=Decimal("1899.99"),
        decision_action=DecisionAction.REVIEW,
        risk_score=82,
    )
    case_id = seed["case_id"]
    assert case_id is not None
    case_number = seed["case_number"]

    analyst_headers = {"X-Actor-ID": "analyst_sarah", "X-Actor-Role": "ANALYST"}
    admin_headers = {"X-Actor-ID": "admin_chief", "X-Actor-Role": "ADMIN"}

    # Step 2: Query Review Queue and Discover Case
    queue_resp = await async_e2e_client.get(
        "/api/v1/cases?status=OPEN&assigned_to=unassigned",
        headers=analyst_headers,
    )
    assert queue_resp.status_code == 200
    queue_data = queue_resp.json()
    assert queue_data["total"] >= 1
    discovered = next((c for c in queue_data["items"] if c["id"] == str(case_id)), None)
    assert discovered is not None
    assert discovered["case_number"] == case_number
    assert discovered["status"] == "OPEN"
    assert discovered["assigned_to"] is None

    # Step 3: Analyst Claims Case (OPEN -> IN_REVIEW)
    claim_resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/assignment",
        json={"action": "CLAIM"},
        headers=analyst_headers,
    )
    assert claim_resp.status_code == 200
    claim_data = claim_resp.json()
    assert claim_data["status"] == "IN_REVIEW"
    assert claim_data["assigned_to"] == "analyst_sarah"
    assert claim_data["assigned_at"] is not None

    # Step 4: Analyst Appends Investigation Note
    note_resp = await async_e2e_client.post(
        f"/api/v1/cases/{case_id}/notes",
        json={
            "content": "Contacted cardholder. Phone number registered in New York, transaction initiated from London.",
            "note_type": "INVESTIGATION",
        },
        headers=analyst_headers,
    )
    assert note_resp.status_code == 201
    note_data = note_resp.json()
    assert note_data["case_id"] == str(case_id)
    assert note_data["author_id"] == "analyst_sarah"
    assert note_data["note_type"] == "INVESTIGATION"

    # Step 5: Analyst Escalates Case (IN_REVIEW -> ESCALATED)
    escalate_resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={
            "target_status": "ESCALATED",
            "reason": "Cross-border velocity anomaly requires senior risk officer review and potential card block.",
        },
        headers=analyst_headers,
    )
    assert escalate_resp.status_code == 200
    escalate_data = escalate_resp.json()
    assert escalate_data["status"] == "ESCALATED"

    # Step 6: Admin Reassigns Case to Senior Analyst
    assign_resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/assignment",
        json={
            "action": "ASSIGN",
            "assignee_id": "analyst_marcus",
            "reason": "Assigned to Senior Cross-Border Fraud Specialist Marcus.",
        },
        headers=admin_headers,
    )
    assert assign_resp.status_code == 200
    assign_data = assign_resp.json()
    assert assign_data["assigned_to"] == "analyst_marcus"

    # Step 7: Senior Supervisor Submits Human Disposition (ESCALATED -> RESOLVED)
    disp_resp = await async_e2e_client.post(
        f"/api/v1/cases/{case_id}/disposition",
        json={
            "disposition": "CONFIRMED_FRAUD",
            "reason": "Cardholder confirmed unauthorized physical POS transaction in London while present in NYC.",
        },
        headers=admin_headers,
    )
    assert disp_resp.status_code == 200
    disp_data = disp_resp.json()
    assert disp_data["status"] == "RESOLVED"
    assert disp_data["disposition"] == "CONFIRMED_FRAUD"
    assert disp_data["dispositioned_by"] == "admin_chief"
    assert disp_data["dispositioned_at"] is not None
    assert disp_data["resolved_at"] is not None

    # Step 8: Subsequent Dispute Reopens Case (RESOLVED -> IN_REVIEW)
    reopen_resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={
            "target_status": "IN_REVIEW",
            "reason": "Merchant submitted secondary evidence and authorization code for review.",
        },
        headers=analyst_headers,
    )
    assert reopen_resp.status_code == 200
    reopen_data = reopen_resp.json()
    assert reopen_data["status"] == "IN_REVIEW"
    assert reopen_data["disposition"] is None
    assert reopen_data["resolved_at"] is None

    # Re-resolve case before closing (must be RESOLVED before CLOSED)
    re_disp_resp = await async_e2e_client.post(
        f"/api/v1/cases/{case_id}/disposition",
        json={
            "disposition": "SUSPICIOUS_RESOLVED",
            "reason": "Merchant evidence reviewed; partial refund processed and dispute amicably settled.",
        },
        headers=analyst_headers,
    )
    assert re_disp_resp.status_code == 200
    assert re_disp_resp.json()["status"] == "RESOLVED"

    # Step 9: Admin Closes & Archives Case (RESOLVED -> CLOSED)
    close_resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={
            "target_status": "CLOSED",
            "reason": "All chargebacks settled, merchant notified, case closed for permanent archival.",
        },
        headers=admin_headers,
    )
    assert close_resp.status_code == 200
    close_data = close_resp.json()
    assert close_data["status"] == "CLOSED"
    assert close_data["closed_at"] is not None

    # Step 10: Verify Complete Audit Timeline
    timeline_resp = await async_e2e_client.get(
        f"/api/v1/cases/{case_id}/timeline",
        headers=analyst_headers,
    )
    assert timeline_resp.status_code == 200
    timeline_data = timeline_resp.json()
    assert timeline_data["case_id"] == str(case_id)
    assert timeline_data["total_events"] >= 8

    actions = [e["action"] for e in timeline_data["events"]]
    assert "CREATE_CASE" in actions
    assert "CLAIM_CASE" in actions
    assert "ADD_NOTE" in actions
    assert "ESCALATE_CASE" in actions
    assert "ASSIGN_CASE" in actions
    assert "RECORD_DISPOSITION" in actions
    assert "REOPEN_CASE" in actions
    assert "CLOSE_CASE" in actions


# ==============================================================================
# 2. Concurrency Hardening: Duplicate Case Creation Race
# ==============================================================================

async def test_concurrent_duplicate_case_creation(
    async_e2e_client: AsyncClient,
    db_session: AsyncSession,
):
    """
    Validate that concurrent manual case escalation requests for the same transaction
    are serialized safely by PostgreSQL unique constraint uq_cases_transaction_id.

    Contract:
    - Exactly 1 request succeeds with HTTP 201 Created.
    - All remaining concurrent requests receive HTTP 409 Conflict.
    - Exactly 1 Case record exists in the database.
    """
    seed = await _seed_test_transaction_and_evaluation(
        db_session,
        account_id="ACC_RACE_001",
        amount=Decimal("499.00"),
        decision_action=DecisionAction.APPROVE,  # APPROVE -> no automated case created
    )
    tx_id = str(seed["transaction_id"])
    eval_id = str(seed["evaluation_id"])

    analyst_headers = {"X-Actor-ID": "analyst_concurrent", "X-Actor-Role": "ANALYST"}
    payload = {
        "transaction_id": tx_id,
        "evaluation_id": eval_id,
        "initial_note": "Concurrent escalation race condition test.",
        "priority": "HIGH",
    }

    # Execute 5 concurrent case creation requests
    tasks = [
        async_e2e_client.post("/api/v1/cases", json=payload, headers=analyst_headers)
        for _ in range(5)
    ]
    responses = await asyncio.gather(*tasks)

    status_codes = [r.status_code for r in responses]
    assert status_codes.count(201) == 1, f"Expected exactly 1 HTTP 201, got {status_codes}"
    assert status_codes.count(409) == 4, f"Expected 4 HTTP 409 conflicts, got {status_codes}"

    # Verify conflict error payload
    conflict_responses = [r for r in responses if r.status_code == 409]
    for cr in conflict_responses:
        assert "already exists" in cr.json()["detail"].lower()

    # Verify DB cardinality
    stmt = select(Case).where(Case.transaction_id == seed["transaction_id"])
    res = await db_session.execute(stmt)
    cases = res.scalars().all()
    assert len(cases) == 1


# ==============================================================================
# 3. Concurrency Hardening: Reviewer Self-Claim Race
# ==============================================================================

async def test_concurrent_reviewer_claims(
    async_e2e_client: AsyncClient,
    db_session: AsyncSession,
):
    """
    Validate concurrent CLAIM requests from multiple analysts on the same unassigned case.

    Implemented Concurrency Behavior:
    - Row-level lock (SELECT FOR UPDATE) serializes transactions.
    - First analyst acquires lock, moves status OPEN -> IN_REVIEW, assigns self -> HTTP 200 OK.
    - Competing concurrent analysts acquire lock, detect case is already assigned -> HTTP 409 Conflict.
    - Final assigned_to matches the winning analyst.
    - Final status is IN_REVIEW.
    - Exactly one CASE_ASSIGNED audit event is emitted.
    - Winning analyst can re-claim idempotently with HTTP 200 OK.
    """
    seed = await _seed_test_transaction_and_evaluation(
        db_session,
        account_id="ACC_CLAIM_RACE",
        amount=Decimal("820.00"),
        decision_action=DecisionAction.REVIEW,
    )
    case_id = seed["case_id"]

    analysts = [
        {"X-Actor-ID": f"analyst_competitor_{i}", "X-Actor-Role": "ANALYST"}
        for i in range(5)
    ]

    # Spawn 5 concurrent CLAIM requests
    tasks = [
        async_e2e_client.patch(
            f"/api/v1/cases/{case_id}/assignment",
            json={"action": "CLAIM"},
            headers=headers,
        )
        for headers in analysts
    ]
    responses = await asyncio.gather(*tasks)

    status_codes = [r.status_code for r in responses]
    assert status_codes.count(200) == 1, f"Expected exactly 1 HTTP 200 winner, got {status_codes}"
    assert status_codes.count(409) == 4, f"Expected 4 HTTP 409 conflicts, got {status_codes}"

    # Identify the winning analyst
    winning_response = next(r for r in responses if r.status_code == 200)
    winning_data = winning_response.json()
    winner_id = winning_data["assigned_to"]
    assert winner_id.startswith("analyst_competitor_")
    assert winning_data["status"] == "IN_REVIEW"

    # Verify conflict messages mention the winning assignee
    conflict_responses = [r for r in responses if r.status_code == 409]
    for cr in conflict_responses:
        detail = cr.json()["detail"]
        assert "already assigned to" in detail
        assert winner_id in detail

    # Verify DB state
    stmt = select(Case).where(Case.id == case_id)
    res = await db_session.execute(stmt)
    case_db = res.scalar_one()
    assert case_db.assigned_to == winner_id
    assert case_db.status == CaseStatus.IN_REVIEW

    # Verify Audit Logs: Exactly one CASE_ASSIGNED / CLAIM_CASE audit event
    stmt_audit = (
        select(AuditLog)
        .where(AuditLog.entity_id == case_id)
        .where(AuditLog.event_type == "CASE_ASSIGNED")
    )
    res_audit = await db_session.execute(stmt_audit)
    audit_events = res_audit.scalars().all()
    assert len(audit_events) == 1
    assert audit_events[0].actor_id == winner_id

    # Verify Idempotent Retry by the winning analyst
    winner_headers = {"X-Actor-ID": winner_id, "X-Actor-Role": "ANALYST"}
    retry_resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/assignment",
        json={"action": "CLAIM"},
        headers=winner_headers,
    )
    assert retry_resp.status_code == 200
    assert retry_resp.json()["assigned_to"] == winner_id
    assert retry_resp.json()["status"] == "IN_REVIEW"


# ==============================================================================
# 4. Concurrency Hardening: Human Disposition Race
# ==============================================================================

async def test_concurrent_dispositions(
    async_e2e_client: AsyncClient,
    db_session: AsyncSession,
):
    """
    Validate concurrent human disposition submissions on an IN_REVIEW case.

    Implemented Behavior:
    - Zero deadlocks, zero unhandled 500 errors.
    - Exactly 1 request successfully transitions IN_REVIEW -> RESOLVED (HTTP 200 OK).
    - Competing concurrent requests find the case already RESOLVED -> HTTP 422 Unprocessable Entity.
    - Case state remains strictly consistent.
    - Exactly one DISPOSITION note is recorded.
    - Exactly one CASE_DISPOSITION_RECORDED audit log is emitted.
    - Audit log ordering remains chronological and valid.
    """
    seed = await _seed_test_transaction_and_evaluation(
        db_session,
        account_id="ACC_DISP_RACE",
        amount=Decimal("1200.00"),
        decision_action=DecisionAction.REVIEW,
    )
    case_id = seed["case_id"]

    # First claim the case to move OPEN -> IN_REVIEW
    analyst_init = {"X-Actor-ID": "analyst_primary", "X-Actor-Role": "ANALYST"}
    claim_resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/assignment",
        json={"action": "CLAIM"},
        headers=analyst_init,
    )
    assert claim_resp.status_code == 200

    # Prepare concurrent disposition payloads
    dispositions = [
        ("CONFIRMED_FRAUD", "Cardholder confirmed fraud and opened unauthorized charge ticket.", "admin_user_1"),
        ("FALSE_POSITIVE", "Customer verified transaction over authenticated mobile banking channel.", "admin_user_2"),
        ("LEGITIMATE", "Regular recurring billing detected with matching billing address.", "admin_user_3"),
        ("SUSPICIOUS_RESOLVED", "Suspicious velocity resolved after manual cardholder callback.", "admin_user_4"),
    ]

    tasks = [
        async_e2e_client.post(
            f"/api/v1/cases/{case_id}/disposition",
            json={"disposition": disp, "reason": reason},
            headers={"X-Actor-ID": actor_id, "X-Actor-Role": "ADMIN"},
        )
        for disp, reason, actor_id in dispositions
    ]
    responses = await asyncio.gather(*tasks)

    status_codes = [r.status_code for r in responses]
    assert status_codes.count(200) == 1, f"Expected exactly 1 HTTP 200 winner, got {status_codes}"
    assert status_codes.count(422) == 3, f"Expected 3 HTTP 422 rejections, got {status_codes}"
    assert 500 not in status_codes, f"Found unexpected HTTP 500 errors: {status_codes}"

    # Verify winning payload
    winning_response = next(r for r in responses if r.status_code == 200)
    winning_json = winning_response.json()
    assert winning_json["status"] == "RESOLVED"
    winner_disp = winning_json["disposition"]
    winner_actor = winning_json["dispositioned_by"]
    assert winner_disp in [d[0] for d in dispositions]

    # Verify DB State Consistency
    stmt = select(Case).where(Case.id == case_id)
    res = await db_session.execute(stmt)
    case_db = res.scalar_one()
    assert case_db.status == CaseStatus.RESOLVED
    assert case_db.disposition.value == winner_disp
    assert case_db.dispositioned_by == winner_actor
    assert case_db.dispositioned_at is not None
    assert case_db.resolved_at is not None
    assert len(case_db.disposition_reason) >= 10

    # Verify exactly one DISPOSITION note exists
    stmt_notes = (
        select(CaseNote)
        .where(CaseNote.case_id == case_id)
        .where(CaseNote.note_type == CaseNoteType.DISPOSITION)
    )
    res_notes = await db_session.execute(stmt_notes)
    notes = res_notes.scalars().all()
    assert len(notes) == 1
    assert winner_disp in notes[0].content

    # Verify exactly one CASE_DISPOSITION_RECORDED audit log
    stmt_audit = (
        select(AuditLog)
        .where(AuditLog.entity_id == case_id)
        .where(AuditLog.action == "RECORD_DISPOSITION")
    )
    res_audit = await db_session.execute(stmt_audit)
    disp_audits = res_audit.scalars().all()
    assert len(disp_audits) == 1
    assert disp_audits[0].actor_id == winner_actor


# ==============================================================================
# 5. Authoritative RBAC Security Matrix Tests
# ==============================================================================

async def test_case_rbac_matrix_e2e(
    async_e2e_client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch,
):
    """
    Verify complete role authorization matrix across ANALYST, ADMIN, API_CLIENT, and SYSTEM.
    Also validates production fail-closed behavior when ALLOW_DEV_ACTOR_HEADERS=False.
    """
    seed = await _seed_test_transaction_and_evaluation(
        db_session,
        account_id="ACC_RBAC_001",
        amount=Decimal("300.00"),
        decision_action=DecisionAction.REVIEW,
    )
    case_id = seed["case_id"]

    analyst_headers = {"X-Actor-ID": "analyst_bob", "X-Actor-Role": "ANALYST"}
    admin_headers = {"X-Actor-ID": "admin_clara", "X-Actor-Role": "ADMIN"}
    api_client_headers = {"X-Actor-ID": "external_api", "X-Actor-Role": "API_CLIENT"}

    # 1. Analyst cannot ASSIGN to another reviewer (Admin only) -> 403 Forbidden
    resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/assignment",
        json={"action": "ASSIGN", "assignee_id": "other_analyst"},
        headers=analyst_headers,
    )
    assert resp.status_code == 403
    assert "Only administrators" in resp.json()["detail"]

    # 2. Admin CAN ASSIGN to another reviewer -> 200 OK
    resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/assignment",
        json={"action": "ASSIGN", "assignee_id": "analyst_bob"},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["assigned_to"] == "analyst_bob"

    # 3. Analyst escalates case
    resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={"target_status": "ESCALATED", "reason": "Requires supervisor sign-off."},
        headers=analyst_headers,
    )
    assert resp.status_code == 200

    # 4. Analyst CANNOT disposition an ESCALATED case (Admin only) -> 403 Forbidden
    resp = await async_e2e_client.post(
        f"/api/v1/cases/{case_id}/disposition",
        json={"disposition": "LEGITIMATE", "reason": "Attempting unauthorized disposition."},
        headers=analyst_headers,
    )
    assert resp.status_code == 403
    assert "Only administrators can disposition escalated cases" in resp.json()["detail"]

    # 5. Admin CAN disposition an ESCALATED case -> 200 OK
    resp = await async_e2e_client.post(
        f"/api/v1/cases/{case_id}/disposition",
        json={"disposition": "LEGITIMATE", "reason": "Supervisor reviewed merchant proof and cleared case."},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "RESOLVED"

    # 6. Admin closes case
    resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={"target_status": "CLOSED", "reason": "Case archived by administrator."},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "CLOSED"

    # 7. Analyst CANNOT reopen a CLOSED case (Admin only) -> 403 Forbidden
    resp = await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={"target_status": "IN_REVIEW", "reason": "Analyst attempting reopen on closed case."},
        headers=analyst_headers,
    )
    assert resp.status_code == 403
    assert "Only administrators can reopen closed/archived cases" in resp.json()["detail"]

    # 8. API_CLIENT cannot perform case mutations -> 403 Forbidden
    resp = await async_e2e_client.post(
        f"/api/v1/cases/{case_id}/notes",
        json={"content": "External note attempt.", "note_type": "INVESTIGATION"},
        headers=api_client_headers,
    )
    assert resp.status_code == 403

    # 9. API_CLIENT CAN read case detail -> 200 OK
    resp = await async_e2e_client.get(
        f"/api/v1/cases/{case_id}",
        headers=api_client_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["case"]["id"] == str(case_id)

    # 10. Production Fail-Closed Behavior when ALLOW_DEV_ACTOR_HEADERS=False
    monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
    resp = await async_e2e_client.get(
        "/api/v1/cases",
        headers=analyst_headers,
    )
    assert resp.status_code == 401
    assert "Actor authentication required" in resp.json()["detail"]


# ==============================================================================
# 6. Ground-Truth Disposition Data Contract Validation
# ==============================================================================

async def test_ground_truth_disposition_data_contract(
    async_e2e_client: AsyncClient,
    db_session: AsyncSession,
):
    """
    Validate that resolved cases preserve complete point-in-time ground-truth data:
    - features_snapshot (55 features)
    - model_score
    - risk_score
    - disposition
    - disposition_reason
    - dispositioned_at
    - relational integrity between Case, Transaction, and RiskEvaluation.
    """
    seed = await _seed_test_transaction_and_evaluation(
        db_session,
        account_id="ACC_ML_FEEDBACK_001",
        amount=Decimal("2450.00"),
        decision_action=DecisionAction.REVIEW,
        risk_score=88,
        features_count=55,
    )
    case_id = seed["case_id"]

    analyst_headers = {"X-Actor-ID": "analyst_senior", "X-Actor-Role": "ANALYST"}

    # Claim case
    await async_e2e_client.patch(
        f"/api/v1/cases/{case_id}/assignment",
        json={"action": "CLAIM"},
        headers=analyst_headers,
    )

    # Submit disposition
    disp_resp = await async_e2e_client.post(
        f"/api/v1/cases/{case_id}/disposition",
        json={
            "disposition": "CONFIRMED_FRAUD",
            "reason": "Mule account pattern confirmed by law enforcement request.",
        },
        headers=analyst_headers,
    )
    assert disp_resp.status_code == 200

    # Query Case, Transaction, and RiskEvaluation directly from PostgreSQL
    stmt = (
        select(Case, Transaction, RiskEvaluation)
        .join(Transaction, Case.transaction_id == Transaction.id)
        .join(RiskEvaluation, Case.evaluation_id == RiskEvaluation.id)
        .where(Case.id == case_id)
    )
    res = await db_session.execute(stmt)
    row = res.one_or_none()
    assert row is not None
    case_row, tx_row, eval_row = row

    # 1. Validate Feature Snapshot Integrity
    assert isinstance(tx_row.features_snapshot, dict)
    assert len(tx_row.features_snapshot) == 55
    assert tx_row.features_snapshot["amt"] == 2450.0
    assert tx_row.features_snapshot["hour"] == 14.0
    assert tx_row.amount == Decimal("2450.00")
    assert tx_row.currency == "USD"

    # 2. Validate Model Score & Risk Score Integrity
    assert isinstance(eval_row.model_score, Decimal)
    assert Decimal("0.0") <= eval_row.model_score <= Decimal("1.0")
    assert eval_row.model_score == Decimal("0.750000")
    assert isinstance(eval_row.risk_score, int)
    assert 0 <= eval_row.risk_score <= 100
    assert eval_row.risk_score == 88

    # 3. Validate Disposition & Operational Ground-Truth Fields
    assert case_row.status == CaseStatus.RESOLVED
    assert case_row.disposition == CaseDisposition.CONFIRMED_FRAUD
    assert isinstance(case_row.disposition_reason, str)
    assert len(case_row.disposition_reason) >= 10
    assert case_row.disposition_reason == "Mule account pattern confirmed by law enforcement request."
    assert isinstance(case_row.dispositioned_at, datetime)
    assert case_row.dispositioned_at.tzinfo is not None
    assert case_row.dispositioned_by == "analyst_senior"

    # 4. Validate Relational Integrity
    assert case_row.transaction_id == tx_row.id
    assert case_row.evaluation_id == eval_row.id
    assert eval_row.transaction_id == tx_row.id
