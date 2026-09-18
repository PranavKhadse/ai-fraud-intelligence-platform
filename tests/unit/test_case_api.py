"""
Unit Tests for Phase 12.3: Case Management REST API Endpoints.

Validates:
1. GET /api/v1/cases - Paginated Review Queue
2. GET /api/v1/cases/summary - Queue Summary KPIs
3. GET /api/v1/cases/{case_id} - Investigation Workspace Detail
4. POST /api/v1/cases - Manual Case Escalation
5. PATCH /api/v1/cases/{case_id}/assignment - Reviewer Assignment Mutations
6. PATCH /api/v1/cases/{case_id}/status - Lifecycle Status Mutations
7. GET /api/v1/cases/{case_id}/notes - Chronological Notes List
8. POST /api/v1/cases/{case_id}/notes - Append Investigation Note
9. POST /api/v1/cases/{case_id}/disposition - Submit Human Disposition
10. GET /api/v1/cases/{case_id}/timeline - Audit Timeline History
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest
from fastapi.testclient import TestClient

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
from backend.app.main import app
from backend.app.repositories.exceptions import (
    PersistenceConflictError,
    PersistenceNotFoundError,
)
from backend.app.services.case_service import CaseService, get_case_service


# ==============================================================================
# Helpers & Mock Entities
# ==============================================================================

def _make_mock_transaction(tx_id: uuid.UUID) -> Transaction:
    now_utc = datetime.now(timezone.utc)
    return Transaction(
        id=tx_id,
        external_transaction_id="EXT-12345",
        account_id="ACC_TEST_001",
        merchant_id="MERCH_999",
        merchant_category="electronics",
        job_category="engineer",
        amount=Decimal("499.99"),
        currency="USD",
        cardholder_lat=Decimal("37.774900"),
        cardholder_long=Decimal("-122.419400"),
        merchant_lat=Decimal("37.784900"),
        merchant_long=Decimal("-122.409400"),
        city_pop=50000,
        transaction_timestamp=now_utc,
        features_snapshot={"amount": 499.99, "txn_count_1h": 3},
        created_at=now_utc,
        updated_at=now_utc,
    )


def _make_mock_evaluation(eval_id: uuid.UUID, tx_id: uuid.UUID) -> RiskEvaluation:
    now_utc = datetime.now(timezone.utc)
    ev = RiskEvaluation(
        id=eval_id,
        transaction_id=tx_id,
        model_version="1.0.0",
        policy_mode=PolicyMode.TRI_TIER,
        model_score=Decimal("0.850000"),
        risk_score=85,
        risk_tier=RiskTier.HIGH,
        decision_action=DecisionAction.REVIEW,
        baseline_action=DecisionAction.REVIEW,
        is_overridden=False,
        rule_action=None,
        decision_reason="High risk score requiring analyst investigation.",
        output_margin=Decimal("1.734000"),
        base_value=Decimal("-2.150000"),
        evaluated_at=now_utc,
        created_at=now_utc,
    )
    ev.rule_matches = []
    ev.reason_codes = []
    ev.feature_attributions = []
    return ev


def _make_mock_case_with_context() -> Case:
    case_id = uuid.uuid4()
    tx_id = uuid.uuid4()
    eval_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)

    tx = _make_mock_transaction(tx_id)
    ev = _make_mock_evaluation(eval_id, tx_id)

    case = Case(
        id=case_id,
        case_number=f"CASE-20260918-{case_id.hex[:6].upper()}",
        transaction_id=tx_id,
        evaluation_id=eval_id,
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
    case.transaction = tx
    case.evaluation = ev
    case.notes = [
        CaseNote(
            id=uuid.uuid4(),
            case_id=case_id,
            author_id="SYSTEM",
            author_role=AuditActorType.SYSTEM,
            note_type=CaseNoteType.SYSTEM_AUDIT,
            content="Automated case created by TRI_TIER review policy.",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    ]
    return case


@pytest.fixture
def mock_service():
    service = MagicMock(spec=CaseService)
    return service


@pytest.fixture
def client(mock_service, monkeypatch) -> TestClient:
    monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", True)
    app.dependency_overrides[get_case_service] = lambda: mock_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ==============================================================================
# API Endpoint Tests
# ==============================================================================

def test_list_cases_endpoint(client: TestClient, mock_service):
    """Verify GET /api/v1/cases returns paginated queue items."""
    case = _make_mock_case_with_context()
    mock_service.get_queue_cases = AsyncMock(return_value=([case], 1))

    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.get("/api/v1/cases?limit=10&offset=0&status=OPEN", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["case_number"] == case.case_number
    assert data["items"][0]["account_id"] == "ACC_TEST_001"
    assert data["items"][0]["risk_score"] == 85


def test_get_summary_endpoint(client: TestClient, mock_service):
    """Verify GET /api/v1/cases/summary returns aggregated KPIs."""
    mock_service.get_summary_metrics = AsyncMock(return_value={
        "total_open": 12,
        "unassigned_count": 5,
        "in_review_count": 4,
        "escalated_count": 3,
        "resolved_today": 8,
        "resolved_last_24h": 14,
        "critical_priority_count": 2,
    })

    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.get("/api/v1/cases/summary", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["total_open"] == 12
    assert data["unassigned_count"] == 5
    assert data["critical_priority_count"] == 2


def test_get_case_detail_endpoint(client: TestClient, mock_service):
    """Verify GET /api/v1/cases/{case_id} returns full investigation context."""
    case = _make_mock_case_with_context()
    mock_service.get_case_detail = AsyncMock(return_value=case)

    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.get(f"/api/v1/cases/{case.id}", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["case"]["id"] == str(case.id)
    assert data["transaction"]["account_id"] == "ACC_TEST_001"
    assert data["evaluation"]["risk_tier"] == "HIGH"
    assert len(data["notes"]) == 1


def test_get_case_detail_not_found(client: TestClient, mock_service):
    """Verify GET /api/v1/cases/{case_id} returns 404 for missing case."""
    missing_id = uuid.uuid4()
    mock_service.get_case_detail = AsyncMock(side_effect=PersistenceNotFoundError(f"Case '{missing_id}' not found."))

    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.get(f"/api/v1/cases/{missing_id}", headers=headers)

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_create_case_endpoint(client: TestClient, mock_service):
    """Verify POST /api/v1/cases creates manual escalation case."""
    case = _make_mock_case_with_context()
    mock_service.create_manual_case = AsyncMock(return_value=case)

    payload = {
        "transaction_id": str(case.transaction_id),
        "initial_note": "Manual escalation following customer call center alert.",
        "priority": "HIGH",
    }
    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.post("/api/v1/cases", json=payload, headers=headers)

    assert response.status_code == 201
    data = response.json()
    assert data["id"] == str(case.id)


def test_update_assignment_endpoint(client: TestClient, mock_service):
    """Verify PATCH /api/v1/cases/{case_id}/assignment mutates assignment."""
    case = _make_mock_case_with_context()
    case.status = CaseStatus.IN_REVIEW
    case.assigned_to = "analyst_1"
    mock_service.update_assignment = AsyncMock(return_value=case)

    payload = {"action": "CLAIM"}
    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.patch(f"/api/v1/cases/{case.id}/assignment", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["assigned_to"] == "analyst_1"
    assert data["status"] == "IN_REVIEW"


def test_update_status_endpoint(client: TestClient, mock_service):
    """Verify PATCH /api/v1/cases/{case_id}/status updates lifecycle status."""
    case = _make_mock_case_with_context()
    case.status = CaseStatus.ESCALATED
    mock_service.update_status = AsyncMock(return_value=case)

    payload = {
        "target_status": "ESCALATED",
        "reason": "Escalating case due to cross-border rapid fire transactions.",
    }
    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.patch(f"/api/v1/cases/{case.id}/status", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ESCALATED"


def test_list_case_notes_endpoint(client: TestClient, mock_service):
    """Verify GET /api/v1/cases/{case_id}/notes returns notes array."""
    note = CaseNote(
        id=uuid.uuid4(),
        case_id=uuid.uuid4(),
        author_id="analyst_1",
        author_role=AuditActorType.ANALYST,
        note_type=CaseNoteType.INVESTIGATION,
        content="Contacted cardholder to verify IP origin.",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    mock_service.get_notes = AsyncMock(return_value=[note])

    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.get(f"/api/v1/cases/{note.case_id}/notes", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["author_id"] == "analyst_1"


def test_add_case_note_endpoint(client: TestClient, mock_service):
    """Verify POST /api/v1/cases/{case_id}/notes appends a note."""
    cid = uuid.uuid4()
    note = CaseNote(
        id=uuid.uuid4(),
        case_id=cid,
        author_id="analyst_1",
        author_role=AuditActorType.ANALYST,
        note_type=CaseNoteType.INVESTIGATION,
        content="Customer verified identity through SMS OTP.",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    mock_service.add_note = AsyncMock(return_value=note)

    payload = {
        "content": "Customer verified identity through SMS OTP.",
        "note_type": "INVESTIGATION",
    }
    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.post(f"/api/v1/cases/{cid}/notes", json=payload, headers=headers)

    assert response.status_code == 201
    data = response.json()
    assert data["author_id"] == "analyst_1"
    assert data["content"] == "Customer verified identity through SMS OTP."


def test_submit_disposition_endpoint(client: TestClient, mock_service):
    """Verify POST /api/v1/cases/{case_id}/disposition records human outcome."""
    case = _make_mock_case_with_context()
    case.status = CaseStatus.RESOLVED
    case.disposition = CaseDisposition.CONFIRMED_FRAUD
    case.disposition_reason = "Cardholder verified device was stolen."
    case.dispositioned_by = "analyst_1"
    case.dispositioned_at = datetime.now(timezone.utc)
    case.resolved_at = datetime.now(timezone.utc)
    mock_service.record_disposition = AsyncMock(return_value=case)

    payload = {
        "disposition": "CONFIRMED_FRAUD",
        "reason": "Cardholder verified device was stolen.",
    }
    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.post(f"/api/v1/cases/{case.id}/disposition", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "RESOLVED"
    assert data["disposition"] == "CONFIRMED_FRAUD"
    assert data["dispositioned_by"] == "analyst_1"


def test_get_case_timeline_endpoint(client: TestClient, mock_service):
    """Verify GET /api/v1/cases/{case_id}/timeline returns AuditEntityType.CASE events."""
    case = _make_mock_case_with_context()
    audit_event = AuditLog(
        id=uuid.uuid4(),
        event_type="CASE_CREATED",
        entity_type=AuditEntityType.CASE,
        entity_id=case.id,
        action="CREATE_CASE",
        actor_type=AuditActorType.SYSTEM,
        actor_id="SYSTEM",
        correlation_id="cid-999",
        payload={"trigger_source": "AUTOMATED_REVIEW_POLICY"},
        event_timestamp=datetime.now(timezone.utc),
    )
    mock_service.get_timeline = AsyncMock(return_value=(case, [audit_event]))

    headers = {"X-Actor-ID": "analyst_1", "X-Actor-Role": "ANALYST"}
    response = client.get(f"/api/v1/cases/{case.id}/timeline", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == str(case.id)
    assert data["total_events"] == 1
    assert data["events"][0]["action"] == "CREATE_CASE"
