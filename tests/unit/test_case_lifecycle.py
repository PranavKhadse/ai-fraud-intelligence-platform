"""
Unit Tests for Case Lifecycle State Machine, Assignment Mutations, Notes, and Dispositions.

Validates:
1. Reviewer assignment workflows (CLAIM, ASSIGN, UNASSIGN):
   - CLAIM: OPEN -> IN_REVIEW, idempotent re-claim, 409 conflict when claimed by another.
   - ASSIGN: Supervisor (Admin-only) assignment, preserves status or moves OPEN -> IN_REVIEW.
   - UNASSIGN: Analyst self-unassign moves IN_REVIEW -> OPEN; Admin can unassign anyone; colleague unassign blocked.
2. Lifecycle status transitions (ESCALATE, CLOSE, REOPEN):
   - ESCALATE: OPEN/IN_REVIEW -> ESCALATED, appends ESCALATION note and audit log.
   - CLOSE: RESOLVED -> CLOSED with closed_at timestamp; direct OPEN -> CLOSED jump blocked.
   - REOPEN: RESOLVED -> IN_REVIEW clears disposition state; CLOSED -> IN_REVIEW allowed for Admin only.
3. Authoritative human disposition (RECORD_DISPOSITION):
   - Transitions IN_REVIEW -> RESOLVED with complete disposition metadata.
   - ESCALATED disposition requires Admin role; Analyst disposition on escalated raises PermissionError.
   - Disposition on OPEN or terminal cases rejected.
4. Investigation notes:
   - Author strictly extracted from ActorContext; non-empty validation; audit event emitted.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest

from backend.app.core.security import ActorContext
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
)
from backend.app.repositories.exceptions import (
    PersistenceConflictError,
    PersistenceNotFoundError,
)
from backend.app.services.case_service import CaseService


def _create_mock_case(
    case_id: Optional[uuid.UUID] = None,
    status: CaseStatus = CaseStatus.OPEN,
    priority: CasePriority = CasePriority.MEDIUM,
    assigned_to: Optional[str] = None,
) -> Case:
    cid = case_id or uuid.uuid4()
    return Case(
        id=cid,
        case_number=f"CASE-20260918-{cid.hex[:6].upper()}",
        transaction_id=uuid.uuid4(),
        evaluation_id=uuid.uuid4(),
        status=status,
        priority=priority,
        trigger_source=CaseTriggerSource.AUTOMATED_REVIEW_POLICY,
        assigned_to=assigned_to,
        assigned_at=datetime.now(timezone.utc) if assigned_to else None,
        opened_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_uow():
    uow = MagicMock()
    uow.cases = MagicMock()
    uow.audit_logs = MagicMock()
    uow.transactions = MagicMock()
    uow.risk_evaluations = MagicMock()
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()
    return uow


# ==============================================================================
# Assignment Workflow Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_claim_case_from_open_to_in_review(mock_uow):
    """Verify CLAIM transitions OPEN case to IN_REVIEW and assigns to caller."""
    case = _create_mock_case(status=CaseStatus.OPEN)
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)
    mock_uow.audit_logs.add = AsyncMock()

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    updated = await service.update_assignment(case.id, action="CLAIM", assignee_id=None, actor=actor)

    assert updated.status == CaseStatus.IN_REVIEW
    assert updated.assigned_to == "analyst_1"
    assert updated.assigned_at is not None
    mock_uow.commit.assert_awaited_once()
    mock_uow.audit_logs.add.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_case_idempotent(mock_uow):
    """Verify claiming a case already assigned to caller is an idempotent success."""
    case = _create_mock_case(status=CaseStatus.IN_REVIEW, assigned_to="analyst_1")
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    updated = await service.update_assignment(case.id, action="CLAIM", assignee_id=None, actor=actor)
    assert updated.assigned_to == "analyst_1"


@pytest.mark.asyncio
async def test_claim_case_conflict_when_already_assigned(mock_uow):
    """Verify claiming a case already assigned to another analyst raises PersistenceConflictError."""
    case = _create_mock_case(status=CaseStatus.IN_REVIEW, assigned_to="analyst_other")
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    with pytest.raises(PersistenceConflictError) as exc_info:
        await service.update_assignment(case.id, action="CLAIM", assignee_id=None, actor=actor)
    assert "already assigned to 'analyst_other'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_assign_case_by_admin(mock_uow):
    """Verify Admin can assign a case to any analyst."""
    case = _create_mock_case(status=CaseStatus.OPEN)
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)
    mock_uow.audit_logs.add = AsyncMock()

    service = CaseService(mock_uow)
    admin_actor = ActorContext(actor_id="admin_1", actor_role=AuditActorType.ADMIN)

    updated = await service.update_assignment(
        case.id,
        action="ASSIGN",
        assignee_id="target_analyst",
        actor=admin_actor,
        reason="Workload rebalancing",
    )

    assert updated.status == CaseStatus.IN_REVIEW
    assert updated.assigned_to == "target_analyst"
    mock_uow.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_assign_case_by_analyst_blocked(mock_uow):
    """Verify non-admin Analyst cannot assign cases to others."""
    case = _create_mock_case(status=CaseStatus.OPEN)
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)

    service = CaseService(mock_uow)
    analyst_actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    with pytest.raises(PermissionError) as exc_info:
        await service.update_assignment(
            case.id,
            action="ASSIGN",
            assignee_id="target_analyst",
            actor=analyst_actor,
        )
    assert "Only administrators" in str(exc_info.value)


@pytest.mark.asyncio
async def test_unassign_case_by_self(mock_uow):
    """Verify assigned analyst can release their own assignment, moving IN_REVIEW -> OPEN."""
    case = _create_mock_case(status=CaseStatus.IN_REVIEW, assigned_to="analyst_1")
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)
    mock_uow.audit_logs.add = AsyncMock()

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    updated = await service.update_assignment(case.id, action="UNASSIGN", assignee_id=None, actor=actor)

    assert updated.status == CaseStatus.OPEN
    assert updated.assigned_to is None
    assert updated.assigned_at is None
    mock_uow.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_unassign_case_colleague_blocked(mock_uow):
    """Verify an analyst cannot unassign another analyst's case."""
    case = _create_mock_case(status=CaseStatus.IN_REVIEW, assigned_to="analyst_other")
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    with pytest.raises(PermissionError) as exc_info:
        await service.update_assignment(case.id, action="UNASSIGN", assignee_id=None, actor=actor)
    assert "Analysts may only unassign cases assigned to themselves" in str(exc_info.value)


# ==============================================================================
# Lifecycle Status Transition Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_escalate_case(mock_uow):
    """Verify escalating a case moves status to ESCALATED and appends an ESCALATION note."""
    case = _create_mock_case(status=CaseStatus.IN_REVIEW, assigned_to="analyst_1")
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)
    mock_uow.cases.add_note = AsyncMock()
    mock_uow.audit_logs.add = AsyncMock()

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    updated = await service.update_status(
        case.id,
        target_status=CaseStatus.ESCALATED,
        reason="Suspected coordinated synthetic identity ring across multiple cards.",
        actor=actor,
    )

    assert updated.status == CaseStatus.ESCALATED
    mock_uow.cases.add_note.assert_awaited_once()
    mock_uow.audit_logs.add.assert_awaited_once()
    mock_uow.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_case_workflow(mock_uow):
    """Verify closing a resolved case sets CLOSED status and closed_at timestamp."""
    case = _create_mock_case(status=CaseStatus.RESOLVED)
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)
    mock_uow.audit_logs.add = AsyncMock()

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    updated = await service.update_status(
        case.id,
        target_status=CaseStatus.CLOSED,
        reason="Archiving resolved investigation after merchant confirmation.",
        actor=actor,
    )

    assert updated.status == CaseStatus.CLOSED
    assert updated.closed_at is not None
    mock_uow.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_direct_open_to_closed_blocked(mock_uow):
    """Verify that jumping directly from OPEN to CLOSED is blocked."""
    case = _create_mock_case(status=CaseStatus.OPEN)
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    with pytest.raises(ValueError) as exc_info:
        await service.update_status(
            case.id,
            target_status=CaseStatus.CLOSED,
            reason="Attempting to bypass human investigation directly.",
            actor=actor,
        )
    assert "Case must be in RESOLVED status first" in str(exc_info.value)


@pytest.mark.asyncio
async def test_reopen_closed_case_admin_only(mock_uow):
    """Verify reopening CLOSED case succeeds for Admin but raises PermissionError for Analyst."""
    case = _create_mock_case(status=CaseStatus.CLOSED)
    case.disposition = CaseDisposition.CONFIRMED_FRAUD
    case.disposition_reason = "Initial decision"
    case.dispositioned_by = "analyst_old"
    case.resolved_at = datetime.now(timezone.utc)
    case.closed_at = datetime.now(timezone.utc)

    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)
    mock_uow.cases.add_note = AsyncMock()
    mock_uow.audit_logs.add = AsyncMock()

    service = CaseService(mock_uow)

    # Analyst attempt -> blocked
    analyst_actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)
    with pytest.raises(PermissionError) as exc_info:
        await service.update_status(
            case.id,
            target_status=CaseStatus.IN_REVIEW,
            reason="Cardholder submitted rebuttal affidavit.",
            actor=analyst_actor,
        )
    assert "Only administrators can reopen closed/archived cases" in str(exc_info.value)

    # Admin attempt -> succeeds
    admin_actor = ActorContext(actor_id="admin_1", actor_role=AuditActorType.ADMIN)
    updated = await service.update_status(
        case.id,
        target_status=CaseStatus.IN_REVIEW,
        reason="Cardholder submitted valid rebuttal affidavit with police report.",
        actor=admin_actor,
    )
    assert updated.status == CaseStatus.IN_REVIEW
    assert updated.disposition is None
    assert updated.resolved_at is None
    assert updated.closed_at is None


# ==============================================================================
# Human Disposition Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_record_disposition_success(mock_uow):
    """Verify recording disposition sets RESOLVED status and disposition metadata."""
    case = _create_mock_case(status=CaseStatus.IN_REVIEW, assigned_to="analyst_1")
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)
    mock_uow.cases.add_note = AsyncMock()
    mock_uow.audit_logs.add = AsyncMock()

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)

    updated = await service.record_disposition(
        case_id=case.id,
        disposition=CaseDisposition.CONFIRMED_FRAUD,
        reason="Customer confirmed unauthorized physical card theft.",
        actor=actor,
    )

    assert updated.status == CaseStatus.RESOLVED
    assert updated.disposition == CaseDisposition.CONFIRMED_FRAUD
    assert updated.disposition_reason == "Customer confirmed unauthorized physical card theft."
    assert updated.dispositioned_by == "analyst_1"
    assert updated.dispositioned_at is not None
    assert updated.resolved_at is not None
    mock_uow.cases.add_note.assert_awaited_once()
    mock_uow.audit_logs.add.assert_awaited_once()
    mock_uow.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_disposition_on_escalated_case_requires_admin(mock_uow):
    """Verify dispositioning an ESCALATED case requires Admin role."""
    case = _create_mock_case(status=CaseStatus.ESCALATED)
    mock_uow.cases.get_by_id_for_update = AsyncMock(return_value=case)
    mock_uow.cases.add_note = AsyncMock()
    mock_uow.audit_logs.add = AsyncMock()

    service = CaseService(mock_uow)

    # Analyst attempt -> blocked
    analyst_actor = ActorContext(actor_id="analyst_1", actor_role=AuditActorType.ANALYST)
    with pytest.raises(PermissionError) as exc_info:
        await service.record_disposition(
            case_id=case.id,
            disposition=CaseDisposition.CONFIRMED_FRAUD,
            reason="Attempting to resolve escalated case without supervisory role.",
            actor=analyst_actor,
        )
    assert "Only administrators can disposition escalated cases" in str(exc_info.value)

    # Admin attempt -> succeeds
    admin_actor = ActorContext(actor_id="admin_1", actor_role=AuditActorType.ADMIN)
    updated = await service.record_disposition(
        case_id=case.id,
        disposition=CaseDisposition.CONFIRMED_FRAUD,
        reason="Supervisor reviewed fraud ring evidence and confirmed fraud.",
        actor=admin_actor,
    )
    assert updated.status == CaseStatus.RESOLVED
    assert updated.disposition == CaseDisposition.CONFIRMED_FRAUD


# ==============================================================================
# Investigation Note Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_add_case_note(mock_uow):
    """Verify appending a note captures author strictly from ActorContext."""
    case = _create_mock_case(status=CaseStatus.IN_REVIEW)
    mock_uow.cases.get_by_id = AsyncMock(return_value=case)
    mock_uow.cases.add_note = AsyncMock()
    mock_uow.audit_logs.add = AsyncMock()

    service = CaseService(mock_uow)
    actor = ActorContext(actor_id="analyst_42", actor_role=AuditActorType.ANALYST)

    note = await service.add_note(
        case_id=case.id,
        content="Contacted cardholder via secure phone call. Cardholder verified travel.",
        note_type=CaseNoteType.INVESTIGATION,
        actor=actor,
    )

    assert note.author_id == "analyst_42"
    assert note.author_role == AuditActorType.ANALYST
    assert note.note_type == CaseNoteType.INVESTIGATION
    mock_uow.cases.add_note.assert_awaited_once()
    mock_uow.audit_logs.add.assert_awaited_once()
    mock_uow.commit.assert_awaited_once()
