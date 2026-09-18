"""
Case Management Domain Service for Human Review Workflows.

Provides domain operations for manual case creation, transaction/evaluation
integrity validation, case-number generation, duplicate case detection,
reviewer assignments, lifecycle status transitions, notes, human dispositions,
queue querying, operational metrics, and audit timeline tracking under a unified
Unit of Work boundary with PostgreSQL row-level locking.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

from fastapi import Depends
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

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
    RiskTier,
)
from backend.app.db.session import get_db_session
from backend.app.repositories.exceptions import (
    PersistenceConflictError,
    PersistenceError,
    PersistenceNotFoundError,
)
from backend.app.services.unit_of_work import FraudPersistenceUnitOfWork


# ==============================================================================
# Domain Utilities & Reference Generators
# ==============================================================================

def generate_case_number(timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique, collision-safe, human-readable case reference number.

    Format: CASE-YYYYMMDD-XXXXXX (e.g. 'CASE-20260918-A1B2C3')
    - Matches approved database constraint VARCHAR(32).
    - Preserves chronological sortability and human readability.

    Args:
        timestamp: Optional reference timestamp. Defaults to current UTC time.

    Returns:
        Formatted case reference string.
    """
    dt = timestamp or datetime.now(timezone.utc)
    date_str = dt.strftime("%Y%m%d")
    unique_suffix = uuid.uuid4().hex[:6].upper()
    return f"CASE-{date_str}-{unique_suffix}"


# ==============================================================================
# Command DTOs
# ==============================================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class CreateManualCaseCommand:
    """
    Input command for analyst-initiated manual case escalation.
    """
    transaction_id: uuid.UUID
    initial_note: str
    actor: ActorContext
    priority: Optional[Union[CasePriority, str]] = None
    evaluation_id: Optional[uuid.UUID] = None


# ==============================================================================
# Service Implementation
# ==============================================================================

class CaseService:
    """
    Domain service orchestrating case creation, lifecycle mutations, notes, and metrics.

    Design Principles:
    - Transactional Unit of Work: Operates through an injected `FraudPersistenceUnitOfWork`.
    - PostgreSQL Row-Level Locking: Uses `SELECT ... FOR UPDATE` to serialize concurrent mutations
      of the same locked `Case` row within the transaction/UoW.
    - Strict 1-to-1 Cardinality: Rejects duplicate case creation per transaction ID.
    - Author Extraction: Captures note authors and dispositioners strictly from authenticated `ActorContext`.
    - Bounded Audit Payload: Enforces strict <= 1 KB payload size for all audit log entries.
    """

    def __init__(self, uow: FraudPersistenceUnitOfWork) -> None:
        """
        Initialize the CaseService with an active Unit of Work.

        Args:
            uow: Injected FraudPersistenceUnitOfWork instance.

        Raises:
            ValueError: If uow is None.
        """
        if uow is None:
            raise ValueError("FraudPersistenceUnitOfWork must not be None.")
        self._uow = uow

    async def create_manual_case(
        self,
        command: CreateManualCaseCommand,
    ) -> Case:
        """
        Manually escalate an existing financial transaction to human review.

        Workflow:
        1. Validate non-empty initial note.
        2. Verify existence of the referenced Transaction.
        3. Validate or resolve the associated RiskEvaluation (rejecting transaction mismatches).
        4. Check for existing Case on this transaction (pre-check for fast 409 conflict).
        5. Determine priority (from explicit command or derived from evaluation risk tier).
        6. Stage Case entity with status OPEN and trigger MANUAL_ANALYST_ESCALATION.
        7. Stage initial investigation CaseNote.
        8. Stage CASE_CREATED audit log event with bounded payload.
        9. Commit atomically; translate race condition database IntegrityError to PersistenceConflictError.

        Args:
            command: CreateManualCaseCommand containing transaction reference, note, and actor.

        Returns:
            The newly created and committed Case ORM entity.

        Raises:
            ValueError: If initial note is empty or whitespace-only.
            PersistenceNotFoundError: If the transaction or evaluation cannot be found.
            PersistenceError: If evaluation does not belong to the transaction or database write fails.
            PersistenceConflictError: If a case already exists for the transaction.
        """
        if not command.initial_note or not command.initial_note.strip():
            raise ValueError("Initial case note cannot be empty or whitespace-only.")

        actor_role = (
            AuditActorType(command.actor.actor_role)
            if isinstance(command.actor.actor_role, str)
            else command.actor.actor_role
        )

        try:
            # 2. Verify Transaction existence
            tx = await self._uow.transactions.get_by_id(command.transaction_id)
            if tx is None:
                raise PersistenceNotFoundError(
                    f"Transaction '{command.transaction_id}' not found.",
                    details={"transaction_id": str(command.transaction_id)},
                )

            # 3. Resolve & validate RiskEvaluation
            if command.evaluation_id is not None:
                eval_entity = await self._uow.risk_evaluations.get_by_id(command.evaluation_id)
                if eval_entity is None:
                    raise PersistenceNotFoundError(
                        f"RiskEvaluation '{command.evaluation_id}' not found.",
                        details={"evaluation_id": str(command.evaluation_id)},
                    )
                if eval_entity.transaction_id != command.transaction_id:
                    raise PersistenceError(
                        f"Evaluation '{command.evaluation_id}' does not belong to Transaction '{command.transaction_id}'.",
                        details={
                            "transaction_id": str(command.transaction_id),
                            "evaluation_id": str(command.evaluation_id),
                        },
                    )
            else:
                evals = await self._uow.risk_evaluations.get_by_transaction_id(command.transaction_id)
                if not evals:
                    raise PersistenceNotFoundError(
                        f"No risk evaluation found for transaction '{command.transaction_id}'.",
                        details={"transaction_id": str(command.transaction_id)},
                    )
                eval_entity = evals[0]

            # 4. Pre-check for duplicate case
            existing_case = await self._uow.cases.get_by_transaction_id(command.transaction_id)
            if existing_case is not None:
                raise PersistenceConflictError(
                    f"A case already exists for transaction '{command.transaction_id}'.",
                    details={
                        "transaction_id": str(command.transaction_id),
                        "existing_case_id": str(existing_case.id),
                        "existing_case_number": existing_case.case_number,
                    },
                )

            # 5. Determine priority
            if command.priority is not None:
                priority = (
                    CasePriority(command.priority)
                    if isinstance(command.priority, str)
                    else command.priority
                )
            else:
                tier_priority_map = {
                    RiskTier.CRITICAL: CasePriority.CRITICAL,
                    RiskTier.HIGH: CasePriority.HIGH,
                    RiskTier.MEDIUM: CasePriority.MEDIUM,
                    RiskTier.LOW: CasePriority.LOW,
                }
                priority = tier_priority_map.get(eval_entity.risk_tier, CasePriority.MEDIUM)

            # 6. Stage Case entity
            case_id = uuid.uuid4()
            now_utc = datetime.now(timezone.utc)
            case_number = generate_case_number(now_utc)

            case_entity = Case(
                id=case_id,
                case_number=case_number,
                transaction_id=command.transaction_id,
                evaluation_id=eval_entity.id,
                status=CaseStatus.OPEN,
                priority=priority,
                trigger_source=CaseTriggerSource.MANUAL_ANALYST_ESCALATION,
                opened_at=now_utc,
            )
            await self._uow.cases.add(case_entity)

            # 7. Stage initial investigation note
            note_id = uuid.uuid4()
            note_entity = CaseNote(
                id=note_id,
                case_id=case_id,
                author_id=command.actor.actor_id,
                author_role=actor_role,
                note_type=CaseNoteType.INVESTIGATION,
                content=command.initial_note.strip(),
            )
            await self._uow.cases.add_note(note_entity)

            # 8. Stage CASE_CREATED audit log event
            audit_entity = AuditLog(
                id=uuid.uuid4(),
                event_type="CASE_CREATED",
                entity_type=AuditEntityType.CASE,
                entity_id=case_id,
                action="CREATE_CASE",
                actor_type=actor_role,
                actor_id=command.actor.actor_id,
                correlation_id=command.actor.correlation_id,
                client_ip=command.actor.client_ip,
                payload={
                    "case_number": case_number,
                    "transaction_id": str(command.transaction_id),
                    "evaluation_id": str(eval_entity.id),
                    "trigger_source": CaseTriggerSource.MANUAL_ANALYST_ESCALATION.value,
                    "priority": priority.value,
                },
                event_timestamp=now_utc,
            )
            await self._uow.audit_logs.add(audit_entity)

            # 9. Commit atomic transaction
            await self._uow.commit()
            return case_entity

        except (PersistenceConflictError, PersistenceNotFoundError, PersistenceError, ValueError):
            await self._uow.rollback()
            raise
        except IntegrityError as exc:
            await self._uow.rollback()
            err_msg = str(exc).lower()
            orig_msg = str(getattr(exc, "orig", "")).lower()
            is_case_conflict = (
                "uq_cases_transaction_id" in err_msg
                or "uq_cases_transaction_id" in orig_msg
                or "cases" in err_msg
                or "unique" in err_msg
                or "duplicate" in err_msg
            )
            if is_case_conflict:
                existing = await self._uow.cases.get_by_transaction_id(command.transaction_id)
                raise PersistenceConflictError(
                    f"A case already exists for transaction '{command.transaction_id}'.",
                    details={
                        "transaction_id": str(command.transaction_id),
                        "existing_case_id": str(existing.id) if existing else None,
                        "existing_case_number": existing.case_number if existing else None,
                    },
                ) from exc
            raise PersistenceError(
                f"Database integrity constraint violation during case creation: {exc}",
                details={"orig": str(exc)},
            ) from exc
        except Exception:
            await self._uow.rollback()
            raise

    async def update_assignment(
        self,
        case_id: uuid.UUID,
        action: str,
        assignee_id: Optional[str],
        actor: ActorContext,
        reason: Optional[str] = None,
    ) -> Case:
        """
        Mutate reviewer assignment for a case (CLAIM, ASSIGN, UNASSIGN) under row lock.

        Behavior:
        - CLAIM: Analyst/Admin self-claim. Transitions OPEN -> IN_REVIEW. Idempotent if already assigned to caller.
        - ASSIGN: Admin supervisor assignment. Transitions OPEN -> IN_REVIEW; preserves IN_REVIEW/ESCALATED.
        - UNASSIGN: Self-unassign (Analyst) or Any-unassign (Admin). Transitions IN_REVIEW -> OPEN; preserves ESCALATED.

        Args:
            case_id: Target Case UUID.
            action: Mutation action ('CLAIM', 'ASSIGN', 'UNASSIGN').
            assignee_id: Target assignee ID (required for ASSIGN).
            actor: Authenticated ActorContext.
            reason: Optional operational justification.

        Returns:
            Updated Case entity.

        Raises:
            PersistenceNotFoundError: If case not found.
            PersistenceConflictError: If already claimed by another analyst.
            PermissionError: If caller lacks permissions (e.g. non-admin assigning others).
            ValueError: If invalid action or transition from resolved/closed status.
        """
        action_normalized = action.strip().upper()
        now_utc = datetime.now(timezone.utc)
        actor_role = (
            AuditActorType(actor.actor_role)
            if isinstance(actor.actor_role, str)
            else actor.actor_role
        )

        try:
            # 1. Acquire exclusive row lock
            case = await self._uow.cases.get_by_id_for_update(case_id)
            if case is None:
                raise PersistenceNotFoundError(
                    f"Case '{case_id}' not found.",
                    details={"case_id": str(case_id)},
                )

            # 2. Check terminal states
            if case.status in (CaseStatus.RESOLVED, CaseStatus.CLOSED):
                raise ValueError(
                    f"Cannot perform assignment action '{action_normalized}' on a {case.status.value} case."
                )

            prev_status = case.status
            prev_assigned_to = case.assigned_to

            # 3. Action-specific logic
            if action_normalized == "CLAIM":
                if case.status == CaseStatus.ESCALATED:
                    raise ValueError("Cannot self-claim an escalated case. Escalated cases require supervisor assignment.")

                if case.assigned_to == actor.actor_id:
                    # Idempotent return: commit releases row lock without modifying state while preserving loaded attributes
                    await self._uow.commit()
                    return case

                if case.assigned_to is not None and case.assigned_to != actor.actor_id:
                    raise PersistenceConflictError(
                        f"Case '{case.case_number}' is already assigned to '{case.assigned_to}'.",
                        details={
                            "case_id": str(case_id),
                            "assigned_to": case.assigned_to,
                            "attempted_by": actor.actor_id,
                        },
                    )

                if case.status == CaseStatus.OPEN:
                    case.status = CaseStatus.IN_REVIEW

                case.assigned_to = actor.actor_id
                case.assigned_at = now_utc

                audit_event = AuditLog(
                    id=uuid.uuid4(),
                    event_type="CASE_ASSIGNED",
                    entity_type=AuditEntityType.CASE,
                    entity_id=case_id,
                    action="CLAIM_CASE",
                    actor_type=actor_role,
                    actor_id=actor.actor_id,
                    correlation_id=actor.correlation_id,
                    client_ip=actor.client_ip,
                    payload={
                        "case_number": case.case_number,
                        "assigned_to": actor.actor_id,
                        "previous_status": prev_status.value,
                        "new_status": case.status.value,
                        "previous_assigned_to": prev_assigned_to,
                    },
                    event_timestamp=now_utc,
                )
                await self._uow.audit_logs.add(audit_event)

            elif action_normalized == "ASSIGN":
                if actor_role != AuditActorType.ADMIN:
                    raise PermissionError("Only administrators can assign cases to other reviewers.")

                if not assignee_id or not assignee_id.strip():
                    raise ValueError("assignee_id is required when action is 'ASSIGN'.")

                clean_assignee = assignee_id.strip()

                if case.status == CaseStatus.OPEN:
                    case.status = CaseStatus.IN_REVIEW

                case.assigned_to = clean_assignee
                case.assigned_at = now_utc

                audit_event = AuditLog(
                    id=uuid.uuid4(),
                    event_type="CASE_ASSIGNED",
                    entity_type=AuditEntityType.CASE,
                    entity_id=case_id,
                    action="ASSIGN_CASE",
                    actor_type=actor_role,
                    actor_id=actor.actor_id,
                    correlation_id=actor.correlation_id,
                    client_ip=actor.client_ip,
                    payload={
                        "case_number": case.case_number,
                        "assigned_to": clean_assignee,
                        "assigned_by": actor.actor_id,
                        "previous_status": prev_status.value,
                        "new_status": case.status.value,
                        "previous_assigned_to": prev_assigned_to,
                        "reason": reason,
                    },
                    event_timestamp=now_utc,
                )
                await self._uow.audit_logs.add(audit_event)

            elif action_normalized == "UNASSIGN":
                if actor_role != AuditActorType.ADMIN and case.assigned_to != actor.actor_id:
                    raise PermissionError("Analysts may only unassign cases assigned to themselves.")

                if case.assigned_to is None:
                    # Idempotent return: commit releases row lock without modifying state while preserving loaded attributes
                    await self._uow.commit()
                    return case

                if case.status == CaseStatus.IN_REVIEW:
                    case.status = CaseStatus.OPEN

                case.assigned_to = None
                case.assigned_at = None

                audit_event = AuditLog(
                    id=uuid.uuid4(),
                    event_type="CASE_UNASSIGNED",
                    entity_type=AuditEntityType.CASE,
                    entity_id=case_id,
                    action="UNASSIGN_CASE",
                    actor_type=actor_role,
                    actor_id=actor.actor_id,
                    correlation_id=actor.correlation_id,
                    client_ip=actor.client_ip,
                    payload={
                        "case_number": case.case_number,
                        "unassigned_by": actor.actor_id,
                        "previous_assigned_to": prev_assigned_to,
                        "previous_status": prev_status.value,
                        "new_status": case.status.value,
                        "reason": reason,
                    },
                    event_timestamp=now_utc,
                )
                await self._uow.audit_logs.add(audit_event)

            else:
                raise ValueError(f"Unsupported assignment action '{action}'. Valid: CLAIM, ASSIGN, UNASSIGN.")

            await self._uow.commit()
            return case

        except (PersistenceNotFoundError, PersistenceConflictError, PermissionError, ValueError):
            await self._uow.rollback()
            raise
        except Exception:
            await self._uow.rollback()
            raise

    async def update_status(
        self,
        case_id: uuid.UUID,
        target_status: Union[CaseStatus, str],
        reason: str,
        actor: ActorContext,
    ) -> Case:
        """
        Transition case lifecycle status (ESCALATE, CLOSE, REOPEN) under row lock.

        Args:
            case_id: Target Case UUID.
            target_status: Target CaseStatus.
            reason: Mandatory reason (min 10 characters).
            actor: Authenticated ActorContext.

        Returns:
            Updated Case entity.

        Raises:
            PersistenceNotFoundError: If case not found.
            PermissionError: If caller lacks authorization.
            ValueError: If invalid transition or empty reason.
        """
        if not reason or len(reason.strip()) < 10:
            raise ValueError("Status update reason must be at least 10 non-whitespace characters.")

        clean_reason = reason.strip()
        status_enum = (
            CaseStatus(target_status)
            if isinstance(target_status, str)
            else target_status
        )
        actor_role = (
            AuditActorType(actor.actor_role)
            if isinstance(actor.actor_role, str)
            else actor.actor_role
        )
        now_utc = datetime.now(timezone.utc)

        try:
            case = await self._uow.cases.get_by_id_for_update(case_id)
            if case is None:
                raise PersistenceNotFoundError(
                    f"Case '{case_id}' not found.",
                    details={"case_id": str(case_id)},
                )

            prev_status = case.status

            if status_enum == CaseStatus.ESCALATED:
                if case.status not in (CaseStatus.OPEN, CaseStatus.IN_REVIEW):
                    raise ValueError(f"Cannot escalate a case in '{case.status.value}' status. Must be OPEN or IN_REVIEW.")

                case.status = CaseStatus.ESCALATED

                # Append ESCALATION note
                note_entity = CaseNote(
                    id=uuid.uuid4(),
                    case_id=case_id,
                    author_id=actor.actor_id,
                    author_role=actor_role,
                    note_type=CaseNoteType.ESCALATION,
                    content=f"Case escalated to senior review. Reason: {clean_reason}",
                )
                await self._uow.cases.add_note(note_entity)

                # Audit log
                audit_event = AuditLog(
                    id=uuid.uuid4(),
                    event_type="CASE_STATUS_CHANGED",
                    entity_type=AuditEntityType.CASE,
                    entity_id=case_id,
                    action="ESCALATE_CASE",
                    actor_type=actor_role,
                    actor_id=actor.actor_id,
                    correlation_id=actor.correlation_id,
                    client_ip=actor.client_ip,
                    payload={
                        "case_number": case.case_number,
                        "previous_status": prev_status.value,
                        "new_status": CaseStatus.ESCALATED.value,
                        "reason": clean_reason,
                    },
                    event_timestamp=now_utc,
                )
                await self._uow.audit_logs.add(audit_event)

            elif status_enum == CaseStatus.CLOSED:
                if case.status != CaseStatus.RESOLVED:
                    raise ValueError(f"Cannot close a case in '{case.status.value}' status. Case must be in RESOLVED status first.")

                case.status = CaseStatus.CLOSED
                case.closed_at = now_utc

                audit_event = AuditLog(
                    id=uuid.uuid4(),
                    event_type="CASE_STATUS_CHANGED",
                    entity_type=AuditEntityType.CASE,
                    entity_id=case_id,
                    action="CLOSE_CASE",
                    actor_type=actor_role,
                    actor_id=actor.actor_id,
                    correlation_id=actor.correlation_id,
                    client_ip=actor.client_ip,
                    payload={
                        "case_number": case.case_number,
                        "previous_status": prev_status.value,
                        "new_status": CaseStatus.CLOSED.value,
                        "reason": clean_reason,
                    },
                    event_timestamp=now_utc,
                )
                await self._uow.audit_logs.add(audit_event)

            elif status_enum == CaseStatus.IN_REVIEW:
                # REOPEN workflow
                if case.status not in (CaseStatus.RESOLVED, CaseStatus.CLOSED):
                    raise ValueError(f"Cannot reopen a case in '{case.status.value}' status. Must be RESOLVED or CLOSED.")

                if case.status == CaseStatus.CLOSED and actor_role != AuditActorType.ADMIN:
                    raise PermissionError("Only administrators can reopen closed/archived cases.")

                case.status = CaseStatus.IN_REVIEW
                case.resolved_at = None
                case.closed_at = None
                case.disposition = None
                case.disposition_reason = None
                case.dispositioned_by = None
                case.dispositioned_at = None

                note_entity = CaseNote(
                    id=uuid.uuid4(),
                    case_id=case_id,
                    author_id=actor.actor_id,
                    author_role=actor_role,
                    note_type=CaseNoteType.INVESTIGATION,
                    content=f"Case reopened: {clean_reason}",
                )
                await self._uow.cases.add_note(note_entity)

                audit_event = AuditLog(
                    id=uuid.uuid4(),
                    event_type="CASE_REOPENED",
                    entity_type=AuditEntityType.CASE,
                    entity_id=case_id,
                    action="REOPEN_CASE",
                    actor_type=actor_role,
                    actor_id=actor.actor_id,
                    correlation_id=actor.correlation_id,
                    client_ip=actor.client_ip,
                    payload={
                        "case_number": case.case_number,
                        "previous_status": prev_status.value,
                        "new_status": CaseStatus.IN_REVIEW.value,
                        "reason": clean_reason,
                    },
                    event_timestamp=now_utc,
                )
                await self._uow.audit_logs.add(audit_event)

            else:
                raise ValueError(
                    f"Direct transition to '{status_enum.value}' via status update is not permitted. "
                    "Use disposition endpoint to resolve cases or assignment endpoint to claim."
                )

            await self._uow.commit()
            return case

        except (PersistenceNotFoundError, PermissionError, ValueError):
            await self._uow.rollback()
            raise
        except Exception:
            await self._uow.rollback()
            raise

    async def record_disposition(
        self,
        case_id: uuid.UUID,
        disposition: Union[CaseDisposition, str],
        reason: str,
        actor: ActorContext,
    ) -> Case:
        """
        Record the authoritative human review outcome under row lock, transitioning status to RESOLVED.

        Args:
            case_id: Target Case UUID.
            disposition: Outcome (CONFIRMED_FRAUD, FALSE_POSITIVE, LEGITIMATE, SUSPICIOUS_RESOLVED).
            reason: Plain-English explanation (min 10 characters).
            actor: Authenticated ActorContext.

        Returns:
            Updated Case entity with status RESOLVED.

        Raises:
            PersistenceNotFoundError: If case not found.
            PermissionError: If an analyst attempts to disposition an escalated case (Admin only).
            ValueError: If case is not in IN_REVIEW or ESCALATED status, or reason is too short.
        """
        if not reason or len(reason.strip()) < 10:
            raise ValueError("Disposition reason must be at least 10 non-whitespace characters.")

        clean_reason = reason.strip()
        disp_enum = (
            CaseDisposition(disposition)
            if isinstance(disposition, str)
            else disposition
        )
        actor_role = (
            AuditActorType(actor.actor_role)
            if isinstance(actor.actor_role, str)
            else actor.actor_role
        )
        now_utc = datetime.now(timezone.utc)

        try:
            case = await self._uow.cases.get_by_id_for_update(case_id)
            if case is None:
                raise PersistenceNotFoundError(
                    f"Case '{case_id}' not found.",
                    details={"case_id": str(case_id)},
                )

            if case.status not in (CaseStatus.IN_REVIEW, CaseStatus.ESCALATED):
                raise ValueError(
                    f"Cannot record disposition on a case in '{case.status.value}' status. Case must be IN_REVIEW or ESCALATED."
                )

            if case.status == CaseStatus.ESCALATED and actor_role != AuditActorType.ADMIN:
                raise PermissionError("Only administrators can disposition escalated cases.")

            prev_status = case.status

            # Atomic disposition fields mutation
            case.disposition = disp_enum
            case.disposition_reason = clean_reason
            case.dispositioned_by = actor.actor_id
            case.dispositioned_at = now_utc
            case.resolved_at = now_utc
            case.status = CaseStatus.RESOLVED

            # Append DISPOSITION note
            note_entity = CaseNote(
                id=uuid.uuid4(),
                case_id=case_id,
                author_id=actor.actor_id,
                author_role=actor_role,
                note_type=CaseNoteType.DISPOSITION,
                content=f"Disposition recorded: {disp_enum.value}. Reason: {clean_reason}",
            )
            await self._uow.cases.add_note(note_entity)

            # Audit log
            audit_event = AuditLog(
                id=uuid.uuid4(),
                event_type="CASE_DISPOSITIONED",
                entity_type=AuditEntityType.CASE,
                entity_id=case_id,
                action="RECORD_DISPOSITION",
                actor_type=actor_role,
                actor_id=actor.actor_id,
                correlation_id=actor.correlation_id,
                client_ip=actor.client_ip,
                payload={
                    "case_number": case.case_number,
                    "previous_status": prev_status.value,
                    "new_status": CaseStatus.RESOLVED.value,
                    "disposition": disp_enum.value,
                    "dispositioned_by": actor.actor_id,
                    "reason": clean_reason,
                },
                event_timestamp=now_utc,
            )
            await self._uow.audit_logs.add(audit_event)

            await self._uow.commit()
            return case

        except (PersistenceNotFoundError, PermissionError, ValueError):
            await self._uow.rollback()
            raise
        except Exception:
            await self._uow.rollback()
            raise

    async def add_note(
        self,
        case_id: uuid.UUID,
        content: str,
        note_type: Union[CaseNoteType, str],
        actor: ActorContext,
    ) -> CaseNote:
        """
        Append an investigation note to a case and stage a CASE_NOTE_ADDED audit event.

        Args:
            case_id: Target Case UUID.
            content: Note text.
            note_type: CaseNoteType.
            actor: Authenticated ActorContext.

        Returns:
            The created CaseNote entity.

        Raises:
            PersistenceNotFoundError: If case not found.
            ValueError: If content is empty.
        """
        if not content or not content.strip():
            raise ValueError("Note content cannot be empty or whitespace-only.")

        clean_content = content.strip()
        ntype_enum = (
            CaseNoteType(note_type)
            if isinstance(note_type, str)
            else note_type
        )
        actor_role = (
            AuditActorType(actor.actor_role)
            if isinstance(actor.actor_role, str)
            else actor.actor_role
        )
        now_utc = datetime.now(timezone.utc)

        try:
            # Check case existence
            case = await self._uow.cases.get_by_id(case_id)
            if case is None:
                raise PersistenceNotFoundError(
                    f"Case '{case_id}' not found.",
                    details={"case_id": str(case_id)},
                )

            note_id = uuid.uuid4()
            note_entity = CaseNote(
                id=note_id,
                case_id=case_id,
                author_id=actor.actor_id,
                author_role=actor_role,
                note_type=ntype_enum,
                content=clean_content,
            )
            await self._uow.cases.add_note(note_entity)

            # Stage audit log for note addition
            preview = clean_content[:200] + ("..." if len(clean_content) > 200 else "")
            audit_event = AuditLog(
                id=uuid.uuid4(),
                event_type="CASE_NOTE_ADDED",
                entity_type=AuditEntityType.CASE,
                entity_id=case_id,
                action="ADD_NOTE",
                actor_type=actor_role,
                actor_id=actor.actor_id,
                correlation_id=actor.correlation_id,
                client_ip=actor.client_ip,
                payload={
                    "case_number": case.case_number,
                    "note_id": str(note_id),
                    "note_type": ntype_enum.value,
                    "author_id": actor.actor_id,
                    "content_preview": preview,
                    "content_length": len(clean_content),
                },
                event_timestamp=now_utc,
            )
            await self._uow.audit_logs.add(audit_event)

            await self._uow.commit()
            return note_entity

        except (PersistenceNotFoundError, ValueError):
            await self._uow.rollback()
            raise
        except Exception:
            await self._uow.rollback()
            raise

    async def get_case_detail(
        self,
        case_id: uuid.UUID,
    ) -> Case:
        """
        Retrieve deeply loaded case entity for the investigation workspace.

        Args:
            case_id: Target Case UUID.

        Returns:
            Case entity with transaction, evaluation, and child collections loaded.

        Raises:
            PersistenceNotFoundError: If case not found.
        """
        case = await self._uow.cases.get_case_detail(case_id)
        if case is None:
            raise PersistenceNotFoundError(
                f"Case '{case_id}' not found.",
                details={"case_id": str(case_id)},
            )
        return case

    async def get_notes(
        self,
        case_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CaseNote]:
        """
        Retrieve chronological investigation notes for a case.

        Args:
            case_id: Target Case UUID.
            limit: Maximum items to return.
            offset: Items to skip.

        Returns:
            List of CaseNote entities.

        Raises:
            PersistenceNotFoundError: If case not found.
        """
        case = await self._uow.cases.get_by_id(case_id)
        if case is None:
            raise PersistenceNotFoundError(
                f"Case '{case_id}' not found.",
                details={"case_id": str(case_id)},
            )

        notes = await self._uow.cases.get_notes(case_id)
        return notes[offset : offset + limit]

    async def get_timeline(
        self,
        case_id: uuid.UUID,
        limit: int = 100,
    ) -> Tuple[Case, List[AuditLog]]:
        """
        Retrieve chronological audit history targeting AuditEntityType.CASE.

        Args:
            case_id: Target Case UUID.
            limit: Maximum events to return.

        Returns:
            Tuple of (Case entity, List of AuditLog entities).

        Raises:
            PersistenceNotFoundError: If case not found.
        """
        case = await self._uow.cases.get_by_id(case_id)
        if case is None:
            raise PersistenceNotFoundError(
                f"Case '{case_id}' not found.",
                details={"case_id": str(case_id)},
            )

        events = await self._uow.cases.get_timeline_events(case_id, limit=limit)
        return case, events

    async def get_queue_cases(
        self,
        status: Optional[CaseStatus] = None,
        priority: Optional[CasePriority] = None,
        assigned_to: Optional[str] = None,
        risk_tier: Optional[RiskTier] = None,
        min_score: Optional[int] = None,
        max_score: Optional[int] = None,
        search_term: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "opened_at",
        sort_order: str = "desc",
    ) -> Tuple[List[Case], int]:
        """
        Query review queue cases with pagination and multi-dimensional filters.
        """
        return await self._uow.cases.get_queue_cases(
            status=status,
            priority=priority,
            assigned_to=assigned_to,
            risk_tier=risk_tier,
            min_score=min_score,
            max_score=max_score,
            search_term=search_term,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    async def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Aggregate queue operational KPI counts.
        """
        return await self._uow.cases.get_summary_metrics()


def get_case_service(
    session: AsyncSession = Depends(get_db_session),
) -> CaseService:
    """
    FastAPI dependency provider constructing a request-scoped CaseService
    backed by an active AsyncSession Unit of Work.
    """
    uow = FraudPersistenceUnitOfWork(session)
    return CaseService(uow)


__all__ = [
    "CaseService",
    "get_case_service",
    "ActorContext",
    "CreateManualCaseCommand",
    "generate_case_number",
]
