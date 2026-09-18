"""
REST API Endpoints for Case Management, Human Review Workflows, and Lifecycle State Machine.

Exposes endpoints for:
- Paginated review queues with multi-dimensional filtering
- Operational queue summary KPIs
- Deep investigation workspace details
- Analyst-initiated manual case escalation
- Reviewer assignment mutations (CLAIM, ASSIGN, UNASSIGN)
- Lifecycle status transitions (ESCALATE, CLOSE, REOPEN)
- Append-only chronological investigation notes
- Authoritative human review disposition recording
- Chronological audit timeline inspection
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.core.security import ActorContext, get_current_actor, require_role
from backend.app.db.models.case import Case, CaseNote
from backend.app.db.models.enums import (
    AuditActorType,
    CaseDisposition,
    CaseNoteType,
    CasePriority,
    CaseStatus,
    CaseTriggerSource,
    DecisionAction,
    RiskTier,
)
from backend.app.repositories.exceptions import (
    PersistenceConflictError,
    PersistenceError,
    PersistenceNotFoundError,
)
from backend.app.schemas.case import (
    CaseAssignmentRequest,
    CaseDetailResponse,
    CaseDispositionRequest,
    CaseEvaluationContext,
    CaseListResponse,
    CaseNoteItem,
    CaseQueueItem,
    CaseResponse,
    CaseStatusUpdateRequest,
    CaseSummaryResponse,
    CaseTimelineResponse,
    CaseTransactionContext,
    CreateCaseNoteRequest,
    CreateCaseRequest,
    TimelineEventItem,
)
from backend.app.schemas.predict import (
    FeatureAttributionResponse,
    ReasonCodeResponse,
    RuleMatchResponse,
)
from backend.app.services.case_service import (
    CaseService,
    CreateManualCaseCommand,
    get_case_service,
)

router = APIRouter(prefix="/cases")


# ==============================================================================
# Helper Serialization Functions
# ==============================================================================

def _map_case_to_queue_item(c: Case) -> CaseQueueItem:
    """
    Project a Case entity into a compact queue item.
    """
    tx = c.transaction
    ev = c.evaluation
    return CaseQueueItem(
        id=c.id,
        case_number=c.case_number,
        transaction_id=c.transaction_id,
        evaluation_id=c.evaluation_id,
        status=c.status,
        priority=c.priority,
        trigger_source=c.trigger_source,
        assigned_to=c.assigned_to,
        assigned_at=c.assigned_at,
        opened_at=c.opened_at,
        resolved_at=c.resolved_at,
        disposition=c.disposition,
        account_id=tx.account_id if tx else "",
        amount=tx.amount if tx else Decimal("0.00"),
        currency=tx.currency if tx else "USD",
        merchant_category=tx.merchant_category if tx else "",
        risk_score=ev.risk_score if ev else 0,
        risk_tier=ev.risk_tier if ev else RiskTier.LOW,
        decision_action=ev.decision_action if ev else DecisionAction.APPROVE,
        model_score=ev.model_score if ev else Decimal("0.000000"),
    )


def _map_case_to_detail_response(c: Case, notes_limit: int = 100) -> CaseDetailResponse:
    """
    Map a deeply-loaded Case entity into the full investigation workspace response.
    """
    tx = c.transaction
    ev = c.evaluation

    tx_ctx = CaseTransactionContext(
        id=tx.id,
        external_transaction_id=tx.external_transaction_id,
        account_id=tx.account_id,
        merchant_id=tx.merchant_id,
        merchant_category=tx.merchant_category,
        job_category=tx.job_category,
        amount=tx.amount,
        currency=tx.currency,
        cardholder_lat=tx.cardholder_lat,
        cardholder_long=tx.cardholder_long,
        merchant_lat=tx.merchant_lat,
        merchant_long=tx.merchant_long,
        city_pop=tx.city_pop,
        transaction_timestamp=tx.transaction_timestamp,
        features_snapshot=tx.features_snapshot,
    )

    rule_matches = [
        RuleMatchResponse(
            rule_id=rm.rule_id,
            description=rm.description,
            feature_name=rm.feature_name,
            operator=rm.operator,
            comparison_value=rm.comparison_value,
            outcome=rm.outcome.value if hasattr(rm.outcome, "value") else str(rm.outcome),
            rule_type=rm.rule_type.value if hasattr(rm.rule_type, "value") else str(rm.rule_type),
            priority=rm.priority,
        )
        for rm in (ev.rule_matches or [])
    ] if ev else []

    reason_codes = [
        ReasonCodeResponse(
            code=rc.code,
            headline=rc.headline,
            description=rc.description,
            category=rc.category,
            source=rc.source.value if hasattr(rc.source, "value") else str(rc.source),
            severity=rc.severity.value if hasattr(rc.severity, "value") else str(rc.severity),
            rank=rc.rank,
        )
        for rc in (ev.reason_codes or [])
    ] if ev else []

    feature_attributions = [
        FeatureAttributionResponse(
            feature_name=fa.feature_name,
            display_name=fa.display_name,
            raw_value=fa.raw_value,
            shap_value=float(fa.shap_value),
            direction=fa.direction.value if hasattr(fa.direction, "value") else str(fa.direction),
            relative_contribution_pct=float(fa.relative_contribution_pct),
            rank=fa.rank,
        )
        for fa in (ev.feature_attributions or [])
    ] if ev else []

    ev_ctx = CaseEvaluationContext(
        id=ev.id,
        model_version=ev.model_version,
        policy_mode=ev.policy_mode,
        model_score=ev.model_score,
        risk_score=ev.risk_score,
        risk_tier=ev.risk_tier,
        decision_action=ev.decision_action,
        baseline_action=ev.baseline_action,
        is_overridden=ev.is_overridden,
        rule_action=ev.rule_action,
        decision_reason=ev.decision_reason,
        output_margin=ev.output_margin,
        base_value=ev.base_value,
        evaluated_at=ev.evaluated_at,
        rule_matches=rule_matches,
        reason_codes=reason_codes,
        feature_attributions=feature_attributions,
    ) if ev else None

    all_notes = c.notes or []
    notes_slice = all_notes[:notes_limit]
    notes_items = [
        CaseNoteItem(
            id=n.id,
            case_id=n.case_id,
            author_id=n.author_id,
            author_role=n.author_role,
            note_type=n.note_type,
            content=n.content,
            created_at=n.created_at,
            updated_at=n.updated_at,
        )
        for n in notes_slice
    ]

    return CaseDetailResponse(
        case=CaseResponse.model_validate(c),
        transaction=tx_ctx,
        evaluation=ev_ctx,
        notes=notes_items,
        notes_total=len(all_notes),
    )


# ==============================================================================
# Endpoint 1: Paginated Review Queue
# ==============================================================================

@router.get(
    "",
    response_model=CaseListResponse,
    summary="Paginated Review Queue",
    description="Retrieve a paginated collection of review queue cases with multi-dimensional filtering.",
)
async def list_cases(
    limit: int = Query(default=20, ge=1, le=100, description="Items per page (1-100)"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    status: Optional[CaseStatus] = Query(default=None, description="Filter by case lifecycle status"),
    priority: Optional[CasePriority] = Query(default=None, description="Filter by triage priority"),
    assigned_to: Optional[str] = Query(default=None, description="Filter by assigned analyst ID or 'unassigned'"),
    risk_tier: Optional[RiskTier] = Query(default=None, description="Filter by evaluation risk tier"),
    min_score: Optional[int] = Query(default=None, ge=0, le=100, description="Minimum risk score (0-100)"),
    max_score: Optional[int] = Query(default=None, ge=0, le=100, description="Maximum risk score (0-100)"),
    search_term: Optional[str] = Query(default=None, description="Search term for case number, account, or merchant"),
    start_date: Optional[datetime] = Query(default=None, description="Earliest case opened_at timestamp"),
    end_date: Optional[datetime] = Query(default=None, description="Latest case opened_at timestamp"),
    sort_by: str = Query(default="opened_at", pattern="^(opened_at|priority|risk_score)$", description="Sort field"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$", description="Sort direction"),
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN)),
    service: CaseService = Depends(get_case_service),
) -> CaseListResponse:
    try:
        cases, total = await service.get_queue_cases(
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
        items = [_map_case_to_queue_item(c) for c in cases]
        return CaseListResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query review queue: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 2: Queue Summary KPIs
# ==============================================================================

@router.get(
    "/summary",
    response_model=CaseSummaryResponse,
    summary="Queue Summary KPIs",
    description="Compute operational summary metrics across all review cases using UTC timestamps.",
)
async def get_case_summary(
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN)),
    service: CaseService = Depends(get_case_service),
) -> CaseSummaryResponse:
    try:
        metrics = await service.get_summary_metrics()
        return CaseSummaryResponse(
            total_open=metrics["total_open"],
            unassigned_count=metrics["unassigned_count"],
            in_review_count=metrics["in_review_count"],
            escalated_count=metrics["escalated_count"],
            resolved_today=metrics["resolved_today"],
            resolved_last_24h=metrics["resolved_last_24h"],
            critical_priority_count=metrics["critical_priority_count"],
        )
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute queue summary metrics: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 3: Case Investigation Workspace Detail
# ==============================================================================

@router.get(
    "/{case_id}",
    response_model=CaseDetailResponse,
    summary="Case Investigation Detail",
    description="Retrieve deep investigation context for a single case, including transaction, evaluation, reason codes, rule matches, and notes.",
)
async def get_case(
    case_id: uuid.UUID,
    notes_limit: int = Query(default=100, ge=1, le=500, description="Maximum chronological notes to include"),
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN, AuditActorType.API_CLIENT)),
    service: CaseService = Depends(get_case_service),
) -> CaseDetailResponse:
    try:
        case = await service.get_case_detail(case_id)
        return _map_case_to_detail_response(case, notes_limit=notes_limit)
    except PersistenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        ) from exc
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve case detail: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 4: Manual Case Escalation
# ==============================================================================

@router.post(
    "",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Manual Case Escalation",
    description="Manually escalate an evaluated financial transaction into a human review case.",
)
async def create_case(
    request: CreateCaseRequest,
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN)),
    service: CaseService = Depends(get_case_service),
) -> CaseResponse:
    cmd = CreateManualCaseCommand(
        transaction_id=request.transaction_id,
        initial_note=request.initial_note,
        actor=actor,
        priority=request.priority,
        evaluation_id=request.evaluation_id,
    )
    try:
        case = await service.create_manual_case(cmd)
        return CaseResponse.model_validate(case)
    except PersistenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PersistenceConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create manual case: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 5: Reviewer Assignment Mutation
# ==============================================================================

@router.patch(
    "/{case_id}/assignment",
    response_model=CaseResponse,
    summary="Case Assignment Mutation",
    description="Mutate case reviewer assignment (CLAIM, ASSIGN, UNASSIGN). Uses row-level locking.",
)
async def update_assignment(
    case_id: uuid.UUID,
    request: CaseAssignmentRequest,
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN)),
    service: CaseService = Depends(get_case_service),
) -> CaseResponse:
    try:
        case = await service.update_assignment(
            case_id=case_id,
            action=request.action,
            assignee_id=request.assignee_id,
            actor=actor,
            reason=request.reason,
        )
        return CaseResponse.model_validate(case)
    except PersistenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PersistenceConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update case assignment: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 6: Lifecycle Status Mutation
# ==============================================================================

@router.patch(
    "/{case_id}/status",
    response_model=CaseResponse,
    summary="Lifecycle Status Mutation",
    description="Mutate case lifecycle status (ESCALATE, CLOSE, REOPEN). Uses row-level locking.",
)
async def update_status(
    case_id: uuid.UUID,
    request: CaseStatusUpdateRequest,
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN)),
    service: CaseService = Depends(get_case_service),
) -> CaseResponse:
    try:
        case = await service.update_status(
            case_id=case_id,
            target_status=request.target_status,
            reason=request.reason,
            actor=actor,
        )
        return CaseResponse.model_validate(case)
    except PersistenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update case status: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 7: Chronological Notes List
# ==============================================================================

@router.get(
    "/{case_id}/notes",
    response_model=List[CaseNoteItem],
    summary="Case Notes List",
    description="Retrieve all chronological investigation notes for a case.",
)
async def list_case_notes(
    case_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum notes to return"),
    offset: int = Query(default=0, ge=0, description="Notes offset"),
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN)),
    service: CaseService = Depends(get_case_service),
) -> List[CaseNoteItem]:
    try:
        notes = await service.get_notes(case_id=case_id, limit=limit, offset=offset)
        return [CaseNoteItem.model_validate(n) for n in notes]
    except PersistenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve case notes: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 8: Append Investigation Note
# ==============================================================================

@router.post(
    "/{case_id}/notes",
    response_model=CaseNoteItem,
    status_code=status.HTTP_201_CREATED,
    summary="Append Investigation Note",
    description="Append an authoritative investigation note to a case with authenticated author identity.",
)
async def add_case_note(
    case_id: uuid.UUID,
    request: CreateCaseNoteRequest,
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN)),
    service: CaseService = Depends(get_case_service),
) -> CaseNoteItem:
    try:
        note = await service.add_note(
            case_id=case_id,
            content=request.content,
            note_type=request.note_type,
            actor=actor,
        )
        return CaseNoteItem.model_validate(note)
    except PersistenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add case note: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 9: Submit Human Disposition
# ==============================================================================

@router.post(
    "/{case_id}/disposition",
    response_model=CaseResponse,
    summary="Submit Human Disposition",
    description="Record authoritative human review outcome, resolving the case. Uses row-level locking.",
)
async def submit_disposition(
    case_id: uuid.UUID,
    request: CaseDispositionRequest,
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN)),
    service: CaseService = Depends(get_case_service),
) -> CaseResponse:
    try:
        case = await service.record_disposition(
            case_id=case_id,
            disposition=request.disposition,
            reason=request.reason,
            actor=actor,
        )
        return CaseResponse.model_validate(case)
    except PersistenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit case disposition: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 10: Chronological Audit Timeline History
# ==============================================================================

@router.get(
    "/{case_id}/timeline",
    response_model=CaseTimelineResponse,
    summary="Case Audit Timeline",
    description="Retrieve chronological audit history targeting AuditEntityType.CASE.",
)
async def get_case_timeline(
    case_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum timeline events to return"),
    actor: ActorContext = Depends(require_role(AuditActorType.ANALYST, AuditActorType.ADMIN, AuditActorType.API_CLIENT)),
    service: CaseService = Depends(get_case_service),
) -> CaseTimelineResponse:
    try:
        case, events = await service.get_timeline(case_id=case_id, limit=limit)
        items = [
            TimelineEventItem(
                id=e.id,
                event_type=e.event_type,
                action=e.action,
                actor_type=e.actor_type,
                actor_id=e.actor_id,
                correlation_id=e.correlation_id,
                payload=e.payload,
                event_timestamp=e.event_timestamp,
            )
            for e in events
        ]
        return CaseTimelineResponse(
            case_id=case.id,
            case_number=case.case_number,
            events=items,
            total_events=len(items),
        )
    except PersistenceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve case timeline: {exc}",
        ) from exc


__all__ = ["router"]
