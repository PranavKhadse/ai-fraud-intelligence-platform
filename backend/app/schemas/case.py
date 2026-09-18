"""
Pydantic Schemas for Case Management & Human Review Lifecycle REST API.

Provides strongly-typed request and response DTOs for review queues, case details,
reviewer assignments, lifecycle transitions, notes, human dispositions, and audit timelines.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    RuleOutcome,
)
from backend.app.schemas.predict import (
    FeatureAttributionResponse,
    ReasonCodeResponse,
    RuleMatchResponse,
)


# ==============================================================================
# Request DTOs
# ==============================================================================

class CreateCaseRequest(BaseModel):
    """
    Request body for analyst-initiated manual case escalation.
    """
    transaction_id: uuid.UUID = Field(
        ...,
        description="Internal UUID of the financial transaction to escalate.",
    )
    initial_note: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Initial case explanation or investigation rationale.",
    )
    priority: Optional[CasePriority] = Field(
        default=None,
        description="Optional triage priority override (CRITICAL, HIGH, MEDIUM, LOW).",
    )
    evaluation_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional specific RiskEvaluation UUID belonging to the transaction.",
    )

    @field_validator("initial_note")
    @classmethod
    def validate_initial_note_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Initial case note cannot be empty or whitespace-only.")
        return v.strip()


class CaseAssignmentRequest(BaseModel):
    """
    Request body for reviewer assignment mutations (CLAIM, ASSIGN, UNASSIGN).
    """
    action: str = Field(
        ...,
        description="Assignment action to perform: 'CLAIM', 'ASSIGN', or 'UNASSIGN'.",
    )
    assignee_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Target analyst ID required when action is 'ASSIGN'.",
    )
    reason: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional operational justification for the assignment change.",
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        normalized = v.strip().upper()
        if normalized not in ("CLAIM", "ASSIGN", "UNASSIGN"):
            raise ValueError("Action must be one of: 'CLAIM', 'ASSIGN', 'UNASSIGN'.")
        return normalized

    @field_validator("assignee_id")
    @classmethod
    def validate_assignee_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_clean = v.strip()
            return v_clean if v_clean else None
        return None


class CaseStatusUpdateRequest(BaseModel):
    """
    Request body for lifecycle status transitions (ESCALATE, CLOSE, REOPEN).
    """
    target_status: CaseStatus = Field(
        ...,
        description="Target CaseStatus (ESCALATED, CLOSED, IN_REVIEW).",
    )
    reason: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Mandatory justification for lifecycle status change (min 10 characters).",
    )

    @field_validator("reason")
    @classmethod
    def validate_reason_non_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 10:
            raise ValueError("Status update reason must be at least 10 non-whitespace characters.")
        return v.strip()


class CreateCaseNoteRequest(BaseModel):
    """
    Request body for appending an investigation note to a case.
    """
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Authoritative investigation note content.",
    )
    note_type: CaseNoteType = Field(
        default=CaseNoteType.INVESTIGATION,
        description="Categorization of the note (INVESTIGATION, ESCALATION, DISPOSITION, SYSTEM_AUDIT).",
    )

    @field_validator("content")
    @classmethod
    def validate_content_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Note content cannot be empty or whitespace-only.")
        return v.strip()


class CaseDispositionRequest(BaseModel):
    """
    Request body for submitting the authoritative human review outcome.
    """
    disposition: CaseDisposition = Field(
        ...,
        description="Final human review outcome (CONFIRMED_FRAUD, FALSE_POSITIVE, LEGITIMATE, SUSPICIOUS_RESOLVED).",
    )
    reason: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Mandatory plain-English explanation for the human disposition (min 10 characters).",
    )

    @field_validator("reason")
    @classmethod
    def validate_reason_non_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 10:
            raise ValueError("Disposition reason must be at least 10 non-whitespace characters.")
        return v.strip()


# ==============================================================================
# Response DTOs
# ==============================================================================

class CaseNoteItem(BaseModel):
    """
    Individual chronological case note response item.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    case_id: uuid.UUID
    author_id: str
    author_role: AuditActorType
    note_type: CaseNoteType
    content: str
    created_at: datetime
    updated_at: datetime


class CaseResponse(BaseModel):
    """
    Core Case entity representation.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    case_number: str
    transaction_id: uuid.UUID
    evaluation_id: uuid.UUID
    status: CaseStatus
    priority: CasePriority
    trigger_source: CaseTriggerSource
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    opened_at: datetime
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    disposition: Optional[CaseDisposition] = None
    disposition_reason: Optional[str] = None
    dispositioned_by: Optional[str] = None
    dispositioned_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class CaseQueueItem(BaseModel):
    """
    Compact projection for review queue listing.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    case_number: str
    transaction_id: uuid.UUID
    evaluation_id: uuid.UUID
    status: CaseStatus
    priority: CasePriority
    trigger_source: CaseTriggerSource
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    opened_at: datetime
    resolved_at: Optional[datetime] = None
    disposition: Optional[CaseDisposition] = None

    # Compact Transaction Attributes
    account_id: str
    amount: Decimal
    currency: str
    merchant_category: str

    # Compact Evaluation Attributes
    risk_score: int
    risk_tier: RiskTier
    decision_action: DecisionAction
    model_score: Decimal


class CaseListResponse(BaseModel):
    """
    Paginated review queue response envelope.
    """
    items: List[CaseQueueItem]
    total: int
    limit: int
    offset: int


class CaseSummaryResponse(BaseModel):
    """
    Aggregated operational summary metrics for the review queue.
    """
    total_open: int = Field(description="Total in-flight cases (OPEN, IN_REVIEW, ESCALATED).")
    unassigned_count: int = Field(description="Total OPEN cases awaiting reviewer assignment.")
    in_review_count: int = Field(description="Total cases currently being investigated.")
    escalated_count: int = Field(description="Total cases escalated for senior review.")
    resolved_today: int = Field(description="Total cases resolved during current UTC day.")
    resolved_last_24h: int = Field(description="Total cases resolved in rolling past 24 hours.")
    critical_priority_count: int = Field(description="Total open/active cases with CRITICAL priority.")


class CaseTransactionContext(BaseModel):
    """
    Detailed transaction context for the investigation workspace.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    external_transaction_id: Optional[str] = None
    account_id: str
    merchant_id: Optional[str] = None
    merchant_category: str
    job_category: str
    amount: Decimal
    currency: str
    cardholder_lat: Decimal
    cardholder_long: Decimal
    merchant_lat: Decimal
    merchant_long: Decimal
    city_pop: Optional[int] = None
    transaction_timestamp: datetime
    features_snapshot: Optional[Dict[str, Any]] = None


class CaseEvaluationContext(BaseModel):
    """
    Detailed risk evaluation context for the investigation workspace.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    model_version: str
    policy_mode: PolicyMode
    model_score: Decimal
    risk_score: int
    risk_tier: RiskTier
    decision_action: DecisionAction
    baseline_action: DecisionAction
    is_overridden: bool
    rule_action: Optional[RuleOutcome] = None
    decision_reason: str
    output_margin: Optional[Decimal] = None
    base_value: Optional[Decimal] = None
    evaluated_at: datetime
    rule_matches: List[RuleMatchResponse] = []
    reason_codes: List[ReasonCodeResponse] = []
    feature_attributions: List[FeatureAttributionResponse] = []


class CaseDetailResponse(BaseModel):
    """
    Full investigation workspace response for a single case.
    """
    case: CaseResponse
    transaction: CaseTransactionContext
    evaluation: CaseEvaluationContext
    notes: List[CaseNoteItem]
    notes_total: int


class TimelineEventItem(BaseModel):
    """
    Individual chronological audit timeline entry for a case.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    event_type: str
    action: str
    actor_type: AuditActorType
    actor_id: Optional[str] = None
    correlation_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    event_timestamp: datetime


class CaseTimelineResponse(BaseModel):
    """
    Chronological audit timeline history for a case.
    """
    case_id: uuid.UUID
    case_number: str
    events: List[TimelineEventItem]
    total_events: int


__all__ = [
    "CreateCaseRequest",
    "CaseAssignmentRequest",
    "CaseStatusUpdateRequest",
    "CreateCaseNoteRequest",
    "CaseDispositionRequest",
    "CaseNoteItem",
    "CaseResponse",
    "CaseQueueItem",
    "CaseListResponse",
    "CaseSummaryResponse",
    "CaseTransactionContext",
    "CaseEvaluationContext",
    "CaseDetailResponse",
    "TimelineEventItem",
    "CaseTimelineResponse",
]
