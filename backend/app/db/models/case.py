"""
Case and CaseNote ORM Models for Human Review & Case Management.

Provides SQLAlchemy 2.0 ORM entities for tracking investigation cases,
analyst notes, review lifecycles, and human dispositions.
"""

from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING
import uuid

from sqlalchemy import (
    String,
    Text,
    DateTime,
    Enum,
    ForeignKey,
    CheckConstraint,
    Index,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models.base import Base, UUIDPrimaryKeyMixin
from backend.app.db.models.enums import (
    AuditActorType,
    CaseDisposition,
    CaseNoteType,
    CasePriority,
    CaseStatus,
    CaseTriggerSource,
)

if TYPE_CHECKING:
    from backend.app.db.models.risk_evaluation import RiskEvaluation
    from backend.app.db.models.transaction import Transaction


class Case(Base, UUIDPrimaryKeyMixin):
    """
    SQLAlchemy 2.0 ORM model for human review cases.

    Design Choices:
    - Exactly one case per Transaction (`transaction_id` is unique).
    - Links to the primary RiskEvaluation (`evaluation_id`).
    - Tracks complete lifecycle status (`OPEN`, `IN_REVIEW`, `ESCALATED`, `RESOLVED`, `CLOSED`).
    - Captures queue priority, trigger source, reviewer assignment, and human disposition.
    - Check constraint guarantees that disposition state requires resolved/closed status.
    """
    __tablename__ = "cases"

    # Human-readable Unique Case Identifier
    case_number: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        unique=True,
        index=True,
        doc="Human-friendly case identifier (e.g. 'CASE-20260918-A1B2C3').",
    )

    # Foreign Key Linkages
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("transactions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
        index=True,
        doc="Reference to the evaluated Transaction (1-to-1).",
    )
    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("risk_evaluations.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        doc="Reference to the primary RiskEvaluation.",
    )

    # Lifecycle State & Queue Management
    status: Mapped[CaseStatus] = mapped_column(
        Enum(CaseStatus, native_enum=False, length=20),
        nullable=False,
        default=CaseStatus.OPEN,
        index=True,
        doc="Current lifecycle status (OPEN, IN_REVIEW, ESCALATED, RESOLVED, CLOSED).",
    )
    priority: Mapped[CasePriority] = mapped_column(
        Enum(CasePriority, native_enum=False, length=16),
        nullable=False,
        default=CasePriority.MEDIUM,
        index=True,
        doc="Queue triage priority (CRITICAL, HIGH, MEDIUM, LOW).",
    )
    trigger_source: Mapped[CaseTriggerSource] = mapped_column(
        Enum(CaseTriggerSource, native_enum=False, length=32),
        nullable=False,
        default=CaseTriggerSource.AUTOMATED_REVIEW_POLICY,
        doc="Mechanism responsible for opening the case.",
    )

    # Reviewer Assignment
    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="Identifier of the assigned reviewer/analyst.",
    )
    assigned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="UTC timestamp when the case was assigned or claimed.",
    )

    # Lifecycle Timestamps
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
        doc="UTC timestamp when the case was opened.",
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="UTC timestamp when human disposition was recorded.",
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="UTC timestamp when the case was closed/archived.",
    )

    # Authoritative Human Disposition
    disposition: Mapped[Optional[CaseDisposition]] = mapped_column(
        Enum(CaseDisposition, native_enum=False, length=32),
        nullable=True,
        index=True,
        doc="Human review outcome (CONFIRMED_FRAUD, FALSE_POSITIVE, LEGITIMATE, SUSPICIOUS_RESOLVED).",
    )
    disposition_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Plain-English explanation of human disposition.",
    )
    dispositioned_by: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        doc="Analyst identifier who submitted the disposition.",
    )
    dispositioned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="UTC timestamp when disposition was submitted.",
    )

    # Standard Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="Database insertion timestamp.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="Database update timestamp.",
    )

    # Relationships
    transaction: Mapped["Transaction"] = relationship(
        "Transaction",
        back_populates="case",
    )
    evaluation: Mapped["RiskEvaluation"] = relationship(
        "RiskEvaluation",
        back_populates="case",
    )
    notes: Mapped[List["CaseNote"]] = relationship(
        "CaseNote",
        back_populates="case",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="CaseNote.created_at.asc()",
    )

    __table_args__ = (
        CheckConstraint(
            "(disposition IS NULL) OR (status IN ('RESOLVED', 'CLOSED') AND dispositioned_by IS NOT NULL)",
            name="chk_cases_disposition_state",
        ),
        Index("ix_cases_status_priority_created", "status", "priority", "created_at"),
        Index("ix_cases_assigned_status", "assigned_to", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<Case id={self.id} number={self.case_number} status={self.status} "
            f"priority={self.priority} assigned_to={self.assigned_to}>"
        )


class CaseNote(Base, UUIDPrimaryKeyMixin):
    """
    SQLAlchemy 2.0 ORM model for chronological case investigation notes.

    Design Choices:
    - Authoritative store for full investigation note text.
    - Append-only design with cascade delete on parent case deletion.
    - Check constraint guarantees non-empty trimmed text.
    """
    __tablename__ = "case_notes"

    # Foreign Key Linkage
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Reference to parent Case.",
    )

    # Author & Classification
    author_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        doc="Analyst identifier or system actor.",
    )
    author_role: Mapped[AuditActorType] = mapped_column(
        Enum(AuditActorType, native_enum=False, length=32),
        nullable=False,
        default=AuditActorType.ANALYST,
        doc="Actor role of the note author.",
    )
    note_type: Mapped[CaseNoteType] = mapped_column(
        Enum(CaseNoteType, native_enum=False, length=32),
        nullable=False,
        default=CaseNoteType.INVESTIGATION,
        doc="Categorization of note (INVESTIGATION, ESCALATION, DISPOSITION, SYSTEM_AUDIT).",
    )

    # Note Content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Authoritative note content.",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
        doc="UTC timestamp when note was created.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="UTC timestamp when note was updated.",
    )

    # Relationships
    case: Mapped["Case"] = relationship(
        "Case",
        back_populates="notes",
    )

    __table_args__ = (
        CheckConstraint("LENGTH(TRIM(content)) > 0", name="chk_case_notes_content_non_empty"),
        Index("ix_case_notes_case_created", "case_id", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<CaseNote id={self.id} case_id={self.case_id} author={self.author_id} "
            f"type={self.note_type} ts={self.created_at}>"
        )
