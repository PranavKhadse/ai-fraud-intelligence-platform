"""
AuditLog ORM Model for Append-Oriented Decision & System Event Tracking.

Provides an immutable-by-design audit foundation capturing operational events,
system decisions, and API interactions.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid

from sqlalchemy import String, DateTime, Enum, JSON, Index, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.models.base import Base, UUIDPrimaryKeyMixin
from backend.app.db.models.enums import AuditActorType, AuditEntityType


class AuditLog(Base, UUIDPrimaryKeyMixin):
    """
    SQLAlchemy 2.0 ORM model for append-oriented audit log entries.

    Design Choices:
    - Independent table without direct foreign key dependencies to avoid blocking historical queries.
    - Polymorphic entity tracking via `entity_type` and `entity_id`.
    - `payload` stores structured context without leaking secrets or credentials.
    - Optimized indexes for entity history lookup and chronological log scanning.
    """
    __tablename__ = "audit_logs"

    # Event Categorization
    event_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Standard event identifier (e.g. 'PREDICTION_EXECUTED', 'TRANSACTION_INGESTED').",
    )
    entity_type: Mapped[AuditEntityType] = mapped_column(
        Enum(AuditEntityType, native_enum=False, length=32),
        nullable=False,
        index=True,
        doc="Target entity type being audited (TRANSACTION, RISK_EVALUATION, POLICY, SYSTEM).",
    )
    entity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
        index=True,
        doc="Target entity primary key identifier if applicable.",
    )
    action: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Action executed on the entity (e.g. 'CREATE', 'EVALUATE', 'OVERRIDE').",
    )

    # Actor & Origin Telemetry
    actor_type: Mapped[AuditActorType] = mapped_column(
        Enum(AuditActorType, native_enum=False, length=32),
        nullable=False,
        default=AuditActorType.SYSTEM,
        doc="Category of the originating actor (SYSTEM, ANALYST, ADMIN, API_CLIENT).",
    )
    actor_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        doc="Identifier of the executing actor or API client key.",
    )
    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="Distributed request correlation ID for end-to-end tracing.",
    )
    client_ip: Mapped[Optional[str]] = mapped_column(
        String(45),
        nullable=True,
        doc="Originating client IP address (IPv4/IPv6).",
    )

    # Event Structured Data
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
        doc="Contextual metadata payload (sanitized of sensitive secrets/credentials).",
    )

    # Timestamps
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
        doc="Timestamp when the event occurred.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="Database insertion timestamp.",
    )

    __table_args__ = (
        Index("ix_audit_logs_entity_lookup", "entity_type", "entity_id", "event_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} event={self.event_type} entity={self.entity_type}:{self.entity_id} "
            f"actor={self.actor_type}:{self.actor_id} ts={self.event_timestamp}>"
        )
