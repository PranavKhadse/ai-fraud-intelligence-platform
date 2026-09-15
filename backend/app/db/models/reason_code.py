"""
EvaluationReasonCode ORM Model for Persisting Plain-English Reason Codes.

Records standardized human-readable reason codes synthesized from TreeSHAP drivers
and business rule triggers.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING
import uuid

from sqlalchemy import String, SmallInteger, Text, DateTime, Enum, ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models.base import Base, UUIDPrimaryKeyMixin
from backend.app.db.models.enums import ReasonSource, ReasonSeverity

if TYPE_CHECKING:
    from backend.app.db.models.risk_evaluation import RiskEvaluation


class EvaluationReasonCode(Base, UUIDPrimaryKeyMixin):
    """
    SQLAlchemy 2.0 ORM model for reason codes associated with a risk evaluation.
    """
    __tablename__ = "evaluation_reason_codes"

    # Foreign Key Linkage
    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("risk_evaluations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Reference to the parent RiskEvaluation.",
    )

    # Reason Code Attributes
    code: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Standard uppercase reason identifier (e.g. 'VELOCITY_BURST_1H').",
    )
    headline: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Concise summary headline for analyst review.",
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Interpolated plain-English explanation of the risk driver.",
    )
    category: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Domain category (AMOUNT, VELOCITY, GEOGRAPHY, ACCOUNT_HISTORY, etc.).",
    )
    source: Mapped[ReasonSource] = mapped_column(
        Enum(ReasonSource, native_enum=False, length=16),
        nullable=False,
        doc="Origin of reason code ('MODEL' for TreeSHAP or 'RULE' for rules).",
    )
    severity: Mapped[ReasonSeverity] = mapped_column(
        Enum(ReasonSeverity, native_enum=False, length=16),
        nullable=False,
        doc="Operational severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO).",
    )
    rank: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        doc="1-indexed presentation priority rank.",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="Database insertion timestamp.",
    )

    # Relationships
    evaluation: Mapped["RiskEvaluation"] = relationship(
        "RiskEvaluation",
        back_populates="reason_codes",
    )

    def __repr__(self) -> str:
        return (
            f"<EvaluationReasonCode id={self.id} evaluation_id={self.evaluation_id} "
            f"code={self.code} source={self.source} rank={self.rank}>"
        )
