"""
EvaluationRuleMatch ORM Model for Persisting Triggered Business Rules.

Records individual business rules evaluated and triggered during a risk evaluation.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING
import uuid

from sqlalchemy import String, SmallInteger, Text, DateTime, Enum, ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models.base import Base, UUIDPrimaryKeyMixin
from backend.app.db.models.enums import RuleOutcome, RuleType

if TYPE_CHECKING:
    from backend.app.db.models.risk_evaluation import RiskEvaluation


class EvaluationRuleMatch(Base, UUIDPrimaryKeyMixin):
    """
    SQLAlchemy 2.0 ORM model for triggered business rules.
    """
    __tablename__ = "evaluation_rule_matches"

    # Foreign Key Linkage
    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("risk_evaluations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Reference to the parent RiskEvaluation.",
    )

    # Rule Metadata
    rule_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Canonical rule identifier string (e.g. 'RULE_VELOCITY_BURST_REVIEW').",
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Human-readable explanation of rule intent.",
    )
    feature_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Exact predictive feature evaluated against the rule threshold.",
    )
    operator: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        doc="Comparison operator ('>', '<', 'in', 'is_true').",
    )
    comparison_value: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Configured threshold comparison value.",
    )
    outcome: Mapped[RuleOutcome] = mapped_column(
        Enum(RuleOutcome, native_enum=False, length=16),
        nullable=False,
        index=True,
        doc="Target outcome tier (BLOCK, REVIEW, MONITOR).",
    )
    rule_type: Mapped[RuleType] = mapped_column(
        Enum(RuleType, native_enum=False, length=32),
        nullable=False,
        doc="Domain category (VELOCITY, AMOUNT, GEOGRAPHY, COMPLIANCE, etc.).",
    )
    priority: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        doc="Evaluation priority integer (lower numbers evaluate earlier).",
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
        back_populates="rule_matches",
    )

    def __repr__(self) -> str:
        return (
            f"<EvaluationRuleMatch id={self.id} evaluation_id={self.evaluation_id} "
            f"rule_id={self.rule_id} outcome={self.outcome}>"
        )
