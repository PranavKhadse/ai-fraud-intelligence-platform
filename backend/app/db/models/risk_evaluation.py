"""
RiskEvaluation ORM Model for Persisting Risk Engine Decisions & Provenance.

Represents an individual evaluation performed by the Risk Engine and TreeSHAP explainer
against a Transaction.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, TYPE_CHECKING
import uuid

from sqlalchemy import (
    String,
    Numeric,
    SmallInteger,
    Boolean,
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
    DecisionAction,
    RiskTier,
    PolicyMode,
    RuleOutcome,
)

if TYPE_CHECKING:
    from backend.app.db.models.transaction import Transaction
    from backend.app.db.models.rule_match import EvaluationRuleMatch
    from backend.app.db.models.reason_code import EvaluationReasonCode
    from backend.app.db.models.feature_attribution import EvaluationFeatureAttribution
    from backend.app.db.models.case import Case


class RiskEvaluation(Base, UUIDPrimaryKeyMixin):
    """
    SQLAlchemy 2.0 ORM model for risk evaluation decisions.

    Captures:
    - Model probability and normalized risk scores.
    - Policy decision actions and rule override flags.
    - Model provenance version.
    - TreeSHAP base value and output margins.
    """
    __tablename__ = "risk_evaluations"

    # Foreign Key Linkage
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("transactions.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        doc="Reference to the evaluated Transaction.",
    )

    # Model Provenance & Policy Mode
    model_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="1.0.0",
        index=True,
        doc="Champion model version identifier active during evaluation.",
    )
    policy_mode: Mapped[PolicyMode] = mapped_column(
        Enum(PolicyMode, native_enum=False, length=32),
        nullable=False,
        default=PolicyMode.TRI_TIER,
        doc="Operating decision policy mode (TRI_TIER or BINARY_AUTO).",
    )

    # Risk Scoring Metrics
    model_score: Mapped[Decimal] = mapped_column(
        Numeric(8, 6),
        nullable=False,
        doc="Continuous model ranking probability in [0.0, 1.0].",
    )
    risk_score: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        doc="Calibrated integer risk score in [0, 100].",
    )
    risk_tier: Mapped[RiskTier] = mapped_column(
        Enum(RiskTier, native_enum=False, length=16),
        nullable=False,
        index=True,
        doc="Categorical risk tier (LOW, MEDIUM, HIGH, CRITICAL).",
    )

    # Decision Actions & Overrides
    decision_action: Mapped[DecisionAction] = mapped_column(
        Enum(DecisionAction, native_enum=False, length=16),
        nullable=False,
        index=True,
        doc="Final operational business decision (APPROVE, REVIEW, BLOCK).",
    )
    baseline_action: Mapped[DecisionAction] = mapped_column(
        Enum(DecisionAction, native_enum=False, length=16),
        nullable=False,
        doc="Baseline model policy action before any business rule overrides.",
    )
    is_overridden: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="True if a business rule escalated or overrode the baseline ML action.",
    )
    rule_action: Mapped[Optional[RuleOutcome]] = mapped_column(
        Enum(RuleOutcome, native_enum=False, length=16),
        nullable=True,
        doc="Highest precedence rule outcome enacting an override.",
    )
    decision_reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Human-readable decision boundary explanation.",
    )

    # TreeSHAP Margin Metrics
    output_margin: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6),
        nullable=True,
        doc="Raw model log-odds margin (logit of probability).",
    )
    base_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6),
        nullable=True,
        doc="Global TreeSHAP baseline expected margin.",
    )

    # Latency & Correlation Telemetry
    evaluation_latency_ms: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 2),
        nullable=True,
        doc="Evaluation execution latency in milliseconds.",
    )
    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="Distributed request correlation tracing ID.",
    )

    # Timestamps
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Timestamp when risk evaluation was executed.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="Database insertion timestamp.",
    )

    # Relationships
    transaction: Mapped["Transaction"] = relationship(
        "Transaction",
        back_populates="evaluations",
    )
    rule_matches: Mapped[List["EvaluationRuleMatch"]] = relationship(
        "EvaluationRuleMatch",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    reason_codes: Mapped[List["EvaluationReasonCode"]] = relationship(
        "EvaluationReasonCode",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    feature_attributions: Mapped[List["EvaluationFeatureAttribution"]] = relationship(
        "EvaluationFeatureAttribution",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    case: Mapped[Optional["Case"]] = relationship(
        "Case",
        back_populates="evaluation",
        uselist=False,
        cascade="save-update, merge",
        passive_deletes=True,
        doc="Human review case associated with this risk evaluation if opened.",
    )

    __table_args__ = (
        CheckConstraint("model_score >= 0.0 AND model_score <= 1.0", name="chk_risk_evaluations_model_score"),
        CheckConstraint("risk_score >= 0 AND risk_score <= 100", name="chk_risk_evaluations_risk_score"),
        Index("ix_risk_evaluations_action_tier_date", "decision_action", "risk_tier", "evaluated_at"),
        Index("ix_risk_evaluations_model_ver_date", "model_version", "evaluated_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<RiskEvaluation id={self.id} transaction_id={self.transaction_id} "
            f"action={self.decision_action} tier={self.risk_tier} score={self.risk_score} "
            f"model_ver={self.model_version}>"
        )
