"""
EvaluationFeatureAttribution ORM Model for Persisting Local TreeSHAP Attributions.

Records individual predictive feature contributions (SHAP values) explaining
a specific RiskEvaluation decision.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional, TYPE_CHECKING
import uuid

from sqlalchemy import String, Numeric, SmallInteger, DateTime, Enum, ForeignKey, JSON, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models.base import Base, UUIDPrimaryKeyMixin
from backend.app.db.models.enums import AttributionDirection

if TYPE_CHECKING:
    from backend.app.db.models.risk_evaluation import RiskEvaluation


class EvaluationFeatureAttribution(Base, UUIDPrimaryKeyMixin):
    """
    SQLAlchemy 2.0 ORM model for individual feature attributions associated with a risk evaluation.
    """
    __tablename__ = "evaluation_feature_attributions"

    # Foreign Key Linkage
    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("risk_evaluations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Reference to the parent RiskEvaluation.",
    )

    # Feature Attribution Details
    feature_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Exact predictive feature name matching model input contract.",
    )
    display_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Human-friendly feature label for UI and reports.",
    )
    raw_value: Mapped[Optional[Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
        doc="Unencoded raw value of the feature from the transaction payload.",
    )
    shap_value: Mapped[Decimal] = mapped_column(
        Numeric(10, 6),
        nullable=False,
        doc="Additive TreeSHAP attribution value in log-odds margin space.",
    )
    direction: Mapped[AttributionDirection] = mapped_column(
        Enum(AttributionDirection, native_enum=False, length=16),
        nullable=False,
        doc="Direction of risk influence ('RISK_INCREASING' or 'MITIGATING').",
    )
    relative_contribution_pct: Mapped[Decimal] = mapped_column(
        Numeric(6, 4),
        nullable=False,
        doc="Percentage contribution within its directional group.",
    )
    rank: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        doc="1-indexed importance ranking within its directional group.",
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
        back_populates="feature_attributions",
    )

    def __repr__(self) -> str:
        return (
            f"<EvaluationFeatureAttribution id={self.id} evaluation_id={self.evaluation_id} "
            f"feature={self.feature_name} shap={self.shap_value} direction={self.direction}>"
        )
