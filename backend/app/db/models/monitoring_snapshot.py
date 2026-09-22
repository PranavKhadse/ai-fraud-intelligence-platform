"""
ModelMonitoringSnapshot ORM Model for Persisting Model & Risk Monitoring Rollups.

Represents persisted hourly or daily monitoring snapshots capturing:
- Overall and component health statuses (data drift, prediction drift, performance)
- Evaluated sample counts (total evaluated transactions and labeled ground-truth cases)
- Serialized feature drift, prediction drift, and performance summary JSON payloads
- Time window boundaries and model version provenance
"""

from datetime import datetime, timezone
from typing import Any, Dict
import uuid

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    JSON,
    CheckConstraint,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.models.base import Base, UUIDPrimaryKeyMixin


class ModelMonitoringSnapshot(Base, UUIDPrimaryKeyMixin):
    """
    SQLAlchemy 2.0 ORM model for persisted model monitoring snapshots.

    Design Choices:
    - UUID primary key (`id`).
    - Model version provenance (`model_version`).
    - Standardized window intervals (`window_type` in 'HOURLY', 'DAILY').
    - Timezone-aware UTC window boundaries (`window_start`, `window_end`).
    - Strict unique constraint on `(model_version, window_type, window_start, window_end)`
      enforces idempotent snapshot calculation and zero duplicates.
    - JSON/JSONB summary payloads storing complete diagnostic profiles.
    """
    __tablename__ = "model_monitoring_snapshots"

    # Model Version Provenance
    model_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="1.0.0",
        index=True,
        doc="Target model release version evaluated.",
    )

    # Window Type & Boundaries
    window_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        doc="Monitoring rollup interval ('HOURLY' or 'DAILY').",
    )
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Inclusive UTC start timestamp of the monitoring window.",
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Exclusive UTC end timestamp of the monitoring window.",
    )

    # Sample Counts
    sample_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total number of evaluated transaction inferences in the window.",
    )
    labeled_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total number of resolved ground-truth cases evaluated in the window.",
    )

    # Status Fields
    overall_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        doc="Precedence-resolved overall health status ('NORMAL', 'WARNING', 'CRITICAL', 'INSUFFICIENT_DATA').",
    )
    data_drift_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="Component health status for input feature drift.",
    )
    prediction_drift_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="Component health status for prediction and risk score distribution drift.",
    )
    performance_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="Component health status for ground-truth model and decision performance.",
    )

    # Diagnostic Summary Payloads
    feature_drift_summary: Mapped[Dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        doc="Serialized summary of feature drift metrics, PSI/KS rankings, and critical alerts.",
    )
    prediction_drift_summary: Mapped[Dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        doc="Serialized summary of model score PSI, 10-bucket risk score distributions, and tier shifts.",
    )
    performance_summary: Mapped[Dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        doc="Serialized summary of primary threshold metrics, operational metrics, and degradation alerts.",
    )

    # Audit Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="UTC timestamp when the monitoring snapshot record was persisted.",
    )

    __table_args__ = (
        UniqueConstraint(
            "model_version",
            "window_type",
            "window_start",
            "window_end",
            name="uq_monitoring_snapshots_version_window",
        ),
        CheckConstraint(
            "window_type IN ('HOURLY', 'DAILY')",
            name="chk_monitoring_snapshots_window_type",
        ),
        CheckConstraint(
            "sample_count >= 0",
            name="chk_monitoring_snapshots_sample_count_non_negative",
        ),
        CheckConstraint(
            "labeled_count >= 0",
            name="chk_monitoring_snapshots_labeled_count_non_negative",
        ),
        CheckConstraint(
            "window_end > window_start",
            name="chk_monitoring_snapshots_window_dates",
        ),
        Index(
            "ix_monitoring_snapshots_window_dates",
            "window_type",
            "window_start",
            "window_end",
        ),
        Index(
            "ix_monitoring_snapshots_overall_status",
            "overall_status",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelMonitoringSnapshot id={self.id} version={self.model_version} "
            f"type={self.window_type} range=[{self.window_start.isoformat()} -> {self.window_end.isoformat()}] "
            f"status={self.overall_status} samples={self.sample_count}>"
        )
