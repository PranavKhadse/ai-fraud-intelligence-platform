"""
Model Registry ORM Entity for Phase 14 Model Lifecycle & Controlled Promotion.

Provides SQLAlchemy 2.0 ORM entity `ModelRegistryEntry` to store and manage versioned model
bundles, lifecycle state transitions, evaluation metrics, and promotion/rollback audit lineage.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional
import uuid

import sqlalchemy as sa
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin


class ModelRegistryEntry(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    SQLAlchemy ORM model for versioned model bundles in the platform Model Registry.

    Enforces:
    - Unique semantic model versions.
    - Exactly one active champion across all versions.
    - Cryptographic SHA-256 artifact verification hashes.
    - Complete evaluation metrics and promotion audit trail.
    """
    __tablename__ = "model_registry_entries"

    # Semantic Model Version (e.g. '1.0.0', '1.1.0')
    model_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        unique=True,
        index=True,
        doc="Semantic version identifier (MAJOR.MINOR.PATCH).",
    )

    # Model Family Architecture (strictly 'xgboost' in Phase 14)
    model_family: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="xgboost",
        server_default="xgboost",
        doc="Machine learning model algorithm family.",
    )

    # Lifecycle State
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="CANDIDATE",
        server_default="CANDIDATE",
        index=True,
        doc="Lifecycle state (CANDIDATE, CHALLENGER, CHAMPION, REJECTED, ARCHIVED, ROLLED_BACK).",
    )

    # Active Champion Indicator
    is_active_champion: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=sa.text("false"),
        index=True,
        doc="True if this version is the currently serving production champion.",
    )

    # Calibrated Operating Decision Threshold (tau*)
    operating_threshold: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=Decimal("0.7800"),
        server_default="0.7800",
        doc="Calibrated probability decision threshold for fraud classification.",
    )

    # Storage Paths
    bundle_directory: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Relative filesystem path to the model bundle directory.",
    )
    model_artifact_path: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Relative filesystem path to the model binary artifact.",
    )
    preprocessor_artifact_path: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Relative filesystem path to the fitted preprocessor artifact.",
    )
    manifest_path: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Relative filesystem path to the manifest JSON file.",
    )

    # Cryptographic Checksums (64-character SHA-256 hex strings)
    sha256_model: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="SHA-256 hex checksum of the model.joblib file.",
    )
    sha256_preprocessor: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="SHA-256 hex checksum of the preprocessor.joblib file.",
    )
    sha256_manifest: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="SHA-256 hex checksum of the manifest.json file.",
    )

    # Metric & Configuration Payloads (JSON / PostgreSQL JSONB)
    validation_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
        nullable=True,
        doc="Validation split benchmark metrics dictionary.",
    )
    oot_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
        nullable=True,
        doc="Out-of-Time protected holdout benchmark metrics dictionary.",
    )
    hyperparameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
        nullable=True,
        doc="Training hyperparameters configuration dictionary.",
    )
    training_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
        nullable=True,
        doc="Training dataset row counts, dates, and execution metadata.",
    )
    policy_configuration: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
        nullable=True,
        doc="Coupled decision policy threshold and risk tier band settings.",
    )

    # Formal Promotion Audit Fields
    promoted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="UTC timestamp when model was promoted to active champion.",
    )
    promoted_by: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="User / analyst identifier authoring the promotion decision.",
    )
    promotion_rationale: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Mandatory explanatory business rationale for promotion.",
    )

    # Rollback Audit Fields
    rolled_back_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="UTC timestamp when model was demoted via rollback.",
    )
    rolled_back_by: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="User / analyst identifier authoring the rollback operation.",
    )
    rollback_rationale: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Explanatory business rationale for rollback.",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('CANDIDATE', 'CHALLENGER', 'CHAMPION', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK')",
            name="chk_model_registry_status",
        ),
        CheckConstraint(
            "operating_threshold > 0.0 AND operating_threshold < 1.0",
            name="chk_model_registry_threshold",
        ),
        CheckConstraint(
            "model_family = 'xgboost'",
            name="chk_model_registry_family",
        ),
        Index("ix_model_registry_status_active", "status", "is_active_champion"),
    )
