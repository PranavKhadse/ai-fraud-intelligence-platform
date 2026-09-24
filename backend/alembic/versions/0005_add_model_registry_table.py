"""Add model_registry_entries table for Phase 14 Model Lifecycle & Controlled Promotion.

Revision ID: 0005_add_model_registry_table
Revises: 0004_add_model_monitoring_snapshots_table
Create Date: 2026-09-24 18:00:00.000000+00:00

Defines ORM table supporting Phase 14 Model Registry and Lifecycle Governance:
- model_registry_entries: Stores versioned model bundle manifests, lifecycle state,
  active champion pointer, operating decision thresholds, checksums, and promotion audit trail.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '0005_add_model_registry_table'
down_revision: Union[str, None] = '0004_add_model_monitoring_snapshots_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'model_registry_entries',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('model_version', sa.String(length=32), nullable=False),
        sa.Column('model_family', sa.String(length=32), server_default='xgboost', nullable=False),
        sa.Column('status', sa.String(length=32), server_default='CANDIDATE', nullable=False),
        sa.Column('is_active_champion', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('operating_threshold', sa.Numeric(precision=5, scale=4), server_default='0.7800', nullable=False),
        sa.Column('bundle_directory', sa.String(length=255), nullable=True),
        sa.Column('model_artifact_path', sa.String(length=255), nullable=True),
        sa.Column('preprocessor_artifact_path', sa.String(length=255), nullable=True),
        sa.Column('manifest_path', sa.String(length=255), nullable=True),
        sa.Column('sha256_model', sa.String(length=64), nullable=False),
        sa.Column('sha256_preprocessor', sa.String(length=64), nullable=False),
        sa.Column('sha256_manifest', sa.String(length=64), nullable=True),
        sa.Column('validation_metrics', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('oot_metrics', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('hyperparameters', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('training_metadata', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('policy_configuration', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('promoted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('promoted_by', sa.String(length=64), nullable=True),
        sa.Column('promotion_rationale', sa.Text(), nullable=True),
        sa.Column('rolled_back_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rolled_back_by', sa.String(length=64), nullable=True),
        sa.Column('rollback_rationale', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('CANDIDATE', 'CHALLENGER', 'CHAMPION', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK')",
            name='chk_model_registry_status',
        ),
        sa.CheckConstraint(
            "operating_threshold > 0.0 AND operating_threshold < 1.0",
            name='chk_model_registry_threshold',
        ),
        sa.CheckConstraint(
            "model_family = 'xgboost'",
            name='chk_model_registry_family',
        ),
        sa.PrimaryKeyConstraint('id', name='pk_model_registry_entries'),
        sa.UniqueConstraint('model_version', name='uq_model_registry_version'),
    )
    op.create_index('ix_model_registry_entries_model_ver', 'model_registry_entries', ['model_version'], unique=True)
    op.create_index('ix_model_registry_entries_status', 'model_registry_entries', ['status'], unique=False)
    op.create_index('ix_model_registry_entries_is_active', 'model_registry_entries', ['is_active_champion'], unique=False)
    op.create_index('ix_model_registry_status_active', 'model_registry_entries', ['status', 'is_active_champion'], unique=False)


def downgrade() -> None:
    op.drop_table('model_registry_entries')
