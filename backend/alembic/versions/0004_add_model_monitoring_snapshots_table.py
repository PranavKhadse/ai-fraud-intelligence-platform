"""Add model_monitoring_snapshots table for Phase 13 ML & Model Monitoring.

Revision ID: 0004_add_model_monitoring_snapshots_table
Revises: 0003_add_cases_and_case_notes_tables
Create Date: 2026-09-22 18:30:00.000000+00:00

Defines ORM table supporting Phase 13 Model Monitoring Persistence & Snapshot Rollups:
- model_monitoring_snapshots: Persisted hourly and daily monitoring rollups capturing
  feature drift, prediction drift, and ground-truth performance diagnostics.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '0004_add_model_monitoring_snapshots_table'
down_revision: Union[str, None] = '0003_add_cases_and_case_notes_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'model_monitoring_snapshots',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('model_version', sa.String(length=32), server_default='1.0.0', nullable=False),
        sa.Column('window_type', sa.String(length=16), nullable=False),
        sa.Column('window_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('window_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sample_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('labeled_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('overall_status', sa.String(length=20), nullable=False),
        sa.Column('data_drift_status', sa.String(length=20), nullable=False),
        sa.Column('prediction_drift_status', sa.String(length=20), nullable=False),
        sa.Column('performance_status', sa.String(length=20), nullable=False),
        sa.Column('feature_drift_summary', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
        sa.Column('prediction_drift_summary', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
        sa.Column('performance_summary', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("window_type IN ('HOURLY', 'DAILY')", name='chk_monitoring_snapshots_window_type'),
        sa.CheckConstraint('sample_count >= 0', name='chk_monitoring_snapshots_sample_count_non_negative'),
        sa.CheckConstraint('labeled_count >= 0', name='chk_monitoring_snapshots_labeled_count_non_negative'),
        sa.CheckConstraint('window_end > window_start', name='chk_monitoring_snapshots_window_dates'),
        sa.PrimaryKeyConstraint('id', name='pk_model_monitoring_snapshots'),
        sa.UniqueConstraint('model_version', 'window_type', 'window_start', 'window_end', name='uq_monitoring_snapshots_version_window'),
    )
    op.create_index('ix_model_monitoring_snapshots_model_ver', 'model_monitoring_snapshots', ['model_version'], unique=False)
    op.create_index('ix_model_monitoring_snapshots_window_start', 'model_monitoring_snapshots', ['window_start'], unique=False)
    op.create_index('ix_model_monitoring_snapshots_window_end', 'model_monitoring_snapshots', ['window_end'], unique=False)
    op.create_index('ix_monitoring_snapshots_window_dates', 'model_monitoring_snapshots', ['window_type', 'window_start', 'window_end'], unique=False)
    op.create_index('ix_monitoring_snapshots_overall_status', 'model_monitoring_snapshots', ['overall_status'], unique=False)


def downgrade() -> None:
    op.drop_table('model_monitoring_snapshots')
