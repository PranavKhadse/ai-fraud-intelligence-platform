"""Add cases and case_notes tables for human review and case management.

Revision ID: 0003_add_cases_and_case_notes_tables
Revises: 0002_add_unique_index_external_tx_id
Create Date: 2026-09-18 10:00:00.000000+00:00

Defines two ORM tables supporting Phase 12 Human Review & Case Management:
1. cases: Human review cases tracking lifecycle state, assignment, and dispositions.
2. case_notes: Authoritative chronological investigation notes linked to cases.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003_add_cases_and_case_notes_tables'
down_revision: Union[str, None] = '0002_add_unique_index_external_tx_id'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create cases table
    op.create_table(
        'cases',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('case_number', sa.String(length=32), nullable=False),
        sa.Column('transaction_id', sa.Uuid(), nullable=False),
        sa.Column('evaluation_id', sa.Uuid(), nullable=False),
        sa.Column(
            'status',
            sa.Enum('OPEN', 'IN_REVIEW', 'ESCALATED', 'RESOLVED', 'CLOSED', name='casestatus', native_enum=False, length=20),
            server_default='OPEN',
            nullable=False,
        ),
        sa.Column(
            'priority',
            sa.Enum('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', name='casepriority', native_enum=False, length=16),
            server_default='MEDIUM',
            nullable=False,
        ),
        sa.Column(
            'trigger_source',
            sa.Enum('AUTOMATED_REVIEW_POLICY', 'AUTOMATED_RULE_OVERRIDE', 'MANUAL_ANALYST_ESCALATION', name='casetriggersource', native_enum=False, length=32),
            server_default='AUTOMATED_REVIEW_POLICY',
            nullable=False,
        ),
        sa.Column('assigned_to', sa.String(length=128), nullable=True),
        sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('opened_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'disposition',
            sa.Enum('CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'LEGITIMATE', 'SUSPICIOUS_RESOLVED', name='casedisposition', native_enum=False, length=32),
            nullable=True,
        ),
        sa.Column('disposition_reason', sa.Text(), nullable=True),
        sa.Column('dispositioned_by', sa.String(length=128), nullable=True),
        sa.Column('dispositioned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(disposition IS NULL) OR (status IN ('RESOLVED', 'CLOSED') AND dispositioned_by IS NOT NULL)",
            name='chk_cases_disposition_state',
        ),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.id'], name='fk_cases_transaction_id', ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['evaluation_id'], ['risk_evaluations.id'], name='fk_cases_evaluation_id', ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name='pk_cases'),
        sa.UniqueConstraint('case_number', name='uq_cases_case_number'),
        sa.UniqueConstraint('transaction_id', name='uq_cases_transaction_id'),
    )
    op.create_index('ix_cases_case_number', 'cases', ['case_number'], unique=True)
    op.create_index('ix_cases_transaction_id', 'cases', ['transaction_id'], unique=True)
    op.create_index('ix_cases_evaluation_id', 'cases', ['evaluation_id'], unique=False)
    op.create_index('ix_cases_status', 'cases', ['status'], unique=False)
    op.create_index('ix_cases_priority', 'cases', ['priority'], unique=False)
    op.create_index('ix_cases_assigned_to', 'cases', ['assigned_to'], unique=False)
    op.create_index('ix_cases_opened_at', 'cases', ['opened_at'], unique=False)
    op.create_index('ix_cases_disposition', 'cases', ['disposition'], unique=False)
    op.create_index('ix_cases_status_priority_created', 'cases', ['status', 'priority', 'created_at'], unique=False)
    op.create_index('ix_cases_assigned_status', 'cases', ['assigned_to', 'status'], unique=False)

    # 2. Create case_notes table
    op.create_table(
        'case_notes',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('case_id', sa.Uuid(), nullable=False),
        sa.Column('author_id', sa.String(length=128), nullable=False),
        sa.Column(
            'author_role',
            sa.Enum('SYSTEM', 'ANALYST', 'ADMIN', 'API_CLIENT', name='note_auditactortype', native_enum=False, length=32),
            server_default='ANALYST',
            nullable=False,
        ),
        sa.Column(
            'note_type',
            sa.Enum('INVESTIGATION', 'ESCALATION', 'DISPOSITION', 'SYSTEM_AUDIT', name='casenotetype', native_enum=False, length=32),
            server_default='INVESTIGATION',
            nullable=False,
        ),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('LENGTH(TRIM(content)) > 0', name='chk_case_notes_content_non_empty'),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id'], name='fk_case_notes_case_id', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name='pk_case_notes'),
    )
    op.create_index('ix_case_notes_case_id', 'case_notes', ['case_id'], unique=False)
    op.create_index('ix_case_notes_author_id', 'case_notes', ['author_id'], unique=False)
    op.create_index('ix_case_notes_created_at', 'case_notes', ['created_at'], unique=False)
    op.create_index('ix_case_notes_case_created', 'case_notes', ['case_id', 'created_at'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse topological order
    op.drop_table('case_notes')
    op.drop_table('cases')
