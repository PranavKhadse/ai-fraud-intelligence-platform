"""Initial core tables migration for PostgreSQL persistence.

Revision ID: 0001_initial_core_tables
Revises: None
Create Date: 2026-09-15 05:00:00.000000+00:00

Defines the six core ORM tables for the AI-Powered Fraud Detection & Risk Intelligence Platform:
1. transactions
2. risk_evaluations
3. evaluation_rule_matches
4. evaluation_reason_codes
5. evaluation_feature_attributions
6. audit_logs
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001_initial_core_tables'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create transactions table
    op.create_table(
        'transactions',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('external_transaction_id', sa.String(length=128), nullable=True),
        sa.Column('account_id', sa.String(length=128), nullable=False),
        sa.Column('merchant_id', sa.String(length=128), nullable=True),
        sa.Column('merchant_category', sa.String(length=64), nullable=False),
        sa.Column('job_category', sa.String(length=64), nullable=False),
        sa.Column('amount', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('currency', sa.String(length=3), server_default='USD', nullable=False),
        sa.Column('cardholder_lat', sa.Numeric(precision=9, scale=6), nullable=False),
        sa.Column('cardholder_long', sa.Numeric(precision=9, scale=6), nullable=False),
        sa.Column('merchant_lat', sa.Numeric(precision=9, scale=6), nullable=False),
        sa.Column('merchant_long', sa.Numeric(precision=9, scale=6), nullable=False),
        sa.Column('city_pop', sa.Integer(), nullable=False),
        sa.Column('transaction_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('features_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.CheckConstraint('amount >= 0.00', name='chk_transactions_amount_positive'),
        sa.PrimaryKeyConstraint('id', name='pk_transactions'),
    )
    op.create_index('ix_transactions_account_id', 'transactions', ['account_id'], unique=False)
    op.create_index('ix_transactions_external_transaction_id', 'transactions', ['external_transaction_id'], unique=False)
    op.create_index('ix_transactions_merchant_id', 'transactions', ['merchant_id'], unique=False)
    op.create_index('ix_transactions_merchant_category', 'transactions', ['merchant_category'], unique=False)
    op.create_index('ix_transactions_transaction_timestamp', 'transactions', ['transaction_timestamp'], unique=False)
    op.create_index('ix_transactions_account_ts', 'transactions', ['account_id', 'transaction_timestamp'], unique=False)

    # 2. Create risk_evaluations table
    op.create_table(
        'risk_evaluations',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('transaction_id', sa.Uuid(), nullable=False),
        sa.Column('model_version', sa.String(length=32), server_default='1.0.0', nullable=False),
        sa.Column('policy_mode', sa.Enum('TRI_TIER', 'BINARY_AUTO', name='policymode', native_enum=False, length=32), nullable=False),
        sa.Column('model_score', sa.Numeric(precision=8, scale=6), nullable=False),
        sa.Column('risk_score', sa.SmallInteger(), nullable=False),
        sa.Column('risk_tier', sa.Enum('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', name='risktier', native_enum=False, length=16), nullable=False),
        sa.Column('decision_action', sa.Enum('APPROVE', 'REVIEW', 'BLOCK', name='decisionaction', native_enum=False, length=16), nullable=False),
        sa.Column('baseline_action', sa.Enum('APPROVE', 'REVIEW', 'BLOCK', name='baseline_decisionaction', native_enum=False, length=16), nullable=False),
        sa.Column('is_overridden', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('rule_action', sa.Enum('BLOCK', 'REVIEW', 'MONITOR', name='ruleoutcome', native_enum=False, length=16), nullable=True),
        sa.Column('decision_reason', sa.Text(), nullable=False),
        sa.Column('output_margin', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('base_value', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('evaluation_latency_ms', sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column('correlation_id', sa.String(length=64), nullable=True),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('model_score >= 0.0 AND model_score <= 1.0', name='chk_risk_evaluations_model_score'),
        sa.CheckConstraint('risk_score >= 0 AND risk_score <= 100', name='chk_risk_evaluations_risk_score'),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.id'], name='fk_risk_evaluations_transaction_id', ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name='pk_risk_evaluations'),
    )
    op.create_index('ix_risk_evaluations_transaction_id', 'risk_evaluations', ['transaction_id'], unique=False)
    op.create_index('ix_risk_evaluations_model_version', 'risk_evaluations', ['model_version'], unique=False)
    op.create_index('ix_risk_evaluations_risk_tier', 'risk_evaluations', ['risk_tier'], unique=False)
    op.create_index('ix_risk_evaluations_decision_action', 'risk_evaluations', ['decision_action'], unique=False)
    op.create_index('ix_risk_evaluations_is_overridden', 'risk_evaluations', ['is_overridden'], unique=False)
    op.create_index('ix_risk_evaluations_correlation_id', 'risk_evaluations', ['correlation_id'], unique=False)
    op.create_index('ix_risk_evaluations_evaluated_at', 'risk_evaluations', ['evaluated_at'], unique=False)
    op.create_index('ix_risk_evaluations_action_tier_date', 'risk_evaluations', ['decision_action', 'risk_tier', 'evaluated_at'], unique=False)
    op.create_index('ix_risk_evaluations_model_ver_date', 'risk_evaluations', ['model_version', 'evaluated_at'], unique=False)

    # 3. Create evaluation_rule_matches table
    op.create_table(
        'evaluation_rule_matches',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('evaluation_id', sa.Uuid(), nullable=False),
        sa.Column('rule_id', sa.String(length=64), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('feature_name', sa.String(length=64), nullable=False),
        sa.Column('operator', sa.String(length=16), nullable=False),
        sa.Column('comparison_value', sa.String(length=64), nullable=False),
        sa.Column('outcome', sa.Enum('BLOCK', 'REVIEW', 'MONITOR', name='match_ruleoutcome', native_enum=False, length=16), nullable=False),
        sa.Column('rule_type', sa.Enum('VELOCITY', 'AMOUNT', 'GEOGRAPHY', 'COMPLIANCE', 'PROFILE', 'PATTERN', name='ruletype', native_enum=False, length=32), nullable=False),
        sa.Column('priority', sa.SmallInteger(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['evaluation_id'], ['risk_evaluations.id'], name='fk_eval_rule_matches_eval_id', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name='pk_evaluation_rule_matches'),
    )
    op.create_index('ix_evaluation_rule_matches_evaluation_id', 'evaluation_rule_matches', ['evaluation_id'], unique=False)
    op.create_index('ix_evaluation_rule_matches_rule_id', 'evaluation_rule_matches', ['rule_id'], unique=False)
    op.create_index('ix_evaluation_rule_matches_outcome', 'evaluation_rule_matches', ['outcome'], unique=False)

    # 4. Create evaluation_reason_codes table
    op.create_table(
        'evaluation_reason_codes',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('evaluation_id', sa.Uuid(), nullable=False),
        sa.Column('code', sa.String(length=64), nullable=False),
        sa.Column('headline', sa.String(length=128), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=32), nullable=False),
        sa.Column('source', sa.Enum('MODEL', 'RULE', name='reasonsource', native_enum=False, length=16), nullable=False),
        sa.Column('severity', sa.Enum('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO', name='reasonseverity', native_enum=False, length=16), nullable=False),
        sa.Column('rank', sa.SmallInteger(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['evaluation_id'], ['risk_evaluations.id'], name='fk_eval_reason_codes_eval_id', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name='pk_evaluation_reason_codes'),
    )
    op.create_index('ix_evaluation_reason_codes_evaluation_id', 'evaluation_reason_codes', ['evaluation_id'], unique=False)
    op.create_index('ix_evaluation_reason_codes_code', 'evaluation_reason_codes', ['code'], unique=False)

    # 5. Create evaluation_feature_attributions table
    op.create_table(
        'evaluation_feature_attributions',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('evaluation_id', sa.Uuid(), nullable=False),
        sa.Column('feature_name', sa.String(length=64), nullable=False),
        sa.Column('display_name', sa.String(length=128), nullable=False),
        sa.Column('raw_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('shap_value', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('direction', sa.Enum('RISK_INCREASING', 'MITIGATING', name='attributiondirection', native_enum=False, length=16), nullable=False),
        sa.Column('relative_contribution_pct', sa.Numeric(precision=6, scale=4), nullable=False),
        sa.Column('rank', sa.SmallInteger(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['evaluation_id'], ['risk_evaluations.id'], name='fk_eval_feature_attrib_eval_id', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name='pk_evaluation_feature_attributions'),
    )
    op.create_index('ix_evaluation_feature_attributions_evaluation_id', 'evaluation_feature_attributions', ['evaluation_id'], unique=False)
    op.create_index('ix_evaluation_feature_attributions_feature_name', 'evaluation_feature_attributions', ['feature_name'], unique=False)

    # 6. Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('entity_type', sa.Enum('TRANSACTION', 'RISK_EVALUATION', 'POLICY', 'SYSTEM', name='auditentitytype', native_enum=False, length=32), nullable=False),
        sa.Column('entity_id', sa.Uuid(), nullable=True),
        sa.Column('action', sa.String(length=64), nullable=False),
        sa.Column('actor_type', sa.Enum('SYSTEM', 'ANALYST', 'ADMIN', 'API_CLIENT', name='auditactortype', native_enum=False, length=32), server_default='SYSTEM', nullable=False),
        sa.Column('actor_id', sa.String(length=128), nullable=True),
        sa.Column('correlation_id', sa.String(length=64), nullable=True),
        sa.Column('client_ip', sa.String(length=45), nullable=True),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('event_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_audit_logs'),
    )
    op.create_index('ix_audit_logs_event_type', 'audit_logs', ['event_type'], unique=False)
    op.create_index('ix_audit_logs_entity_type', 'audit_logs', ['entity_type'], unique=False)
    op.create_index('ix_audit_logs_entity_id', 'audit_logs', ['entity_id'], unique=False)
    op.create_index('ix_audit_logs_correlation_id', 'audit_logs', ['correlation_id'], unique=False)
    op.create_index('ix_audit_logs_event_timestamp', 'audit_logs', ['event_timestamp'], unique=False)
    op.create_index('ix_audit_logs_entity_lookup', 'audit_logs', ['entity_type', 'entity_id', 'event_timestamp'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse topological order
    op.drop_table('audit_logs')
    op.drop_table('evaluation_feature_attributions')
    op.drop_table('evaluation_reason_codes')
    op.drop_table('evaluation_rule_matches')
    op.drop_table('risk_evaluations')
    op.drop_table('transactions')
