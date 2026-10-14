"""Add users, api_keys, and audit hash-chain columns for Phase 15.

Revision ID: 0006_auth_and_audit_hash_chain
Revises: 0005_add_model_registry_table
Create Date: 2026-10-14 14:30:00.000000+00:00

Defines two new ORM tables and audit log schema enhancements supporting Phase 15 Security & Authentication:
1. users: Enterprise user identity, credential hashes, roles, and status flags.
2. api_keys: Machine-to-machine API key credentials with key prefixes and SHA-256 hashed tokens.
3. audit_logs: Adds nullable prev_hash and entry_hash columns for backward-compatible cryptographic hash chaining.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0006_auth_and_audit_hash_chain'
down_revision: Union[str, None] = '0005_add_model_registry_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 0. Widen alembic_version.version_num on PostgreSQL if needed
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'alembic_version' AND column_name = 'version_num'
                ) THEN
                    ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(64);
                END IF;
            END $$;
            """
        )

    # 1. Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('email', sa.String(length=128), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=128), nullable=True),
        sa.Column(
            'role',
            sa.Enum('ANALYST', 'ADMIN', name='user_role', native_enum=False, length=32),
            server_default='ANALYST',
            nullable=False,
        ),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('is_superuser', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_users'),
        sa.UniqueConstraint('username', name='uq_users_username'),
        sa.UniqueConstraint('email', name='uq_users_email'),
        sa.CheckConstraint('LENGTH(TRIM(username)) >= 3', name='chk_users_username_min_length'),
        sa.CheckConstraint('LENGTH(TRIM(email)) >= 5', name='chk_users_email_min_length'),
        sa.CheckConstraint('LENGTH(TRIM(hashed_password)) > 0', name='chk_users_hashed_password_non_empty'),
        sa.CheckConstraint("role IN ('ANALYST', 'ADMIN')", name='chk_users_role_human'),
    )
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_role', 'users', ['role'], unique=False)
    op.create_index('ix_users_is_active', 'users', ['is_active'], unique=False)
    op.create_index('ix_users_role_active', 'users', ['role', 'is_active'], unique=False)

    # 2. Create api_keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('key_prefix', sa.String(length=16), nullable=False),
        sa.Column('hashed_key', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('client_id', sa.String(length=128), nullable=False),
        sa.Column(
            'role',
            sa.Enum('SYSTEM', 'ANALYST', 'ADMIN', 'API_CLIENT', name='apikey_auditactortype', native_enum=False, length=32),
            server_default='API_CLIENT',
            nullable=False,
        ),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_api_keys'),
        sa.UniqueConstraint('hashed_key', name='uq_api_keys_hashed_key'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_api_keys_user_id', ondelete='CASCADE'),
        sa.CheckConstraint('LENGTH(TRIM(key_prefix)) >= 3', name='chk_api_keys_prefix_min_length'),
        sa.CheckConstraint('LENGTH(TRIM(hashed_key)) > 0', name='chk_api_keys_hashed_key_non_empty'),
        sa.CheckConstraint('LENGTH(TRIM(client_id)) > 0', name='chk_api_keys_client_id_non_empty'),
        sa.CheckConstraint('LENGTH(TRIM(name)) > 0', name='chk_api_keys_name_non_empty'),
        sa.CheckConstraint("role IN ('API_CLIENT')", name='chk_api_keys_role_machine'),
    )
    op.create_index('ix_api_keys_key_prefix', 'api_keys', ['key_prefix'], unique=False)
    op.create_index('ix_api_keys_hashed_key', 'api_keys', ['hashed_key'], unique=True)
    op.create_index('ix_api_keys_client_id', 'api_keys', ['client_id'], unique=False)
    op.create_index('ix_api_keys_role', 'api_keys', ['role'], unique=False)
    op.create_index('ix_api_keys_is_active', 'api_keys', ['is_active'], unique=False)
    op.create_index('ix_api_keys_expires_at', 'api_keys', ['expires_at'], unique=False)
    op.create_index('ix_api_keys_user_id', 'api_keys', ['user_id'], unique=False)
    op.create_index('ix_api_keys_prefix_active', 'api_keys', ['key_prefix', 'is_active'], unique=False)

    # 3. Add cryptographic hash chain columns to audit_logs (nullable for backward compatibility)
    op.add_column('audit_logs', sa.Column('prev_hash', sa.String(length=64), nullable=True))
    op.add_column('audit_logs', sa.Column('entry_hash', sa.String(length=64), nullable=True))
    op.create_index('ix_audit_logs_entry_hash', 'audit_logs', ['entry_hash'], unique=False)

    # 4. Update auditentitytype check constraint if present on PostgreSQL
    if bind.dialect.name == 'postgresql':
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'auditentitytype'
                ) THEN
                    ALTER TABLE audit_logs DROP CONSTRAINT auditentitytype;
                    ALTER TABLE audit_logs ADD CONSTRAINT auditentitytype
                        CHECK (entity_type IN ('TRANSACTION', 'RISK_EVALUATION', 'POLICY', 'SYSTEM', 'CASE', 'USER', 'API_KEY', 'MODEL'));
                END IF;
            END $$;
            """
        )


def downgrade() -> None:
    bind = op.get_bind()

    # 1. Revert check constraint on PostgreSQL if needed
    if bind.dialect.name == 'postgresql':
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'auditentitytype'
                ) THEN
                    ALTER TABLE audit_logs DROP CONSTRAINT auditentitytype;
                    ALTER TABLE audit_logs ADD CONSTRAINT auditentitytype
                        CHECK (entity_type IN ('TRANSACTION', 'RISK_EVALUATION', 'POLICY', 'SYSTEM', 'CASE'));
                END IF;
            END $$;
            """
        )

    # 2. Remove audit_logs hash chaining columns
    op.drop_index('ix_audit_logs_entry_hash', table_name='audit_logs')
    op.drop_column('audit_logs', 'entry_hash')
    op.drop_column('audit_logs', 'prev_hash')

    # 3. Drop tables in reverse dependency order
    op.drop_table('api_keys')
    op.drop_table('users')
