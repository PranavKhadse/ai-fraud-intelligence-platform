"""Add partial unique index on external_transaction_id.

Revision ID: 0002_add_unique_index_external_tx_id
Revises: 0001_initial_core_tables
Create Date: 2026-09-15 10:00:00.000000+00:00

Creates partial unique index uq_transactions_external_tx_id on transactions(external_transaction_id)
WHERE external_transaction_id IS NOT NULL to guarantee PostgreSQL-backed concurrency safety and idempotency.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0002_add_unique_index_external_tx_id"
down_revision: Union[str, None] = "0001_initial_core_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create partial unique index on transactions.external_transaction_id
    op.create_index(
        "uq_transactions_external_tx_id",
        "transactions",
        ["external_transaction_id"],
        unique=True,
        postgresql_where=sa.text("external_transaction_id IS NOT NULL"),
        sqlite_where=sa.text("external_transaction_id IS NOT NULL"),
    )


def downgrade() -> None:
    # Drop partial unique index on downgrade
    op.drop_index(
        "uq_transactions_external_tx_id",
        table_name="transactions",
    )
