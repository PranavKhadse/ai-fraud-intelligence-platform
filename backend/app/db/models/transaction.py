"""
Transaction ORM Model for Persisting Canonical Financial Transactions.

Represents an incoming transaction event, its canonical features, and the point-in-time
engineered feature snapshot.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import uuid

from sqlalchemy import String, Numeric, Integer, DateTime, JSON, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models.base import Base, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from backend.app.db.models.risk_evaluation import RiskEvaluation


class Transaction(Base, UUIDPrimaryKeyMixin):
    """
    SQLAlchemy 2.0 ORM model for financial transaction records.

    Design Choices:
    - Primary key is an internal UUID (`id`).
    - `external_transaction_id` captures optional client-provided identifiers.
    - `amount` uses `Numeric(15, 2)` to eliminate floating-point precision loss.
    - `features_snapshot` stores the complete 55-feature point-in-time vector in JSON/JSONB.
    - Sensitive card numbers, CVVs, and plain passwords are NEVER stored.
    """
    __tablename__ = "transactions"

    # External / Client Identifier
    external_transaction_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="Optional client-supplied transaction identifier (not guaranteed globally unique).",
    )

    # Core Identifiers and Categoricals
    account_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        doc="Cardholder or account identifier token.",
    )
    merchant_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="Optional merchant account identifier.",
    )
    merchant_category: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Merchant industry category classification.",
    )
    job_category: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Cardholder employment category.",
    )

    # Monetary Attributes
    amount: Mapped[Decimal] = mapped_column(
        Numeric(15, 2),
        nullable=False,
        doc="Transaction amount in specified currency.",
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
        default="USD",
        doc="ISO 4217 three-letter currency code.",
    )

    # Geographical and Demographic Predictors
    cardholder_lat: Mapped[Decimal] = mapped_column(
        Numeric(9, 6),
        nullable=False,
        doc="Cardholder home latitude coordinates.",
    )
    cardholder_long: Mapped[Decimal] = mapped_column(
        Numeric(9, 6),
        nullable=False,
        doc="Cardholder home longitude coordinates.",
    )
    merchant_lat: Mapped[Decimal] = mapped_column(
        Numeric(9, 6),
        nullable=False,
        doc="Merchant terminal latitude coordinates.",
    )
    merchant_long: Mapped[Decimal] = mapped_column(
        Numeric(9, 6),
        nullable=False,
        doc="Merchant terminal longitude coordinates.",
    )
    city_pop: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Cardholder city population.",
    )

    # Timestamps
    transaction_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Original timestamp when the transaction occurred.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="UTC timestamp when the record was ingested into the database.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="UTC timestamp of the latest update.",
    )

    # Point-in-time Feature Vector Snapshot
    features_snapshot: Mapped[Dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        doc="Complete 55-feature engineered behavioral vector at evaluation time.",
    )

    # Relationships
    evaluations: Mapped[List["RiskEvaluation"]] = relationship(
        "RiskEvaluation",
        back_populates="transaction",
        cascade="save-update, merge",
        passive_deletes=True,
        lazy="selectin",
        doc="Risk evaluations performed against this transaction.",
    )

    __table_args__ = (
        CheckConstraint("amount >= 0.00", name="chk_transactions_amount_positive"),
        Index("ix_transactions_account_ts", "account_id", "transaction_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction id={self.id} account_id={self.account_id} "
            f"amount={self.amount} {self.currency} ts={self.transaction_timestamp}>"
        )
