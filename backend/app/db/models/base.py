"""
SQLAlchemy 2.0 Declarative Base and Shared Mixins for Database Models.

Provides:
- `Base`: DeclarativeBase class for all application ORM models.
- `UUIDPrimaryKeyMixin`: Standardized UUID primary key column.
- `TimestampMixin`: Standardized timezone-aware UTC created_at / updated_at columns.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import Uuid, DateTime


class Base(DeclarativeBase):
    """
    SQLAlchemy 2.0 DeclarativeBase superclass for all database models.
    """
    pass


class UUIDPrimaryKeyMixin:
    """
    Mixin providing a UUID primary key using PostgreSQL-compatible UUID types.
    """
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        sort_order=-100,
    )


class TimestampMixin:
    """
    Mixin providing created_at and updated_at timezone-aware UTC timestamps.
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        sort_order=900,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        sort_order=901,
    )
