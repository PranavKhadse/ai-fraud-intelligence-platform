"""
User ORM Model for Authentication, Identity, and RBAC Management.

Provides declarative persistence for user accounts, role-based access control,
and administrative security state.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional
import sqlalchemy as sa
from sqlalchemy import Boolean, CheckConstraint, DateTime, Enum, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from backend.app.db.models.enums import UserRole

if TYPE_CHECKING:
    from backend.app.db.models.api_key import APIKey


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    SQLAlchemy 2.0 ORM model for user accounts and access control credentials.

    Design Principles:
    - Independent identity table with standardized UUID primary key and timestamps.
    - Strong uniqueness constraints on username and email.
    - Typed AuditActorType enum role mapping directly to existing system actor taxonomy.
    - Active and superuser status flags for authorization checks.
    """
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
        doc="Unique username for platform authentication.",
    )
    email: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
        doc="Unique email address for user communications and identification.",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Securely hashed password (bcrypt / Argon2).",
    )
    full_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        doc="Optional human-readable full name of the user.",
    )
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=False, length=32),
        nullable=False,
        default=UserRole.ANALYST,
        server_default=UserRole.ANALYST.value,
        index=True,
        doc="Assigned human RBAC role (ANALYST, ADMIN).",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=sa.text("true"),
        index=True,
        doc="Whether user account is active. Inactive users cannot authenticate.",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=sa.text("false"),
        doc="Whether user possesses root administrative privileges.",
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp of last successful authentication.",
    )

    # Relationships
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey",
        back_populates="user",
        cascade="all, delete-orphan",
        doc="API keys associated with this user account.",
    )

    __table_args__ = (
        CheckConstraint("LENGTH(TRIM(username)) >= 3", name="chk_users_username_min_length"),
        CheckConstraint("LENGTH(TRIM(email)) >= 5", name="chk_users_email_min_length"),
        CheckConstraint("LENGTH(TRIM(hashed_password)) > 0", name="chk_users_hashed_password_non_empty"),
        CheckConstraint("role IN ('ANALYST', 'ADMIN')", name="chk_users_role_human"),
        Index("ix_users_role_active", "role", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username} role={self.role} active={self.is_active}>"


__all__ = ["User"]
