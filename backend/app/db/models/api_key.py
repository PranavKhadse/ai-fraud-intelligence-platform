"""
APIKey ORM Model for Machine-to-Machine (M2M) Authentication & Token Authorization.

Provides declarative persistence for external service API keys, identifying
client applications, scoping permissions, and managing cryptographic revocation.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
import uuid
import sqlalchemy as sa
from sqlalchemy import Boolean, CheckConstraint, DateTime, Enum, ForeignKey, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from backend.app.db.models.enums import AuditActorType

if TYPE_CHECKING:
    from backend.app.db.models.user import User


class APIKey(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    SQLAlchemy 2.0 ORM model for automated machine-to-machine API credentials.

    Design Principles:
    - Never stores raw API key tokens; stores cryptographic SHA-256 hash digests.
    - Public key_prefix enables fast indexed database lookup without secret disclosure.
    - client_id maps directly to actor_id in ActorContext telemetry.
    - Configurable expiration, revocation flag, and optional link to a managing User.
    """
    __tablename__ = "api_keys"

    key_prefix: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        index=True,
        doc="Public key prefix (e.g. 'pk_live_') for lookup and identification.",
    )
    hashed_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        doc="Cryptographic SHA-256 digest of the API key secret token.",
    )
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Human-readable label describing the calling application or integration.",
    )
    client_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        doc="Client identifier mapping to actor_id in ActorContext telemetry.",
    )
    role: Mapped[AuditActorType] = mapped_column(
        Enum(AuditActorType, native_enum=False, length=32),
        nullable=False,
        default=AuditActorType.API_CLIENT,
        server_default=AuditActorType.API_CLIENT.value,
        index=True,
        doc="Actor role granted to this API key (default: API_CLIENT).",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=sa.text("true"),
        index=True,
        doc="Whether this API key is active. Revoked keys have is_active=False.",
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Optional expiration timestamp. Expired keys are rejected.",
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when this API key was last used for an authenticated request.",
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", name="fk_api_keys_user_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        doc="Optional owning User account ID if key is user-managed.",
    )

    # Relationship
    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="api_keys",
    )

    __table_args__ = (
        CheckConstraint("LENGTH(TRIM(key_prefix)) >= 3", name="chk_api_keys_prefix_min_length"),
        CheckConstraint("LENGTH(TRIM(hashed_key)) > 0", name="chk_api_keys_hashed_key_non_empty"),
        CheckConstraint("LENGTH(TRIM(client_id)) > 0", name="chk_api_keys_client_id_non_empty"),
        CheckConstraint("LENGTH(TRIM(name)) > 0", name="chk_api_keys_name_non_empty"),
        CheckConstraint("role IN ('API_CLIENT')", name="chk_api_keys_role_machine"),
        Index("ix_api_keys_prefix_active", "key_prefix", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<APIKey id={self.id} prefix={self.key_prefix} client_id={self.client_id} active={self.is_active}>"


__all__ = ["APIKey"]
