"""
Authentication and Identity Schemas for FastAPI Fraud Detection Platform.

Provides Pydantic request and response models for user login, JWT tokens,
user profiles, administrator user provisioning, and machine API key creation.
"""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from backend.app.db.models.enums import UserRole


class LoginRequest(BaseModel):
    """Credentials payload for human user authentication."""
    model_config = ConfigDict(extra="forbid")

    username: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique account username.",
        examples=["investigator_jane"],
    )
    password: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Plaintext password credential.",
    )


class RefreshTokenRequest(BaseModel):
    """Payload for exchanging an unrevoked refresh token for a new token pair."""
    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(
        ...,
        min_length=16,
        description="Cryptographic refresh token JWT string.",
    )


class UserProfileResponse(BaseModel):
    """Public human user profile representation (excludes hashed_password)."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    username: str
    email: str
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool
    is_superuser: bool
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


class TokenResponse(BaseModel):
    """Response payload containing access token, rotated refresh token, and user info."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(
        ...,
        description="Lifetime of the access token in seconds.",
    )
    user: UserProfileResponse


class UserCreateRequest(BaseModel):
    """
    Administrator payload for provisioning a new human user account.

    Security Policies:
    - Only human roles (ANALYST, ADMIN) are permitted; machine/system types are rejected.
    - Client cannot specify `is_superuser`; all provisioned accounts default to False.
    - Password must satisfy length requirements (8-72 chars).
    """
    model_config = ConfigDict(extra="ignore")

    username: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique username for platform login.",
        examples=["analyst_alex"],
    )
    email: EmailStr = Field(
        ...,
        description="Valid email address for notification and identity.",
        examples=["alex@fraud-platform.internal"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Secure account password (8 to 72 characters).",
    )
    full_name: Optional[str] = Field(
        None,
        max_length=128,
        description="Optional display name.",
    )
    role: UserRole = Field(
        default=UserRole.ANALYST,
        description="Permitted human RBAC role (ANALYST or ADMIN).",
    )


class APIKeyCreateRequest(BaseModel):
    """
    Administrator payload for creating machine-to-machine (M2M) API credentials.

    Security Policies:
    - API keys are permanently locked to API_CLIENT role.
    - Keys cannot receive human or system roles.
    """
    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Descriptive name of the external application or service.",
        examples=["Payment Ingestion Gateway Key"],
    )
    client_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Machine actor identity recorded in audit logs.",
        examples=["payment_gateway_prod"],
    )
    description: Optional[str] = Field(
        None,
        max_length=255,
        description="Optional operational description of the machine key.",
    )
    expires_in_days: Optional[int] = Field(
        None,
        ge=1,
        le=365,
        description="Optional lifetime in days. Null indicates non-expiring.",
    )


class APIKeyCreateResponse(BaseModel):
    """
    Response returned when an API key is created.
    Exposes raw_key ONCE for secure storage by the client.
    """
    id: uuid.UUID
    name: str
    client_id: str
    key_prefix: str
    raw_key: str = Field(
        ...,
        description="Unredacted raw API key secret. Save securely; cannot be retrieved again.",
    )
    role: str = "API_CLIENT"
    is_active: bool = True
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


__all__ = [
    "LoginRequest",
    "RefreshTokenRequest",
    "UserProfileResponse",
    "TokenResponse",
    "UserCreateRequest",
    "APIKeyCreateRequest",
    "APIKeyCreateResponse",
]
