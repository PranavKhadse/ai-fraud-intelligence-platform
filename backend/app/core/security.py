"""
Security, Cryptographic Utilities & Actor Identity Resolution Module.

Provides:
- Bcrypt password hashing and verification.
- Secure API key generation and constant-time SHA-256 verification.
- JWT access and refresh token creation, validation, rotation, and revocation.
- FastAPI dependencies for resolving authenticated Users, Administrators, and ActorContext.
- Role-based authorization guardrails enforcing separation of human and machine identities.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from pathlib import Path
import secrets
import threading
from typing import Callable, Optional, Sequence, Set, Tuple, Union
import uuid

import bcrypt
from fastapi import Depends, Header, HTTPException, Request, Security, status
from fastapi.params import Depends as DependsParam
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.db.models.api_key import APIKey
from backend.app.db.models.enums import (
    HUMAN_ROLES,
    MACHINE_ROLES,
    SYSTEM_ROLES,
    AuditActorType,
    AuditEntityType,
    UserRole,
)
from backend.app.db.models.user import User
from backend.app.db.session import get_db_session


# ==============================================================================
# 0. Actor Context Definition (at top to prevent circular imports)
# ==============================================================================

@dataclass(frozen=True)
class ActorContext:
    """
    Context information identifying the actor initiating an action.
    """
    actor_id: str
    actor_role: Union[AuditActorType, str] = AuditActorType.ANALYST
    correlation_id: Optional[str] = None
    client_ip: Optional[str] = None


# Bearer token extractor for FastAPI dependencies
bearer_scheme = HTTPBearer(auto_error=False)


# ==============================================================================
# 1. Password Hashing & Verification
# ==============================================================================

def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt with salt rounds 12.

    Args:
        password: Plaintext password string.

    Returns:
        Bcrypt hashed password string ($2b$12$...).

    Raises:
        ValueError: If password is empty, too short (< 8 chars), or exceeds 72 bytes.
    """
    if not password:
        raise ValueError("Password cannot be empty.")
    password_bytes = password.encode("utf-8")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if len(password_bytes) > 72:
        raise ValueError("Password cannot exceed 72 bytes due to bcrypt limitation.")

    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.

    Args:
        plain_password: Provided password string.
        hashed_password: Stored bcrypt hash string.

    Returns:
        True if password matches hash, False otherwise. Never throws.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


# ==============================================================================
# 2. API Key Generation & Verification
# ==============================================================================

def generate_api_key(prefix: str = "pk_live_") -> Tuple[str, str]:
    """
    Generate a cryptographically secure random API key and its SHA-256 digest.

    Args:
        prefix: 8-character public key prefix (default: 'pk_live_').

    Returns:
        Tuple of (unhashed_raw_key, sha256_hashed_key).
        The raw key is shown ONCE to the caller; the hash is stored in the database.
    """
    if len(prefix) != 8:
        raise ValueError(f"API key prefix must be exactly 8 characters long, got '{prefix}'.")

    secret = secrets.token_urlsafe(32)
    raw_key = f"{prefix}{secret}"
    hashed_key = hash_api_key(raw_key)
    return raw_key, hashed_key


def hash_api_key(raw_key: str) -> str:
    """
    Compute deterministic SHA-256 hexadecimal digest of an API key string.

    Args:
        raw_key: Raw unhashed API key string.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def verify_api_key(raw_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against a stored SHA-256 hash using constant-time comparison.

    Args:
        raw_key: Raw API key presented by the client.
        stored_hash: Stored 64-character SHA-256 digest.

    Returns:
        True if the raw key produces the stored hash, False otherwise.
    """
    if not raw_key or not stored_hash:
        return False
    computed = hash_api_key(raw_key)
    return hmac.compare_digest(computed.lower(), stored_hash.strip().lower())


# ==============================================================================
# 3. Refresh Token Revocation & Rotation Store
# ==============================================================================

class RefreshTokenRevocationStore:
    """
    Thread-safe cryptographic revocation store for rotated or invalidated refresh tokens.

    Security Guarantees:
    - Never stores plaintext refresh tokens; stores SHA-256 digests of unique JWT IDs (jti).
    - Prevents token replay and unlimited replacement token issuance.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revoked_jti_hashes: Set[str] = set()

    def _hash_jti(self, jti: str) -> str:
        return hashlib.sha256(jti.strip().encode("utf-8")).hexdigest()

    def revoke_jti(self, jti: str) -> None:
        """Mark a refresh token JTI as revoked."""
        if not jti:
            return
        h = self._hash_jti(jti)
        with self._lock:
            self._revoked_jti_hashes.add(h)

    def is_jti_revoked(self, jti: str) -> bool:
        """Check whether a refresh token JTI has been revoked or already rotated."""
        if not jti:
            return True
        h = self._hash_jti(jti)
        with self._lock:
            return h in self._revoked_jti_hashes

    def reset(self) -> None:
        """Clear all revocation state (used primarily in test teardown)."""
        with self._lock:
            self._revoked_jti_hashes.clear()

    clear = reset


# Global singleton instance for token revocation tracking
token_revocation_store = RefreshTokenRevocationStore()


# ==============================================================================
# 4. JWT Token Creation, Decoding & Rotation
# ==============================================================================

def create_access_token(
    subject: Union[str, uuid.UUID],
    role: Union[UserRole, str],
    expires_delta: Optional[timedelta] = None,
    username: Optional[str] = None,
    claims: Optional[dict] = None,
) -> str:
    """
    Create a signed short-lived JWT access token for human user authorization.

    Security Policies:
    - Strictly validates that role belongs to HUMAN_ROLES (ANALYST, ADMIN).
    - Machine or system roles cannot obtain user access tokens.

    Args:
        subject: User ID UUID or unique identifier.
        role: UserRole or role string.
        expires_delta: Optional custom duration.
        username: Optional human username claim.
        claims: Optional additional non-sensitive payload claims.

    Returns:
        Encoded JWT token string.
    """
    role_str = role.value if hasattr(role, "value") else str(role).strip().upper()
    if role_str not in HUMAN_ROLES:
        raise ValueError(
            f"Access token creation rejected: Role '{role_str}' is not a permitted human role. "
            f"Permitted: {sorted(list(HUMAN_ROLES))}"
        )

    now = datetime.now(timezone.utc)
    lifetime = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = now + lifetime

    payload = {
        "sub": str(subject),
        "role": role_str,
        "token_type": "access",
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": str(uuid.uuid4()),
    }
    if username:
        payload["username"] = username
    if claims:
        payload.update(claims)

    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(
    subject: Union[str, uuid.UUID],
    expires_delta: Optional[timedelta] = None,
    username: Optional[str] = None,
    claims: Optional[dict] = None,
) -> Tuple[str, str]:
    """
    Create a signed long-lived JWT refresh token with unique JTI.

    Args:
        subject: User ID UUID or unique identifier.
        expires_delta: Optional custom duration.
        username: Optional human username claim.
        claims: Optional additional non-sensitive payload claims.

    Returns:
        Tuple of (encoded_refresh_token, jti).
    """
    now = datetime.now(timezone.utc)
    lifetime = expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    expire = now + lifetime
    jti = str(uuid.uuid4())

    payload = {
        "sub": str(subject),
        "token_type": "refresh",
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": jti,
    }
    if username:
        payload["username"] = username
    if claims:
        payload.update(claims)

    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, jti


def decode_token(token: str, expected_type: Optional[str] = None) -> dict:
    """
    Decode, cryptographically verify, and validate a JWT token string.

    Args:
        token: Raw JWT token string.
        expected_type: Optional expected 'token_type' ('access' or 'refresh').

    Returns:
        Decoded claims dictionary.

    Raises:
        HTTPException: 401 Unauthorized if token is expired, invalid, tampered, or revoked.
    """
    if not token or not str(token).strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Token string is missing or empty.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token.strip(),
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unauthorized: Token signature or format is invalid ({exc}).",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate token_type claim
    actual_type = payload.get("token_type")
    if expected_type and actual_type != expected_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unauthorized: Invalid token type '{actual_type}'. Expected '{expected_type}'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check refresh token revocation status
    if actual_type == "refresh":
        jti = payload.get("jti")
        if not jti or token_revocation_store.is_jti_revoked(jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Refresh token has been revoked or already rotated.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return payload


def rotate_refresh_token(old_refresh_token: str, subject: Union[str, uuid.UUID]) -> Tuple[str, str]:
    """
    Validate, revoke old refresh token JTI, and issue a new replacement token.

    Args:
        old_refresh_token: Refresh token presented by client.
        subject: User identifier.

    Returns:
        Tuple of (new_refresh_token, new_jti).
    """
    old_payload = decode_token(old_refresh_token, expected_type="refresh")
    old_jti = old_payload.get("jti")

    # Revoke old JTI immediately
    if old_jti:
        token_revocation_store.revoke_jti(old_jti)

    # Issue rotated token
    return create_refresh_token(subject)


# ==============================================================================
# 5. FastAPI Dependencies for Authentication & User Resolution
# ==============================================================================

async def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """
    FastAPI dependency that resolves the authenticated User from Bearer JWT.

    Security Policies:
    - Verifies token signature, expiration, and 'access' token type.
    - Validates token role belongs to HUMAN_ROLES.
    - Fetches current User record from database.
    - Rejects inactive or missing user accounts.
    """
    token_str: Optional[str] = None
    if isinstance(auth, HTTPAuthorizationCredentials) and auth.credentials:
        token_str = auth.credentials
    elif isinstance(auth, str) and auth.strip():
        token_str = auth.strip()

    if not token_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Authentication required. Bearer token missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token_str, expected_type="access")
    sub_str = payload.get("sub")
    role_str = payload.get("role")

    if not sub_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Token contains no subject identifier.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if role_str not in HUMAN_ROLES:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unauthorized: Access token role '{role_str}' is not a valid human role.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = uuid.UUID(sub_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid subject identifier format.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from backend.app.repositories.user_repository import UserRepository

    user_repo = UserRepository(session)
    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: User account not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: User account is inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def require_admin_user(
    current_user: User = Depends(get_current_user),
    request: Request = None,
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """
    FastAPI dependency that enforces that the caller has an ADMIN role.
    """
    user_role_str = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role).upper()
    if user_role_str != UserRole.ADMIN.value:
        try:
            from backend.app.services.security_audit import record_security_event

            path = request.url.path if request and hasattr(request, "url") else None
            method = request.method if request and hasattr(request, "method") else None
            ip = None
            if request and hasattr(request, "client") and request.client:
                ip = request.client.host
            cid = getattr(request.state, "correlation_id", None) if request and hasattr(request, "state") else None

            await record_security_event(
                session=session,
                event_type="AUTHZ_ACCESS_DENIED",
                action="ACCESS_DENIED",
                actor_type=AuditActorType(user_role_str) if user_role_str in AuditActorType.__members__ else AuditActorType.ANALYST,
                actor_id=current_user.username,
                entity_type=AuditEntityType.SYSTEM,
                correlation_id=cid,
                client_ip=ip,
                payload={
                    "path": path,
                    "method": method,
                    "attempted_role": user_role_str,
                    "required_roles": [UserRole.ADMIN.value],
                },
                commit=True,
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Administrative privileges required for this operation.",
        )
    return current_user


async def get_current_actor(
    request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_actor_id: Optional[str] = Header(None, alias="X-Actor-ID"),
    x_actor_role: Optional[str] = Header(None, alias="X-Actor-Role"),
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
    session: AsyncSession = Depends(get_db_session),
) -> ActorContext:
    """
    FastAPI dependency resolving the caller's authoritative ActorContext.

    Priority Order:
    1. Bearer JWT: Authenticated human user -> ActorContext(actor_id=user.username, actor_role=user.role)
    2. X-API-Key: Authenticated machine -> ActorContext(actor_id=api_key.client_id, actor_role=API_CLIENT)
    3. Dev/Test Headers (X-Actor-ID, X-Actor-Role): Permitted only if ALLOW_DEV_ACTOR_HEADERS=True.
    4. Fails closed (HTTP 401 Unauthorized).
    """
    # Clean up FastAPI params if called directly as a Python coroutine
    credentials_str: Optional[str] = None
    if isinstance(auth, HTTPAuthorizationCredentials) and auth.credentials:
        credentials_str = auth.credentials
    elif isinstance(auth, str) and auth.strip():
        credentials_str = auth.strip()

    api_key_str: Optional[str] = x_api_key if isinstance(x_api_key, str) and x_api_key.strip() else None
    actor_id_str: Optional[str] = x_actor_id if isinstance(x_actor_id, str) and x_actor_id.strip() else None
    actor_role_str: Optional[str] = x_actor_role if isinstance(x_actor_role, str) and x_actor_role.strip() else None
    correlation_id_str: Optional[str] = x_correlation_id if isinstance(x_correlation_id, str) and x_correlation_id.strip() else None

    # Extract client IP
    client_ip: Optional[str] = None
    if settings.TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.headers.get("X-Real-IP")
    if not client_ip and request.client:
        client_ip = request.client.host

    # Extract correlation ID
    correlation_id = (
        correlation_id_str
        or getattr(request.state, "correlation_id", None)
        or str(uuid.uuid4())
    )

    # 1. Bearer JWT Authentication
    if credentials_str:
        payload = decode_token(credentials_str, expected_type="access")
        sub_str = payload.get("sub")
        role_str = payload.get("role", "ANALYST")
        username_str = payload.get("username")
        try:
            actor_role = AuditActorType(role_str)
        except ValueError:
            actor_role = AuditActorType.ANALYST

        # Prefer username claim from token, fallback to resolving from DB if subject is UUID
        actor_id = username_str or sub_str
        if not username_str and session is not None and not isinstance(session, DependsParam):
            try:
                from backend.app.repositories.user_repository import UserRepository

                user_id = uuid.UUID(sub_str)
                user_repo = UserRepository(session)
                user = await user_repo.get_by_id(user_id)
                if user:
                    actor_id = user.username
            except (ValueError, Exception):
                pass

        return ActorContext(
            actor_id=actor_id,
            actor_role=actor_role,
            correlation_id=correlation_id,
            client_ip=client_ip,
        )

    # 2. API Key Authentication
    if api_key_str:
        if session is None or isinstance(session, DependsParam):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session required for API key authentication.",
            )

        from backend.app.repositories.api_key_repository import APIKeyRepository

        raw_key = api_key_str
        hashed = hash_api_key(raw_key)
        key_repo = APIKeyRepository(session)
        api_key = await key_repo.get_by_hashed_key(hashed)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid API key secret.",
            )

        if not api_key.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: API key is revoked or inactive.",
            )

        if api_key.expires_at:
            now_utc = datetime.now(timezone.utc)
            if api_key.expires_at <= now_utc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized: API key has expired.",
                )

        # Update last used timestamp asynchronously
        await key_repo.update_last_used(api_key.id)

        return ActorContext(
            actor_id=api_key.client_id,
            actor_role=AuditActorType.API_CLIENT,
            correlation_id=correlation_id,
            client_ip=client_ip,
        )

    # 3. Development/Test Mode Header Fallback
    if settings.ALLOW_DEV_ACTOR_HEADERS:
        if not actor_id_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: X-Actor-ID header is required in dev header mode.",
            )
        if not actor_role_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: X-Actor-Role header is required in dev header mode.",
            )

        role_str = actor_role_str.upper()
        try:
            actor_role = AuditActorType(role_str)
        except ValueError:
            valid_roles = [e.value for e in AuditActorType]
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Unauthorized: Invalid X-Actor-Role '{actor_role_str}'. Valid roles: {valid_roles}",
            )

        return ActorContext(
            actor_id=actor_id_str,
            actor_role=actor_role,
            correlation_id=correlation_id,
            client_ip=client_ip,
        )

    # 4. Fails closed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Actor authentication required. Provide a valid Bearer token or X-API-Key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(
    *allowed_roles: Union[AuditActorType, str],
) -> Callable[..., ActorContext]:
    """
    Factory for role-based authorization dependencies on ActorContext.

    Args:
        *allowed_roles: Variable list of permitted AuditActorType roles or string names.

    Returns:
        A FastAPI dependency function that validates the caller has one of the allowed roles.
    """
    normalized_allowed: set[str] = {
        r.value if isinstance(r, AuditActorType) else str(r).upper()
        for r in allowed_roles
    }

    async def _role_checker(
        actor: ActorContext = Depends(get_current_actor),
        request: Request = None,
        session: AsyncSession = Depends(get_db_session),
    ) -> ActorContext:
        actor_role_val = (
            actor.actor_role.value
            if isinstance(actor.actor_role, AuditActorType)
            else str(actor.actor_role).upper()
        )
        if actor_role_val not in normalized_allowed:
            try:
                from backend.app.services.security_audit import record_security_event

                path = request.url.path if request and hasattr(request, "url") else None
                method = request.method if request and hasattr(request, "method") else None

                await record_security_event(
                    session=session,
                    event_type="AUTHZ_ACCESS_DENIED",
                    action="ACCESS_DENIED",
                    actor_type=actor.actor_role,
                    actor_id=actor.actor_id,
                    entity_type=AuditEntityType.SYSTEM,
                    correlation_id=actor.correlation_id,
                    client_ip=actor.client_ip,
                    payload={
                        "path": path,
                        "method": method,
                        "attempted_role": actor_role_val,
                        "required_roles": sorted(list(normalized_allowed)),
                    },
                    commit=True,
                )
            except Exception:
                pass

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Forbidden: Actor role '{actor_role_val}' is not authorized for this operation. "
                    f"Required: {sorted(list(normalized_allowed))}"
                ),
            )
        return actor

    return _role_checker


__all__ = [
    "ActorContext",
    "hash_password",
    "verify_password",
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    "token_revocation_store",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "rotate_refresh_token",
    "get_current_user",
    "require_admin_user",
    "get_current_actor",
    "require_role",
]
