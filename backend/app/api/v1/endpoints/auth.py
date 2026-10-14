"""
Authentication, User Provisioning & API Key Management Endpoints.

Exposes REST APIs for:
- POST /api/v1/auth/login: Authenticate credentials, issue access & refresh tokens.
- POST /api/v1/auth/refresh: Exchange and rotate refresh token for new token pair.
- GET /api/v1/auth/me: Inspect current authenticated user identity and role.
- POST /api/v1/auth/users: Administrator-only user account provisioning.
- POST /api/v1/auth/api-keys: Administrator-only machine API key generation.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.core.config import settings
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    get_current_user,
    hash_password,
    require_admin_user,
    rotate_refresh_token,
    verify_password,
)
from backend.app.db.models.api_key import APIKey
from backend.app.db.models.enums import AuditActorType, AuditEntityType, UserRole
from backend.app.db.models.user import User
from backend.app.schemas.auth import (
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserCreateRequest,
    UserProfileResponse,
)
from backend.app.services.security_audit import record_security_event
from backend.app.services.unit_of_work import FraudPersistenceUnitOfWork, get_unit_of_work

router = APIRouter()


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate User & Issue Tokens",
    description="Validate username and password, update last login timestamp, and issue an access and refresh token pair.",
)
async def login(
    payload: LoginRequest,
    request: Request,
    uow: FraudPersistenceUnitOfWork = Depends(get_unit_of_work),
) -> TokenResponse:
    """
    Authenticate human user credentials.

    Fails with 401 Unauthorized on:
    - Non-existent username
    - Incorrect password
    - Inactive user account
    """
    # Resolve client IP
    client_ip: Optional[str] = None
    if settings.TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.headers.get("X-Real-IP")
    if not client_ip and request.client:
        client_ip = request.client.host

    correlation_id = getattr(request.state, "correlation_id", None) or str(uuid.uuid4())

    user = await uow.users.get_by_username(payload.username.strip())
    if not user or not verify_password(payload.password, user.hashed_password):
        # Audit login failure: invalid credentials
        await record_security_event(
            session=getattr(uow, "session", None),
            event_type="AUTH_LOGIN_FAILURE",
            action="LOGIN_ATTEMPT",
            actor_type=AuditActorType.SYSTEM,
            actor_id=payload.username.strip(),
            entity_type=AuditEntityType.USER,
            entity_id=user.id if user else None,
            correlation_id=correlation_id,
            client_ip=client_ip,
            payload={"username": payload.username.strip(), "reason": "INVALID_CREDENTIALS"},
            commit=True,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        # Audit login failure: inactive account
        await record_security_event(
            session=getattr(uow, "session", None),
            event_type="AUTH_LOGIN_FAILURE",
            action="LOGIN_ATTEMPT",
            actor_type=AuditActorType.SYSTEM,
            actor_id=user.username,
            entity_type=AuditEntityType.USER,
            entity_id=user.id,
            correlation_id=correlation_id,
            client_ip=client_ip,
            payload={"username": user.username, "reason": "ACCOUNT_INACTIVE"},
            commit=True,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive. Please contact your platform administrator.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Record last login timestamp
    await uow.users.update_last_login(user.id)

    # Record login success audit event
    await record_security_event(
        session=getattr(uow, "session", None),
        event_type="AUTH_LOGIN_SUCCESS",
        action="LOGIN",
        actor_type=AuditActorType(user.role.value),
        actor_id=user.username,
        entity_type=AuditEntityType.USER,
        entity_id=user.id,
        correlation_id=correlation_id,
        client_ip=client_ip,
        payload={"username": user.username, "role": user.role.value},
        commit=False,
    )
    await uow.commit()

    # Issue token pair
    access_token = create_access_token(subject=user.id, role=user.role, username=user.username)
    refresh_token, _ = create_refresh_token(subject=user.id, username=user.username)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserProfileResponse.model_validate(user),
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate Refresh Token",
    description="Exchange a valid unrevoked refresh token for a newly issued access token and rotated refresh token.",
)
async def refresh_token(
    payload: RefreshTokenRequest,
    uow: FraudPersistenceUnitOfWork = Depends(get_unit_of_work),
) -> TokenResponse:
    """
    Execute cryptographic refresh token rotation.

    Revokes the presented refresh token immediately and issues a new token pair.
    Re-using an already-revoked refresh token triggers an immediate 401 Unauthorized.
    """
    decoded = decode_token(payload.refresh_token, expected_type="refresh")
    sub_str = decoded.get("sub")
    if not sub_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Malformed refresh token payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = uuid.UUID(sub_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid token subject identifier.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await uow.users.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: User account is inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Atomically rotate token: revokes old JTI and generates new refresh token
    new_refresh_token, _ = rotate_refresh_token(payload.refresh_token, user.id)
    new_access_token = create_access_token(subject=user.id, role=user.role)

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserProfileResponse.model_validate(user),
    )


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get Current User Profile",
    description="Inspect identity, assigned role, and status of the current authenticated user.",
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserProfileResponse:
    """Return profile details of the calling authenticated human user."""
    return UserProfileResponse.model_validate(current_user)


@router.post(
    "/users",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Provision New User (Admin Only)",
    description="Provision a new human user account. Restricted strictly to callers with the ADMIN role.",
)
async def create_user(
    payload: UserCreateRequest,
    request: Request,
    current_admin: User = Depends(require_admin_user),
    uow: FraudPersistenceUnitOfWork = Depends(get_unit_of_work),
) -> UserProfileResponse:
    """
    Administrator endpoint for creating human operator accounts.

    Security Policies:
    - Requires ADMIN role authorization.
    - Permits only ANALYST or ADMIN roles.
    - Disallows setting is_superuser (always False).
    - Checks uniqueness of username and email.
    """
    # 1. Uniqueness check for username
    existing_username = await uow.users.get_by_username(payload.username.strip())
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Conflict: Username '{payload.username}' is already in use.",
        )

    # 2. Uniqueness check for email
    existing_email = await uow.users.get_by_email(payload.email.strip().lower())
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Conflict: Email '{payload.email}' is already in use.",
        )

    # 3. Hash password and build User entity
    hashed_pw = hash_password(payload.password)
    new_user = User(
        id=uuid.uuid4(),
        username=payload.username.strip(),
        email=payload.email.strip().lower(),
        hashed_password=hashed_pw,
        full_name=payload.full_name.strip() if payload.full_name else None,
        role=payload.role,
        is_active=True,
        is_superuser=False,  # Client cannot control is_superuser
    )

    uow.users.add(new_user)

    correlation_id = getattr(request.state, "correlation_id", None) or str(uuid.uuid4())
    client_ip = request.client.host if request and request.client else None

    await record_security_event(
        session=getattr(uow, "session", None),
        event_type="USER_CREATED",
        action="CREATE",
        actor_type=AuditActorType.ADMIN,
        actor_id=current_admin.username,
        entity_type=AuditEntityType.USER,
        entity_id=new_user.id,
        correlation_id=correlation_id,
        client_ip=client_ip,
        payload={
            "created_username": new_user.username,
            "role": new_user.role.value,
            "email": new_user.email,
        },
        commit=False,
    )
    await uow.commit()

    return UserProfileResponse.model_validate(new_user)


@router.post(
    "/api-keys",
    response_model=APIKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Machine API Key (Admin Only)",
    description="Provision machine-to-machine API credentials. Restricted strictly to callers with the ADMIN role.",
)
async def create_api_key(
    payload: APIKeyCreateRequest,
    request: Request,
    current_admin: User = Depends(require_admin_user),
    uow: FraudPersistenceUnitOfWork = Depends(get_unit_of_work),
) -> APIKeyCreateResponse:
    """
    Administrator endpoint for creating automated machine integration keys.

    Security Policies:
    - Requires ADMIN role authorization.
    - Locks key role strictly to API_CLIENT.
    - Raw secret is returned exactly ONCE; only SHA-256 digest is persisted.
    """
    raw_key, hashed_key = generate_api_key(prefix=settings.API_KEY_PREFIX)

    expires_at: Optional[datetime] = None
    if payload.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=payload.expires_in_days)

    now_utc = datetime.now(timezone.utc)
    key_record = APIKey(
        id=uuid.uuid4(),
        key_prefix=settings.API_KEY_PREFIX,
        hashed_key=hashed_key,
        name=payload.name.strip(),
        client_id=payload.client_id.strip() if payload.client_id else f"client_{uuid.uuid4().hex[:8]}",
        role=AuditActorType.API_CLIENT,
        is_active=True,
        expires_at=expires_at,
        user_id=current_admin.id,
        created_at=now_utc,
        updated_at=now_utc,
    )

    uow.api_keys.add(key_record)

    correlation_id = getattr(request.state, "correlation_id", None) or str(uuid.uuid4())
    client_ip = request.client.host if request and request.client else None

    await record_security_event(
        session=getattr(uow, "session", None),
        event_type="API_KEY_CREATED",
        action="CREATE",
        actor_type=AuditActorType.ADMIN,
        actor_id=current_admin.username,
        entity_type=AuditEntityType.API_KEY,
        entity_id=key_record.id,
        correlation_id=correlation_id,
        client_ip=client_ip,
        payload={
            "key_name": key_record.name,
            "client_id": key_record.client_id,
            "key_prefix": key_record.key_prefix,
            "expires_at": str(key_record.expires_at) if key_record.expires_at else None,
        },
        commit=False,
    )
    await uow.commit()

    return APIKeyCreateResponse(
        id=key_record.id,
        name=key_record.name,
        client_id=key_record.client_id,
        key_prefix=key_record.key_prefix,
        raw_key=raw_key,
        role="API_CLIENT",
        is_active=True,
        expires_at=key_record.expires_at,
        created_at=key_record.created_at,
    )

