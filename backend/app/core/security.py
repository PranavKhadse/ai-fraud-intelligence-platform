"""
Security & Actor Identity Resolution Module.

Provides FastAPI dependencies for extracting, validating, and authorizing actor identity
across human review and case management endpoints.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union
import uuid

from fastapi import Depends, Header, HTTPException, Request, status

from backend.app.core.config import settings
from backend.app.db.models.enums import AuditActorType


@dataclass(frozen=True)
class ActorContext:
    """
    Context information identifying the actor initiating a case management action.
    """
    actor_id: str
    actor_role: Union[AuditActorType, str] = AuditActorType.ANALYST
    correlation_id: Optional[str] = None
    client_ip: Optional[str] = None


async def get_current_actor(
    request: Request,
    x_actor_id: Optional[str] = Header(None, alias="X-Actor-ID"),
    x_actor_role: Optional[str] = Header(None, alias="X-Actor-Role"),
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
) -> ActorContext:
    """
    FastAPI dependency that extracts and validates the caller's actor context.

    Security Policies:
    - Production Mode (ALLOW_DEV_ACTOR_HEADERS=False): Fails closed (401 Unauthorized)
      unless verified through an enterprise authentication gateway.
    - Development/Test Mode (ALLOW_DEV_ACTOR_HEADERS=True): Requires explicit, valid
      X-Actor-ID and X-Actor-Role headers. Never silently defaults to an assumed analyst.
    - Role Validation: Strictly validates X-Actor-Role against AuditActorType enum.

    Args:
        request: FastAPI HTTP request object.
        x_actor_id: Optional header specifying the actor's identifier.
        x_actor_role: Optional header specifying the actor's role.
        x_correlation_id: Optional header specifying request correlation ID.

    Returns:
        Validated ActorContext instance.

    Raises:
        HTTPException: 401 Unauthorized if headers are missing, invalid, or dev auth is disabled.
    """
    if not settings.ALLOW_DEV_ACTOR_HEADERS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Actor authentication required. Header-based identity injection is disabled in this environment.",
        )

    # Validate X-Actor-ID presence
    if not x_actor_id or not x_actor_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: X-Actor-ID header is required.",
        )

    # Validate X-Actor-Role presence
    if not x_actor_role or not x_actor_role.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: X-Actor-Role header is required.",
        )

    role_str = x_actor_role.strip().upper()
    try:
        actor_role = AuditActorType(role_str)
    except ValueError:
        valid_roles = [e.value for e in AuditActorType]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unauthorized: Invalid X-Actor-Role '{x_actor_role}'. Valid roles: {valid_roles}",
        )

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
        x_correlation_id
        or getattr(request.state, "correlation_id", None)
        or str(uuid.uuid4())
    )

    return ActorContext(
        actor_id=x_actor_id.strip(),
        actor_role=actor_role,
        correlation_id=correlation_id,
        client_ip=client_ip,
    )


def require_role(
    *allowed_roles: Union[AuditActorType, str],
) -> Callable[..., ActorContext]:
    """
    Factory for role-based authorization dependencies.

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
    ) -> ActorContext:
        actor_role_val = (
            actor.actor_role.value
            if isinstance(actor.actor_role, AuditActorType)
            else str(actor.actor_role).upper()
        )
        if actor_role_val not in normalized_allowed:
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
    "get_current_actor",
    "require_role",
]
