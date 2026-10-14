"""
Security Event Auditing Utility for Authentication & Authorization Tracking.

Provides:
- Sanitization helper stripping credentials, tokens, secrets, and raw keys from audit payloads.
- Centralized `record_security_event` logging security events directly into the
  tamper-evident SHA-256 audit hash chain.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional, Set, Union
from unittest.mock import MagicMock
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import AuditActorType, AuditEntityType
from backend.app.repositories.audit_log_repository import AuditLogRepository

logger = logging.getLogger("security_audit")

# Sensitive parameter keys that must never be persisted in audit log payloads
SENSITIVE_AUDIT_KEYS: Set[str] = frozenset({
    "password",
    "hashed_password",
    "raw_key",
    "secret",
    "hashed_key",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "bearer",
    "credentials",
})


def sanitize_audit_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Recursively sanitize an audit payload, stripping any credentials, secrets, or tokens.

    Args:
        payload: Dictionary containing event metadata.

    Returns:
        Sanitized dictionary with sensitive fields removed.
    """
    if payload is None:
        return None

    if not isinstance(payload, dict):
        return {"data": str(payload)}

    sanitized: Dict[str, Any] = {}
    for key, val in payload.items():
        key_lower = str(key).lower().strip()
        # Filter exact matches or partial secret indicators
        if key_lower in SENSITIVE_AUDIT_KEYS or any(
            sec in key_lower for sec in ("password", "raw_key", "secret", "token", "credential", "bearer")
        ):
            continue

        if isinstance(val, dict):
            sanitized[key] = sanitize_audit_payload(val)
        elif isinstance(val, (list, tuple)):
            sanitized[key] = [
                sanitize_audit_payload(item) if isinstance(item, dict) else item
                for item in val
            ]
        else:
            sanitized[key] = val

    return sanitized


async def record_security_event(
    session: Optional[AsyncSession],
    event_type: str,
    action: str,
    actor_type: Union[AuditActorType, str],
    actor_id: Optional[str],
    entity_type: Union[AuditEntityType, str] = AuditEntityType.SYSTEM,
    entity_id: Optional[uuid.UUID] = None,
    correlation_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    commit: bool = False,
) -> Optional[AuditLog]:
    """
    Record a security audit event into the tamper-evident cryptographic hash chain.

    Guarantees:
    - Never persists passwords, raw API keys, JWT tokens, or credentials.
    - Uses `AuditLogRepository.add` ensuring continuous SHA-256 hash chaining.
    - Transaction safety: optional immediate commit for error/denial outcomes.
    - Defensive: failures in audit recording log a warning without masking application errors.

    Args:
        session: Active SQLAlchemy AsyncSession.
        event_type: Event classification identifier (e.g. 'AUTH_LOGIN_SUCCESS', 'AUTHZ_ACCESS_DENIED').
        action: Executed action (e.g. 'LOGIN', 'ACCESS_DENIED', 'CREATE').
        actor_type: Role/type of originating actor.
        actor_id: Identifier of actor (username, client_id, or IP).
        entity_type: Category of target entity.
        entity_id: Target entity UUID if applicable.
        correlation_id: Request correlation tracing identifier.
        client_ip: Originating client IP address.
        payload: Event metadata dictionary (will be sanitized).
        commit: If True, executes `await session.commit()` immediately.

    Returns:
        The staged and chained AuditLog entity if persisted, otherwise None.
    """
    if session is None:
        return None

    try:
        clean_payload = sanitize_audit_payload(payload)

        # Normalize enum types
        atype = (
            AuditActorType(actor_type)
            if isinstance(actor_type, str) and actor_type in AuditActorType.__members__
            else actor_type
        )
        if not isinstance(atype, AuditActorType):
            try:
                atype = AuditActorType(str(actor_type).upper())
            except ValueError:
                atype = AuditActorType.SYSTEM

        etype = (
            AuditEntityType(entity_type)
            if isinstance(entity_type, str) and entity_type in AuditEntityType.__members__
            else entity_type
        )
        if not isinstance(etype, AuditEntityType):
            try:
                etype = AuditEntityType(str(entity_type).upper())
            except ValueError:
                etype = AuditEntityType.SYSTEM

        now_utc = datetime.now(timezone.utc)
        log_entry = AuditLog(
            id=uuid.uuid4(),
            event_type=event_type,
            entity_type=etype,
            entity_id=entity_id,
            action=action,
            actor_type=atype,
            actor_id=str(actor_id) if actor_id is not None else None,
            correlation_id=str(correlation_id) if correlation_id is not None else None,
            client_ip=str(client_ip) if client_ip is not None else None,
            payload=clean_payload,
            event_timestamp=now_utc,
            created_at=now_utc,
        )

        repo = AuditLogRepository(session)
        await repo.add(log_entry)

        if commit:
            await session.commit()

        return log_entry

    except Exception as exc:
        logger.warning(
            "Failed to record security audit event '%s': %s",
            event_type,
            exc,
            exc_info=True,
        )
        return None
