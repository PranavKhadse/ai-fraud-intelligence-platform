"""
Audit Log Verification & Tamper-Evidence Endpoints.

Exposes REST APIs for:
- GET /api/v1/audit/verify: Inspect cryptographic hash chain integrity (Admin Only).
"""

from typing import Optional
import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import ActorContext, require_role
from backend.app.db.models.enums import AuditActorType, AuditEntityType
from backend.app.db.session import get_db_session
from backend.app.schemas.audit import AuditIntegrityErrorSchema, AuditVerificationResponse
from backend.app.services.audit_verifier import AuditIntegrityVerifier
from backend.app.services.security_audit import record_security_event

router = APIRouter()


@router.get(
    "/verify",
    response_model=AuditVerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify Audit Log Cryptographic Integrity (Admin Only)",
    description=(
        "Scans stored audit log records in deterministic chronological order and verifies "
        "SHA-256 hash chaining, detection of payload tampering, severed links, missing/reordered "
        "records, and unchained injections. Restricted strictly to callers with the ADMIN role."
    ),
)
async def verify_audit_log_integrity(
    limit: Optional[int] = Query(None, ge=1, le=100000, description="Optional maximum records to scan."),
    actor: ActorContext = Depends(require_role(AuditActorType.ADMIN)),
    session: AsyncSession = Depends(get_db_session),
) -> AuditVerificationResponse:
    """
    Execute full cryptographic hash chain verification across stored audit logs.

    Security Policies:
    - Requires ADMIN role authorization (fails with 403 Forbidden for ANALYST or API_CLIENT).
    - Logs an immutable AUDIT_VERIFICATION_EXECUTED security event documenting caller and outcome.
    """
    verifier = AuditIntegrityVerifier()
    result = await verifier.verify_database(session, limit=limit)

    # Record the verification execution in the audit log
    await record_security_event(
        session=session,
        event_type="AUDIT_VERIFICATION_EXECUTED",
        action="VERIFY",
        actor_type=actor.actor_role,
        actor_id=actor.actor_id,
        entity_type=AuditEntityType.SYSTEM,
        correlation_id=actor.correlation_id,
        client_ip=actor.client_ip,
        payload={
            "is_valid": result.is_valid,
            "total_records": result.total_records,
            "verified_records": result.verified_records,
            "legacy_records": result.legacy_records,
            "error_count": len(result.errors),
        },
        commit=True,
    )

    return AuditVerificationResponse(
        is_valid=result.is_valid,
        total_records=result.total_records,
        verified_records=result.verified_records,
        legacy_records=result.legacy_records,
        errors=[
            AuditIntegrityErrorSchema(
                error_type=err.error_type.value,
                message=err.message,
                record_id=err.record_id,
                record_index=err.record_index,
                expected=err.expected,
                actual=err.actual,
            )
            for err in result.errors
        ],
        first_chained_id=result.first_chained_id,
        last_verified_id=result.last_verified_id,
        last_verified_hash=result.last_verified_hash,
        verified_at=result.verified_at,
        limitations=result.limitations,
        summary=result.summary,
    )
