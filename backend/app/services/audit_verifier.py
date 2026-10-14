"""
Tamper-Evident Audit Logging, Cryptographic Hash Chaining & Integrity Verification Engine.

Provides:
1. Canonical, deterministic serialization for AuditLog records and JSON payloads.
2. SHA-256 cryptographic hash computation linking audit entries into a tamper-evident chain.
3. AuditIntegrityVerifier: Comprehensive verification engine detecting modified records,
   broken links, missing records, invalid hashes, and illegal unchained entries.
4. Explicit legacy record handling and formal documentation of verification limitations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import AuditActorType, AuditEntityType


# ==============================================================================
# Constants
# ==============================================================================

GENESIS_PREV_HASH: str = "0" * 64

AUDIT_CHAIN_ADVISORY_LOCK_ID: int = 84920491827491


# ==============================================================================
# Error Types & Result Data Structures
# ==============================================================================

class AuditErrorType(str, Enum):
    """Categorization of cryptographic audit integrity anomalies."""
    INVALID_HASH = "INVALID_HASH"
    BROKEN_LINK = "BROKEN_LINK"
    MISSING_RECORD = "MISSING_RECORD"
    ILLEGAL_UNCHAINED_RECORD = "ILLEGAL_UNCHAINED_RECORD"
    GENESIS_MISMATCH = "GENESIS_MISMATCH"
    DUPLICATE_HASH = "DUPLICATE_HASH"


@dataclass(frozen=True)
class AuditIntegrityError:
    """Detailed record of a single detected cryptographic audit integrity violation."""
    error_type: AuditErrorType
    message: str
    record_id: Optional[uuid.UUID] = None
    record_index: int = 0
    expected: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class AuditVerificationResult:
    """Authoritative outcome report of an audit integrity verification run."""
    is_valid: bool
    total_records: int
    verified_records: int
    legacy_records: int
    errors: List[AuditIntegrityError] = field(default_factory=list)
    first_chained_id: Optional[uuid.UUID] = None
    last_verified_id: Optional[uuid.UUID] = None
    last_verified_hash: Optional[str] = None
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    limitations: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Formatted human-readable verification summary."""
        if self.is_valid:
            return (
                f"AUDIT INTEGRITY VERIFIED: {self.verified_records} chained records verified intact "
                f"({self.legacy_records} legacy unchained records acknowledged, 0 anomalies detected)."
            )
        return (
            f"AUDIT INTEGRITY VIOLATION DETECTED: Found {len(self.errors)} anomalies across "
            f"{self.total_records} records ({self.verified_records} chained, {self.legacy_records} legacy)."
        )


# Standard documented limitations of hash-chained audit logging
STANDARD_VERIFICATION_LIMITATIONS: List[str] = [
    "Legacy Records: Audit entries created prior to Phase 15 have null hashes and cannot be cryptographically verified.",
    "Tail Truncation: Deletion of the most recent head record(s) cannot be detected without external hash anchors (e.g., git commits or remote timestamping).",
    "Complete Table Destruction: Dropping or truncating the audit table entirely destroys chain state unless backed up externally.",
    "Timestamp Monotonicity: Event timestamps are verified as part of the immutable payload; system clock adjustments do not invalidate cryptographic signatures.",
]


# ==============================================================================
# Canonical Serialization & Hash Computation
# ==============================================================================

def normalize_value_for_canonical_json(val: Any) -> Any:
    """
    Recursively normalize data values into deterministic primitives for JSON encoding.

    Guarantees:
    - Decimal objects normalized to string representations.
    - UUIDs normalized to lowercase string format.
    - Datetime objects normalized to UTC ISO-8601 with trailing 'Z'.
    - Enum instances converted to their .value string representation.
    - Dictionaries recursively sorted by key.
    - Lists/tuples recursively converted preserving sequence.
    """
    if val is None:
        return None
    if isinstance(val, (int, str, bool)):
        return val
    if isinstance(val, float):
        return val
    if isinstance(val, Decimal):
        return str(val)
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, datetime):
        dt = val
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    if hasattr(val, "value"):
        return val.value
    if isinstance(val, dict):
        return {
            str(k): normalize_value_for_canonical_json(v)
            for k, v in sorted(val.items())
        }
    if isinstance(val, (list, tuple, set)):
        return [normalize_value_for_canonical_json(item) for item in val]
    return str(val)


def compute_canonical_audit_json(
    id: Union[uuid.UUID, str],
    event_type: str,
    entity_type: Union[AuditEntityType, str],
    entity_id: Optional[Union[uuid.UUID, str]],
    action: str,
    actor_type: Union[AuditActorType, str],
    actor_id: Optional[str],
    correlation_id: Optional[str],
    client_ip: Optional[str],
    payload: Optional[Dict[str, Any]],
    event_timestamp: Union[datetime, str],
    prev_hash: Optional[str],
) -> str:
    """
    Produce a strictly deterministic, sorted canonical JSON string representing an audit record.
    """
    etype_str = entity_type.value if hasattr(entity_type, "value") else str(entity_type)
    atype_str = actor_type.value if hasattr(actor_type, "value") else str(actor_type)

    # Format event timestamp to UTC ISO 8601 string
    ts_str: str
    if isinstance(event_timestamp, datetime):
        dt = event_timestamp
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        ts_str = dt.isoformat().replace("+00:00", "Z")
    else:
        ts_str = str(event_timestamp)

    canonical_obj = {
        "action": str(action),
        "actor_id": str(actor_id) if actor_id is not None else None,
        "actor_type": atype_str,
        "client_ip": str(client_ip) if client_ip is not None else None,
        "correlation_id": str(correlation_id) if correlation_id is not None else None,
        "entity_id": str(entity_id) if entity_id is not None else None,
        "entity_type": etype_str,
        "event_timestamp": ts_str,
        "event_type": str(event_type),
        "id": str(id),
        "payload": normalize_value_for_canonical_json(payload) if payload is not None else None,
        "prev_hash": str(prev_hash) if prev_hash is not None else None,
    }

    return json.dumps(canonical_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_audit_hash(
    id: Union[uuid.UUID, str],
    event_type: str,
    entity_type: Union[AuditEntityType, str],
    entity_id: Optional[Union[uuid.UUID, str]],
    action: str,
    actor_type: Union[AuditActorType, str],
    actor_id: Optional[str],
    correlation_id: Optional[str],
    client_ip: Optional[str],
    payload: Optional[Dict[str, Any]],
    event_timestamp: Union[datetime, str],
    prev_hash: Optional[str],
) -> str:
    """
    Compute deterministic SHA-256 hexadecimal digest for an audit record.
    """
    canonical_json = compute_canonical_audit_json(
        id=id,
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        actor_type=actor_type,
        actor_id=actor_id,
        correlation_id=correlation_id,
        client_ip=client_ip,
        payload=payload,
        event_timestamp=event_timestamp,
        prev_hash=prev_hash,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_entry_hash_for_entity(entity: AuditLog, prev_hash: Optional[str]) -> str:
    """
    Compute deterministic entry_hash for an AuditLog ORM entity given a prev_hash.
    """
    return compute_audit_hash(
        id=entity.id,
        event_type=entity.event_type,
        entity_type=entity.entity_type,
        entity_id=entity.entity_id,
        action=entity.action,
        actor_type=entity.actor_type,
        actor_id=entity.actor_id,
        correlation_id=entity.correlation_id,
        client_ip=entity.client_ip,
        payload=entity.payload,
        event_timestamp=entity.event_timestamp,
        prev_hash=prev_hash,
    )


# ==============================================================================
# AuditIntegrityVerifier Engine
# ==============================================================================

class AuditIntegrityVerifier:
    """
    Cryptographic verification service for tamper-evident AuditLog records.

    Capabilities:
    1. Recomputes and verifies SHA-256 hashes against canonical payloads.
    2. Validates parent-child cryptographic links (`prev_hash == prior.entry_hash`).
    3. Detects broken links, modified payloads, missing/reordered records, and forks.
    4. Explicitly classifies and handles legacy unchained records.
    5. Flags illegal unchained records inserted after chain initialization.
    """

    def __init__(self, genesis_prev_hash: str = GENESIS_PREV_HASH) -> None:
        self.genesis_prev_hash = genesis_prev_hash

    def verify_records(
        self,
        records: Sequence[AuditLog],
    ) -> AuditVerificationResult:
        """
        Verify cryptographic integrity across an ordered sequence of AuditLog records.

        Args:
            records: Sequence of AuditLog entities ordered chronologically from oldest to newest.

        Returns:
            AuditVerificationResult documenting validation outcome and any detected anomalies.
        """
        errors: List[AuditIntegrityError] = []
        legacy_count = 0
        verified_count = 0
        chain_started = False
        first_chained_id: Optional[uuid.UUID] = None
        last_verified_id: Optional[uuid.UUID] = None
        last_verified_hash: Optional[str] = None
        seen_entry_hashes: set[str] = set()

        expected_prev_hash = self.genesis_prev_hash

        for idx, rec in enumerate(records):
            # 1. Handle Legacy (unchained) records
            if rec.entry_hash is None:
                if chain_started:
                    # Illegal unchained record injected after the chain was initialized!
                    errors.append(
                        AuditIntegrityError(
                            error_type=AuditErrorType.ILLEGAL_UNCHAINED_RECORD,
                            message=(
                                f"Record {rec.id} at index {idx} has null entry_hash but appears "
                                "after cryptographic hash chain was already initialized."
                            ),
                            record_id=rec.id,
                            record_index=idx,
                        )
                    )
                else:
                    legacy_count += 1
                continue

            # 2. First Chained (Genesis) Record
            if not chain_started:
                chain_started = True
                first_chained_id = rec.id

                # Verify genesis predecessor hash
                if rec.prev_hash != self.genesis_prev_hash:
                    errors.append(
                        AuditIntegrityError(
                            error_type=AuditErrorType.GENESIS_MISMATCH,
                            message=(
                                f"Genesis record {rec.id} at index {idx} has invalid prev_hash "
                                f"'{rec.prev_hash}'; expected genesis constant '{self.genesis_prev_hash}'."
                            ),
                            record_id=rec.id,
                            record_index=idx,
                            expected=self.genesis_prev_hash,
                            actual=rec.prev_hash,
                        )
                    )

            # 3. Subsequent Chained Records: Verify Link to Prior Hash
            else:
                if rec.prev_hash != expected_prev_hash:
                    errors.append(
                        AuditIntegrityError(
                            error_type=AuditErrorType.BROKEN_LINK,
                            message=(
                                f"Record {rec.id} at index {idx} has broken chain link: "
                                f"prev_hash '{rec.prev_hash}' does not match prior entry_hash '{expected_prev_hash}'."
                            ),
                            record_id=rec.id,
                            record_index=idx,
                            expected=expected_prev_hash,
                            actual=rec.prev_hash,
                        )
                    )

            # 4. Verify Payload & Hash Match
            expected_hash = compute_entry_hash_for_entity(rec, rec.prev_hash)
            if rec.entry_hash != expected_hash:
                errors.append(
                    AuditIntegrityError(
                        error_type=AuditErrorType.INVALID_HASH,
                        message=(
                            f"Record {rec.id} at index {idx} failed hash verification: "
                            f"stored entry_hash '{rec.entry_hash}' does not match canonical recomputed hash '{expected_hash}'."
                        ),
                        record_id=rec.id,
                        record_index=idx,
                        expected=expected_hash,
                        actual=rec.entry_hash,
                    )
                )

            # 5. Check for Duplicate Hash (fork or replay)
            if rec.entry_hash in seen_entry_hashes:
                errors.append(
                    AuditIntegrityError(
                        error_type=AuditErrorType.DUPLICATE_HASH,
                        message=f"Duplicate entry_hash '{rec.entry_hash}' detected at index {idx} for record {rec.id}.",
                        record_id=rec.id,
                        record_index=idx,
                        actual=rec.entry_hash,
                    )
                )
            seen_entry_hashes.add(rec.entry_hash)

            # Advance chain pointer
            expected_prev_hash = rec.entry_hash
            last_verified_id = rec.id
            last_verified_hash = rec.entry_hash
            verified_count += 1

        is_valid = len(errors) == 0

        return AuditVerificationResult(
            is_valid=is_valid,
            total_records=len(records),
            verified_records=verified_count,
            legacy_records=legacy_count,
            errors=errors,
            first_chained_id=first_chained_id,
            last_verified_id=last_verified_id,
            last_verified_hash=last_verified_hash,
            limitations=list(STANDARD_VERIFICATION_LIMITATIONS),
        )

    async def verify_database(
        self,
        session: AsyncSession,
        limit: Optional[int] = None,
    ) -> AuditVerificationResult:
        """
        Stream and verify all audit log records from database storage in deterministic chronological order.

        Args:
            session: Active asynchronous SQLAlchemy database session.
            limit: Optional maximum records to scan.

        Returns:
            AuditVerificationResult documenting integrity across stored logs.
        """
        stmt = select(AuditLog).order_by(AuditLog.created_at.asc(), AuditLog.event_timestamp.asc(), AuditLog.id.asc())
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)

        try:
            result = await session.execute(stmt)
            records = list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise RuntimeError(f"Database query failure scanning audit logs: {exc}") from exc

        return self.verify_records(records)


__all__ = [
    "GENESIS_PREV_HASH",
    "AUDIT_CHAIN_ADVISORY_LOCK_ID",
    "AuditErrorType",
    "AuditIntegrityError",
    "AuditVerificationResult",
    "STANDARD_VERIFICATION_LIMITATIONS",
    "normalize_value_for_canonical_json",
    "compute_canonical_audit_json",
    "compute_audit_hash",
    "compute_entry_hash_for_entity",
    "AuditIntegrityVerifier",
]
