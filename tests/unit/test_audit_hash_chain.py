"""
Unit and Integration Tests for Phase 15.4: Tamper-Evident Audit Logging & Verification.

Tests:
1. Canonical JSON deterministic normalization and serialization.
2. SHA-256 hash generation and sensitivity to payload/field tampering.
3. Valid single and multi-entry audit hash chains.
4. Tampering detection (payload alteration, action modification, timestamp change).
5. Broken chain link detection (broken predecessor hashes).
6. Missing and reordered record detection.
7. Legacy unchained record handling and illegal unchained record injection detection.
8. Duplicate hash / fork detection.
9. Repository automatic chain computation (add and add_many).
10. Concurrent append anti-forking simulation via asyncio.gather.
11. AuditIntegrityVerifier database streaming and error reporting.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import AuditActorType, AuditEntityType
from backend.app.repositories.audit_log_repository import AuditLogRepository
from backend.app.services.audit_verifier import (
    AUDIT_CHAIN_ADVISORY_LOCK_ID,
    GENESIS_PREV_HASH,
    STANDARD_VERIFICATION_LIMITATIONS,
    AuditErrorType,
    AuditIntegrityError,
    AuditIntegrityVerifier,
    AuditVerificationResult,
    compute_audit_hash,
    compute_canonical_audit_json,
    compute_entry_hash_for_entity,
    normalize_value_for_canonical_json,
)


# ==============================================================================
# Helper Factories
# ==============================================================================

def create_audit_record(
    event_type: str = "CASE_CREATED",
    entity_type: AuditEntityType = AuditEntityType.CASE,
    entity_id: uuid.UUID = None,
    action: str = "CREATE",
    actor_type: AuditActorType = AuditActorType.ANALYST,
    actor_id: str = "analyst_1",
    correlation_id: str = "corr-100",
    client_ip: str = "127.0.0.1",
    payload: dict = None,
    event_timestamp: datetime = None,
    prev_hash: str = None,
    entry_hash: str = None,
) -> AuditLog:
    """Create an AuditLog entity instance with optional pre-computed hashes."""
    rec_id = uuid.uuid4()
    ts = event_timestamp or datetime.now(timezone.utc)
    log = AuditLog(
        id=rec_id,
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id or uuid.uuid4(),
        action=action,
        actor_type=actor_type,
        actor_id=actor_id,
        correlation_id=correlation_id,
        client_ip=client_ip,
        payload=payload if payload is not None else {"status": "OPEN", "priority": "HIGH"},
        event_timestamp=ts,
        created_at=ts,
        prev_hash=prev_hash,
        entry_hash=entry_hash,
    )
    return log


def build_valid_audit_chain(count: int, start_prev_hash: str = GENESIS_PREV_HASH) -> list[AuditLog]:
    """Construct an unbroken, cryptographically valid chain of AuditLog records."""
    records = []
    current_prev = start_prev_hash
    base_time = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)

    for i in range(count):
        ts = base_time + timedelta(seconds=i * 10)
        log = create_audit_record(
            event_type=f"EVENT_{i+1}",
            action=f"ACTION_{i+1}",
            actor_id=f"actor_{i%3}",
            payload={"step": i + 1, "metric": Decimal("123.45"), "active": True},
            event_timestamp=ts,
            prev_hash=current_prev,
        )
        log.entry_hash = compute_entry_hash_for_entity(log, current_prev)
        records.append(log)
        current_prev = log.entry_hash

    return records


# ==============================================================================
# 1. Canonical Serialization & Determinism Tests
# ==============================================================================

class TestCanonicalSerialization:
    """Validates deterministic JSON representation and SHA-256 hash sensitivity."""

    def test_normalize_value_primitives(self):
        assert normalize_value_for_canonical_json(None) is None
        assert normalize_value_for_canonical_json(42) == 42
        assert normalize_value_for_canonical_json("test") == "test"
        assert normalize_value_for_canonical_json(True) is True
        assert normalize_value_for_canonical_json(Decimal("99.95")) == "99.95"

    def test_normalize_value_uuid_and_datetime(self):
        uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert normalize_value_for_canonical_json(uid) == "12345678-1234-5678-1234-567812345678"

        dt_naive = datetime(2026, 10, 14, 12, 30, 0)
        assert normalize_value_for_canonical_json(dt_naive) == "2026-10-14T12:30:00Z"

        dt_tz = datetime(2026, 10, 14, 12, 30, 0, tzinfo=timezone.utc)
        assert normalize_value_for_canonical_json(dt_tz) == "2026-10-14T12:30:00Z"

    def test_normalize_value_nested_sorting(self):
        # Unsorted nested dicts should always produce identical sorted representations
        d1 = {"z": 1, "a": {"b": 2, "a": 1}}
        d2 = {"a": {"a": 1, "b": 2}, "z": 1}
        assert normalize_value_for_canonical_json(d1) == normalize_value_for_canonical_json(d2)

    def test_canonical_json_key_order_independence(self):
        rec_id = uuid.uuid4()
        ent_id = uuid.uuid4()
        ts = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)

        json1 = compute_canonical_audit_json(
            id=rec_id,
            event_type="LOGIN",
            entity_type=AuditEntityType.USER,
            entity_id=ent_id,
            action="AUTHENTICATE",
            actor_type=AuditActorType.ANALYST,
            actor_id="user-1",
            correlation_id="corr-1",
            client_ip="10.0.0.1",
            payload={"role": "ANALYST", "method": "PASSWORD"},
            event_timestamp=ts,
            prev_hash=GENESIS_PREV_HASH,
        )

        json2 = compute_canonical_audit_json(
            id=rec_id,
            event_type="LOGIN",
            entity_type="USER",
            entity_id=ent_id,
            action="AUTHENTICATE",
            actor_type="ANALYST",
            actor_id="user-1",
            correlation_id="corr-1",
            client_ip="10.0.0.1",
            payload={"method": "PASSWORD", "role": "ANALYST"},  # reversed keys
            event_timestamp=ts,
            prev_hash=GENESIS_PREV_HASH,
        )

        assert json1 == json2
        assert compute_audit_hash(
            id=rec_id,
            event_type="LOGIN",
            entity_type=AuditEntityType.USER,
            entity_id=ent_id,
            action="AUTHENTICATE",
            actor_type=AuditActorType.ANALYST,
            actor_id="user-1",
            correlation_id="corr-1",
            client_ip="10.0.0.1",
            payload={"role": "ANALYST", "method": "PASSWORD"},
            event_timestamp=ts,
            prev_hash=GENESIS_PREV_HASH,
        ) == compute_audit_hash(
            id=rec_id,
            event_type="LOGIN",
            entity_type="USER",
            entity_id=ent_id,
            action="AUTHENTICATE",
            actor_type="ANALYST",
            actor_id="user-1",
            correlation_id="corr-1",
            client_ip="10.0.0.1",
            payload={"method": "PASSWORD", "role": "ANALYST"},
            event_timestamp=ts,
            prev_hash=GENESIS_PREV_HASH,
        )


# ==============================================================================
# 2. Valid Chain Verification Tests
# ==============================================================================

class TestValidAuditChains:
    """Validates verification of genuine, untampered audit log chains."""

    def test_empty_sequence_is_valid(self):
        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records([])
        assert result.is_valid is True
        assert result.total_records == 0
        assert result.verified_records == 0
        assert result.legacy_records == 0
        assert len(result.errors) == 0

    def test_single_genesis_record_valid(self):
        chain = build_valid_audit_chain(1)
        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(chain)

        assert result.is_valid is True
        assert result.total_records == 1
        assert result.verified_records == 1
        assert result.first_chained_id == chain[0].id
        assert result.last_verified_id == chain[0].id
        assert result.last_verified_hash == chain[0].entry_hash
        assert len(result.errors) == 0

    def test_multi_record_unbroken_chain_valid(self):
        chain = build_valid_audit_chain(15)
        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(chain)

        assert result.is_valid is True
        assert result.total_records == 15
        assert result.verified_records == 15
        assert result.first_chained_id == chain[0].id
        assert result.last_verified_id == chain[-1].id
        assert result.last_verified_hash == chain[-1].entry_hash
        assert len(result.errors) == 0
        assert "AUDIT INTEGRITY VERIFIED: 15 chained records" in result.summary


# ==============================================================================
# 3. Tampering Detection Tests
# ==============================================================================

class TestTamperingDetection:
    """Validates that any alteration of audit fields or payload is instantly detected."""

    def test_detect_modified_payload(self):
        chain = build_valid_audit_chain(5)
        # Maliciously tamper with payload of record index 2
        chain[2].payload = {"step": 3, "metric": "999.99", "active": True}

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(chain)

        assert result.is_valid is False
        assert any(e.error_type == AuditErrorType.INVALID_HASH and e.record_id == chain[2].id for e in result.errors)

    def test_detect_modified_action(self):
        chain = build_valid_audit_chain(4)
        chain[1].action = "MALICIOUS_OVERRIDE"

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(chain)

        assert result.is_valid is False
        error = next(e for e in result.errors if e.record_id == chain[1].id)
        assert error.error_type == AuditErrorType.INVALID_HASH

    def test_detect_modified_actor(self):
        chain = build_valid_audit_chain(3)
        chain[0].actor_id = "attacker"

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(chain)

        assert result.is_valid is False
        assert result.errors[0].error_type == AuditErrorType.INVALID_HASH
        assert result.errors[0].record_id == chain[0].id

    def test_detect_modified_timestamp(self):
        chain = build_valid_audit_chain(3)
        chain[1].event_timestamp = chain[1].event_timestamp + timedelta(hours=1)

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(chain)

        assert result.is_valid is False
        assert any(e.error_type == AuditErrorType.INVALID_HASH and e.record_id == chain[1].id for e in result.errors)


# ==============================================================================
# 4. Broken Link & Chain Continuity Tests
# ==============================================================================

class TestBrokenLinkDetection:
    """Validates detection of severed links, broken parent hashes, and genesis mismatches."""

    def test_detect_genesis_mismatch(self):
        chain = build_valid_audit_chain(3)
        # Alter genesis prev_hash
        chain[0].prev_hash = "f" * 64
        # Recompute its hash so hash itself is valid, but prev_hash is not genesis
        chain[0].entry_hash = compute_entry_hash_for_entity(chain[0], chain[0].prev_hash)

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(chain)

        assert result.is_valid is False
        genesis_err = next(e for e in result.errors if e.error_type == AuditErrorType.GENESIS_MISMATCH)
        assert genesis_err.record_id == chain[0].id
        assert genesis_err.expected == GENESIS_PREV_HASH

    def test_detect_severed_middle_link(self):
        chain = build_valid_audit_chain(6)
        # Tamper with prev_hash of record 3 without changing preceding record's entry_hash
        chain[3].prev_hash = "a" * 64
        chain[3].entry_hash = compute_entry_hash_for_entity(chain[3], chain[3].prev_hash)

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(chain)

        assert result.is_valid is False
        broken_link = next(e for e in result.errors if e.error_type == AuditErrorType.BROKEN_LINK)
        assert broken_link.record_id == chain[3].id
        assert broken_link.expected == chain[2].entry_hash

    def test_detect_deleted_record(self):
        chain = build_valid_audit_chain(5)
        # Delete record index 2 (simulating middle record deletion)
        tampered_chain = [chain[0], chain[1], chain[3], chain[4]]

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(tampered_chain)

        assert result.is_valid is False
        # Record 3 will have prev_hash pointing to deleted record 2, not record 1
        broken_err = next(e for e in result.errors if e.error_type == AuditErrorType.BROKEN_LINK)
        assert broken_err.record_id == chain[3].id
        assert broken_err.expected == chain[1].entry_hash
        assert broken_err.actual == chain[2].entry_hash

    def test_detect_reordered_records(self):
        chain = build_valid_audit_chain(4)
        # Swap records 1 and 2
        swapped_chain = [chain[0], chain[2], chain[1], chain[3]]

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(swapped_chain)

        assert result.is_valid is False
        broken_errors = [e for e in result.errors if e.error_type == AuditErrorType.BROKEN_LINK]
        assert len(broken_errors) >= 1


# ==============================================================================
# 5. Legacy Record & Unchained Injection Tests
# ==============================================================================

class TestLegacyAndUnchainedRecords:
    """Validates legacy record acknowledgment and detection of illegal unchained records."""

    def test_legacy_records_acknowledged_at_chain_head(self):
        # 3 legacy records with null hashes
        legacy_records = [
            create_audit_record(event_type=f"LEGACY_{i}", prev_hash=None, entry_hash=None)
            for i in range(3)
        ]
        # Followed by 4 chained records
        chained_records = build_valid_audit_chain(4)
        combined = legacy_records + chained_records

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(combined)

        assert result.is_valid is True
        assert result.total_records == 7
        assert result.legacy_records == 3
        assert result.verified_records == 4
        assert result.first_chained_id == chained_records[0].id
        assert len(result.errors) == 0

    def test_illegal_unchained_record_after_chain_started(self):
        chained_records = build_valid_audit_chain(3)
        # Unchained injection in the middle of a chained sequence
        illegal_rec = create_audit_record(event_type="UNAUTH_EVENT", prev_hash=None, entry_hash=None)
        more_chained = build_valid_audit_chain(2, start_prev_hash=chained_records[-1].entry_hash)

        sequence = chained_records + [illegal_rec] + more_chained

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(sequence)

        assert result.is_valid is False
        illegal_err = next(e for e in result.errors if e.error_type == AuditErrorType.ILLEGAL_UNCHAINED_RECORD)
        assert illegal_err.record_id == illegal_rec.id

    def test_duplicate_hash_detected(self):
        chain = build_valid_audit_chain(3)
        # Inject duplicate record
        duplicate = chain[1]
        sequence = [chain[0], chain[1], duplicate, chain[2]]

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(sequence)

        assert result.is_valid is False
        dup_err = next(e for e in result.errors if e.error_type == AuditErrorType.DUPLICATE_HASH)
        assert dup_err.actual == chain[1].entry_hash


# ==============================================================================
# 6. Repository Automatic Hash Chaining Tests
# ==============================================================================

class TestAuditLogRepositoryChaining:
    """Validates that AuditLogRepository automatically populates unbroken hash chains."""

    @pytest.mark.asyncio
    async def test_repository_add_computes_genesis_hash(self):
        mock_session = MagicMock()
        mock_session.new = set()
        repo = AuditLogRepository(mock_session)

        log = create_audit_record(event_type="LOGIN", prev_hash=None, entry_hash=None)
        assert log.prev_hash is None
        assert log.entry_hash is None

        added = await repo.add(log)

        assert added.prev_hash == GENESIS_PREV_HASH
        assert added.entry_hash is not None
        assert len(added.entry_hash) == 64
        # Verify stored entry_hash matches cryptographic recomputation
        expected = compute_entry_hash_for_entity(added, GENESIS_PREV_HASH)
        assert added.entry_hash == expected
        mock_session.add.assert_called_once_with(added)

    @pytest.mark.asyncio
    async def test_repository_add_many_chains_sequential_records(self):
        mock_session = MagicMock()
        mock_session.new = set()
        repo = AuditLogRepository(mock_session)

        logs = [
            create_audit_record(event_type=f"EVENT_{i}", prev_hash=None, entry_hash=None)
            for i in range(5)
        ]

        added_list = await repo.add_many(logs)

        assert len(added_list) == 5
        assert added_list[0].prev_hash == GENESIS_PREV_HASH

        for i in range(1, 5):
            assert added_list[i].prev_hash == added_list[i - 1].entry_hash

        # Run AuditIntegrityVerifier on the output of add_many
        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(added_list)
        assert result.is_valid is True
        assert result.verified_records == 5

    @pytest.mark.asyncio
    async def test_repository_sequential_adds_preserve_chain_via_session_staging(self):
        # When add() is called repeatedly on the same session, session.new links them
        session_set = set()
        mock_session = MagicMock()
        mock_session.new = session_set

        def fake_add(obj):
            session_set.add(obj)

        mock_session.add.side_effect = fake_add

        repo = AuditLogRepository(mock_session)

        log1 = create_audit_record(event_type="STEP_1")
        log2 = create_audit_record(event_type="STEP_2")
        log3 = create_audit_record(event_type="STEP_3")

        await repo.add(log1)
        await repo.add(log2)
        await repo.add(log3)

        assert log1.prev_hash == GENESIS_PREV_HASH
        assert log2.prev_hash == log1.entry_hash
        assert log3.prev_hash == log2.entry_hash

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records([log1, log2, log3])
        assert result.is_valid is True
        assert result.verified_records == 3


# ==============================================================================
# 7. Concurrent Writes & Anti-Forking Simulation Tests
# ==============================================================================

class TestConcurrentAuditWrites:
    """Validates that concurrent audit writes produce a single, unbroken, non-forked chain."""

    @pytest.mark.asyncio
    async def test_concurrent_appends_serialize_without_forking(self):
        """
        Simulate 20 coroutines concurrently appending audit entries to the repository.
        Validates that no two records receive the same prev_hash (no chain forks)
        and all records form a single verified unbroken chain.
        """
        session_set = set()
        mock_session = MagicMock()
        mock_session.new = session_set

        def fake_add(obj):
            session_set.add(obj)

        mock_session.add.side_effect = fake_add
        repo = AuditLogRepository(mock_session)

        records_to_add = [
            create_audit_record(event_type=f"CONCURRENT_{i}", action=f"ACTION_{i}")
            for i in range(20)
        ]

        # Execute all 20 appends concurrently
        tasks = [repo.add(rec) for rec in records_to_add]
        added_records = await asyncio.gather(*tasks)

        # 1. Verify every record has a valid 64-char hex hash
        for rec in added_records:
            assert rec.entry_hash is not None
            assert len(rec.entry_hash) == 64
            assert rec.prev_hash is not None

        # 2. Sort by creation order and verify linear chain continuity
        # In a serialized chain with 1 genesis entry, exactly one record has prev_hash == GENESIS_PREV_HASH
        genesis_records = [r for r in added_records if r.prev_hash == GENESIS_PREV_HASH]
        assert len(genesis_records) == 1, "Expected exactly 1 genesis record; multiple indicates a chain fork!"

        # 3. Verify no duplicate prev_hash exists (no forks)
        prev_hashes = [r.prev_hash for r in added_records]
        assert len(prev_hashes) == len(set(prev_hashes)), "Duplicate prev_hash detected: chain forked during concurrent writes!"

        # 4. Assemble in sequence and verify full cryptographic integrity
        chain_map = {r.prev_hash: r for r in added_records}
        ordered_chain = []
        curr_prev = GENESIS_PREV_HASH
        while curr_prev in chain_map:
            rec = chain_map[curr_prev]
            ordered_chain.append(rec)
            curr_prev = rec.entry_hash

        assert len(ordered_chain) == 20

        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records(ordered_chain)
        assert result.is_valid is True
        assert result.verified_records == 20
        assert len(result.errors) == 0


# ==============================================================================
# 8. Verifier Database Streaming & Limitations Tests
# ==============================================================================

class TestVerifierDatabaseStreaming:
    """Validates database streaming verification and limitation disclosure."""

    @pytest.mark.asyncio
    async def test_verify_database_streams_chronological_records(self):
        chain = build_valid_audit_chain(4)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = chain
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        verifier = AuditIntegrityVerifier()
        result = await verifier.verify_database(mock_session, limit=100)

        assert result.is_valid is True
        assert result.total_records == 4
        assert result.verified_records == 4
        mock_session.execute.assert_awaited_once()

    def test_standard_limitations_included_in_result(self):
        verifier = AuditIntegrityVerifier()
        result = verifier.verify_records([])

        assert len(result.limitations) == 4
        assert any("Legacy Records" in lim for lim in result.limitations)
        assert any("Tail Truncation" in lim for lim in result.limitations)
        assert any("Complete Table Destruction" in lim for lim in result.limitations)
        assert any("Timestamp Monotonicity" in lim for lim in result.limitations)
