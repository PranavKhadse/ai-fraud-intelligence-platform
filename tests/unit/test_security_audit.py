"""
Unit and Integration Tests for Phase 15.5: Security-Event Auditing & Audit Verification Access.

Tests:
1. Secret stripping and payload sanitization (passwords, tokens, raw keys, secrets).
2. Security event recording with automatic cryptographic hash chaining.
3. Login success audit event recording.
4. Login failure audit event recording (invalid credentials and inactive user).
5. Administrator user creation security audit event.
6. Administrator API-key creation security audit event.
7. 403 Forbidden access denied audit event recording in RBAC dependencies.
8. Admin-only audit verification endpoint:
   - 401 Unauthorized for unauthenticated requests.
   - 403 Forbidden for ANALYST and API_CLIENT identities.
   - 200 OK for ADMIN identity returning comprehensive verification report.
   - Tampering detection reported in verification endpoint response.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.security import (
    ActorContext,
    create_access_token,
    generate_api_key,
    hash_password,
    require_role,
)
from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import AuditActorType, AuditEntityType, UserRole
from backend.app.db.models.user import User
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.repositories.audit_log_repository import AuditLogRepository
from backend.app.repositories.user_repository import UserRepository
from backend.app.services.audit_verifier import (
    GENESIS_PREV_HASH,
    AuditIntegrityVerifier,
    compute_entry_hash_for_entity,
)
from backend.app.services.security_audit import (
    SENSITIVE_AUDIT_KEYS,
    record_security_event,
    sanitize_audit_payload,
)
from backend.app.services.unit_of_work import FraudPersistenceUnitOfWork, get_unit_of_work


# ==============================================================================
# 1. Payload Sanitization Tests
# ==============================================================================

class TestPayloadSanitization:
    """Validates that credentials, tokens, and secrets are strictly stripped from audit logs."""

    def test_sanitize_strips_sensitive_keys(self):
        raw_payload = {
            "username": "alice",
            "password": "SuperSecretPassword123!",
            "hashed_password": "$2b$12$eIX7K..fakehash",
            "raw_key": "live_apikey_xyz789",
            "secret": "top_secret_value",
            "token": "eyJh...jwt_token",
            "access_token": "access_xyz",
            "refresh_token": "refresh_abc",
            "role": "ANALYST",
            "safe_metadata": {"action": "LOGIN", "status": "OK"},
        }

        cleaned = sanitize_audit_payload(raw_payload)

        # Sensitive keys MUST be removed
        for key in SENSITIVE_AUDIT_KEYS:
            assert key not in cleaned

        # Safe keys MUST be preserved
        assert cleaned["username"] == "alice"
        assert cleaned["role"] == "ANALYST"
        assert cleaned["safe_metadata"] == {"action": "LOGIN", "status": "OK"}

    def test_sanitize_nested_dictionaries_and_lists(self):
        nested = {
            "outer": {
                "user_password": "leak_attempt",
                "valid": 123,
            },
            "items": [
                {"secret_token": "leak2", "name": "item1"},
                {"name": "item2"},
            ],
        }

        cleaned = sanitize_audit_payload(nested)

        assert "user_password" not in cleaned["outer"]
        assert cleaned["outer"]["valid"] == 123
        assert "secret_token" not in cleaned["items"][0]
        assert cleaned["items"][0]["name"] == "item1"
        assert cleaned["items"][1]["name"] == "item2"


# ==============================================================================
# 2. Security Event Recording & Hash Chaining Tests
# ==============================================================================

class TestSecurityEventRecording:
    """Validates that security events are correctly appended with cryptographic hash chaining."""

    @pytest.mark.asyncio
    async def test_record_security_event_creates_valid_chained_log(self):
        session_set = set()
        mock_session = MagicMock(spec=AsyncSession)
        mock_session.new = session_set

        def fake_add(obj):
            session_set.add(obj)

        mock_session.add.side_effect = fake_add
        mock_session.commit = AsyncMock()

        log = await record_security_event(
            session=mock_session,
            event_type="AUTH_LOGIN_SUCCESS",
            action="LOGIN",
            actor_type=AuditActorType.ANALYST,
            actor_id="analyst_jane",
            entity_type=AuditEntityType.USER,
            entity_id=uuid.uuid4(),
            correlation_id="corr-999",
            client_ip="192.168.1.100",
            payload={"username": "analyst_jane", "role": "ANALYST", "password": "MUST_NOT_APPEAR"},
            commit=True,
        )

        assert log is not None
        assert log.event_type == "AUTH_LOGIN_SUCCESS"
        assert log.actor_id == "analyst_jane"
        assert log.prev_hash == GENESIS_PREV_HASH
        assert log.entry_hash is not None
        assert len(log.entry_hash) == 64
        # Verify secret was sanitized
        assert "password" not in log.payload
        assert log.payload["username"] == "analyst_jane"

        # Verify entry_hash is mathematically valid
        recomputed = compute_entry_hash_for_entity(log, GENESIS_PREV_HASH)
        assert log.entry_hash == recomputed
        mock_session.commit.assert_awaited_once()


# ==============================================================================
# 3. Authentication Endpoints Audit Integration Tests
# ==============================================================================

class TestAuthEndpointsAuditIntegration:
    """Validates that login and provisioning endpoints record security audit events."""

    @pytest.fixture
    def test_client(self):
        return TestClient(app)

    @pytest.fixture
    def active_analyst_user(self):
        return User(
            id=uuid.uuid4(),
            username="analyst_bob",
            email="bob@example.com",
            hashed_password=hash_password("ValidPassword123!"),
            role=UserRole.ANALYST,
            is_active=True,
            is_superuser=False,
        )

    @pytest.fixture
    def inactive_user(self):
        return User(
            id=uuid.uuid4(),
            username="inactive_dan",
            email="dan@example.com",
            hashed_password=hash_password("ValidPassword123!"),
            role=UserRole.ANALYST,
            is_active=False,
            is_superuser=False,
        )

    @pytest.fixture
    def admin_user(self):
        return User(
            id=uuid.uuid4(),
            username="admin_super",
            email="admin@example.com",
            hashed_password=hash_password("AdminPassword123!"),
            role=UserRole.ADMIN,
            is_active=True,
            is_superuser=False,
        )

    def test_login_success_records_audit_event(self, test_client, active_analyst_user):
        mock_uow = MagicMock(spec=FraudPersistenceUnitOfWork)
        mock_users_repo = MagicMock(spec=UserRepository)
        mock_audit_repo = MagicMock(spec=AuditLogRepository)
        mock_uow.users = mock_users_repo
        mock_uow.audit_logs = mock_audit_repo
        mock_uow.session = MagicMock(spec=AsyncSession)
        mock_uow.session.new = set()

        mock_users_repo.get_by_username = AsyncMock(return_value=active_analyst_user)
        mock_users_repo.update_last_login = AsyncMock()
        mock_uow.commit = AsyncMock()

        app.dependency_overrides[get_unit_of_work] = lambda: mock_uow

        try:
            resp = test_client.post(
                "/api/v1/auth/login",
                json={"username": "analyst_bob", "password": "ValidPassword123!"},
            )

            assert resp.status_code == status.HTTP_200_OK
            mock_uow.commit.assert_awaited_once()

            # Verify audit log was attached to session
            staged_audit = [call[0][0] for call in mock_uow.session.add.call_args_list]
            login_events = [e for e in staged_audit if e.event_type == "AUTH_LOGIN_SUCCESS"]
            assert len(login_events) == 1
            assert login_events[0].actor_id == "analyst_bob"
            assert login_events[0].entity_id == active_analyst_user.id
            assert "password" not in login_events[0].payload
        finally:
            app.dependency_overrides.pop(get_unit_of_work, None)

    def test_login_failure_invalid_credentials_records_audit_event(self, test_client, active_analyst_user):
        mock_uow = MagicMock(spec=FraudPersistenceUnitOfWork)
        mock_users_repo = MagicMock(spec=UserRepository)
        mock_uow.users = mock_users_repo
        mock_uow.session = MagicMock(spec=AsyncSession)
        mock_uow.session.new = set()
        mock_uow.session.commit = AsyncMock()

        # Wrong password
        mock_users_repo.get_by_username = AsyncMock(return_value=active_analyst_user)

        app.dependency_overrides[get_unit_of_work] = lambda: mock_uow

        try:
            resp = test_client.post(
                "/api/v1/auth/login",
                json={"username": "analyst_bob", "password": "WrongPassword999!"},
            )

            assert resp.status_code == status.HTTP_401_UNAUTHORIZED
            # Audit log must be committed on failure
            mock_uow.session.commit.assert_awaited_once()

            staged_audit = [call[0][0] for call in mock_uow.session.add.call_args_list]
            failure_events = [e for e in staged_audit if e.event_type == "AUTH_LOGIN_FAILURE"]
            assert len(failure_events) == 1
            assert failure_events[0].payload["reason"] == "INVALID_CREDENTIALS"
            assert "password" not in failure_events[0].payload
        finally:
            app.dependency_overrides.pop(get_unit_of_work, None)

    def test_login_failure_inactive_user_records_audit_event(self, test_client, inactive_user):
        mock_uow = MagicMock(spec=FraudPersistenceUnitOfWork)
        mock_users_repo = MagicMock(spec=UserRepository)
        mock_uow.users = mock_users_repo
        mock_uow.session = MagicMock(spec=AsyncSession)
        mock_uow.session.new = set()
        mock_uow.session.commit = AsyncMock()

        mock_users_repo.get_by_username = AsyncMock(return_value=inactive_user)

        app.dependency_overrides[get_unit_of_work] = lambda: mock_uow

        try:
            resp = test_client.post(
                "/api/v1/auth/login",
                json={"username": "inactive_dan", "password": "ValidPassword123!"},
            )

            assert resp.status_code == status.HTTP_401_UNAUTHORIZED
            mock_uow.session.commit.assert_awaited_once()

            staged_audit = [call[0][0] for call in mock_uow.session.add.call_args_list]
            failure_events = [e for e in staged_audit if e.event_type == "AUTH_LOGIN_FAILURE"]
            assert len(failure_events) == 1
            assert failure_events[0].payload["reason"] == "ACCOUNT_INACTIVE"
        finally:
            app.dependency_overrides.pop(get_unit_of_work, None)

    def test_admin_create_user_records_audit_event(self, test_client, admin_user):
        mock_uow = MagicMock(spec=FraudPersistenceUnitOfWork)
        mock_users_repo = MagicMock(spec=UserRepository)
        mock_uow.users = mock_users_repo
        mock_uow.session = MagicMock(spec=AsyncSession)
        mock_uow.session.new = set()
        mock_uow.commit = AsyncMock()

        mock_users_repo.get_by_username = AsyncMock(return_value=None)
        mock_users_repo.get_by_email = AsyncMock(return_value=None)
        mock_users_repo.get_by_id = AsyncMock(return_value=admin_user)

        admin_token = create_access_token(subject=admin_user.id, role=UserRole.ADMIN, username=admin_user.username)

        from backend.app.core.security import require_admin_user
        app.dependency_overrides[require_admin_user] = lambda: admin_user
        app.dependency_overrides[get_unit_of_work] = lambda: mock_uow
        app.dependency_overrides[get_db_session] = lambda: mock_uow.session

        try:
            resp = test_client.post(
                "/api/v1/auth/users",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={
                    "username": "new_analyst",
                    "email": "new@example.com",
                    "password": "SecurePassword123!",
                    "role": "ANALYST",
                },
            )

            assert resp.status_code == status.HTTP_201_CREATED
            mock_uow.commit.assert_awaited_once()

            staged_audit = [call[0][0] for call in mock_uow.session.add.call_args_list if isinstance(call[0][0], AuditLog)]
            user_events = [e for e in staged_audit if e.event_type == "USER_CREATED"]
            assert len(user_events) == 1
            assert user_events[0].actor_id == admin_user.username
            assert user_events[0].payload["created_username"] == "new_analyst"
            assert user_events[0].payload["role"] == "ANALYST"
            assert "password" not in user_events[0].payload
        finally:
            app.dependency_overrides.pop(require_admin_user, None)
            app.dependency_overrides.pop(get_unit_of_work, None)
            app.dependency_overrides.pop(get_db_session, None)

    def test_admin_create_api_key_records_audit_event(self, test_client, admin_user):
        mock_uow = MagicMock(spec=FraudPersistenceUnitOfWork)
        mock_api_keys_repo = MagicMock()
        mock_uow.api_keys = mock_api_keys_repo
        mock_uow.session = MagicMock(spec=AsyncSession)
        mock_uow.session.new = set()
        mock_uow.commit = AsyncMock()

        from backend.app.core.security import require_admin_user
        app.dependency_overrides[require_admin_user] = lambda: admin_user
        app.dependency_overrides[get_unit_of_work] = lambda: mock_uow
        app.dependency_overrides[get_db_session] = lambda: mock_uow.session

        try:
            resp = test_client.post(
                "/api/v1/auth/api-keys",
                headers={"Authorization": "Bearer dummy_token"},
                json={
                    "name": "Integration Service Key",
                    "client_id": "service_xyz",
                    "expires_in_days": 30,
                },
            )

            assert resp.status_code == status.HTTP_201_CREATED
            mock_uow.commit.assert_awaited_once()

            staged_audit = [call[0][0] for call in mock_uow.session.add.call_args_list if isinstance(call[0][0], AuditLog)]
            key_events = [e for e in staged_audit if e.event_type == "API_KEY_CREATED"]
            assert len(key_events) == 1
            assert key_events[0].actor_id == admin_user.username
            assert key_events[0].payload["client_id"] == "service_xyz"
            assert "raw_key" not in key_events[0].payload
            assert "hashed_key" not in key_events[0].payload
        finally:
            app.dependency_overrides.pop(require_admin_user, None)
            app.dependency_overrides.pop(get_unit_of_work, None)
            app.dependency_overrides.pop(get_db_session, None)


# ==============================================================================
# 4. RBAC Denied Authorization (403) Security Event Tests
# ==============================================================================

class TestRBACAccessDeniedAudit:
    """Validates that 403 Forbidden authorization rejections are logged as audit events."""

    @pytest.mark.asyncio
    async def test_require_role_denial_records_security_event(self):
        mock_session = MagicMock(spec=AsyncSession)
        mock_session.new = set()
        mock_session.commit = AsyncMock()

        checker = require_role(AuditActorType.ADMIN)
        forbidden_actor = ActorContext(
            actor_id="analyst_attacker",
            actor_role=AuditActorType.ANALYST,
            correlation_id="corr-deny",
            client_ip="10.0.0.99",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/api/v1/cases/batch-export"
        mock_request.method = "POST"

        with pytest.raises(HTTPException) as exc_info:
            await checker(actor=forbidden_actor, request=mock_request, session=mock_session)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        mock_session.commit.assert_awaited_once()

        staged_audit = [call[0][0] for call in mock_session.add.call_args_list if isinstance(call[0][0], AuditLog)]
        denied_events = [e for e in staged_audit if e.event_type == "AUTHZ_ACCESS_DENIED"]
        assert len(denied_events) == 1
        assert denied_events[0].actor_id == "analyst_attacker"
        assert denied_events[0].payload["attempted_role"] == "ANALYST"
        assert "ADMIN" in denied_events[0].payload["required_roles"]
        assert denied_events[0].payload["path"] == "/api/v1/cases/batch-export"


# ==============================================================================
# 5. Audit Verification Admin Endpoint Tests
# ==============================================================================

class TestAuditVerificationEndpoint:
    """Validates the authenticated, ADMIN-only /api/v1/audit/verify endpoint."""

    @pytest.fixture
    def test_client(self):
        return TestClient(app)

    @pytest.fixture
    def admin_user(self):
        return User(
            id=uuid.uuid4(),
            username="admin_auditor",
            email="auditor@example.com",
            hashed_password=hash_password("AdminPass123!"),
            role=UserRole.ADMIN,
            is_active=True,
            is_superuser=False,
        )

    @pytest.fixture
    def analyst_user(self):
        return User(
            id=uuid.uuid4(),
            username="analyst_viewer",
            email="viewer@example.com",
            hashed_password=hash_password("AnalystPass123!"),
            role=UserRole.ANALYST,
            is_active=True,
            is_superuser=False,
        )

    def test_verify_unauthenticated_fails_with_401(self, test_client):
        resp = test_client.get("/api/v1/audit/verify")
        assert resp.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_analyst_fails_with_403(self, test_client, analyst_user):
        token = create_access_token(subject=analyst_user.id, role=UserRole.ANALYST, username=analyst_user.username)
        resp = test_client.get(
            "/api/v1/audit/verify",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == status.HTTP_403_FORBIDDEN
        assert "Forbidden" in resp.json()["detail"]

    def test_verify_api_client_fails_with_403(self, test_client, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", True)
        # Machine API client header or dev header
        resp = test_client.get(
            "/api/v1/audit/verify",
            headers={"X-Actor-ID": "machine_client", "X-Actor-Role": "API_CLIENT"},
        )
        assert resp.status_code == status.HTTP_403_FORBIDDEN

    def test_verify_admin_returns_valid_verification_report(self, test_client, admin_user):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.new = set()
        mock_session.commit = AsyncMock()

        # Build 3 valid chained records
        base_time = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
        records = []
        prev_h = GENESIS_PREV_HASH
        for i in range(3):
            r = AuditLog(
                id=uuid.uuid4(),
                event_type=f"EVENT_{i}",
                entity_type=AuditEntityType.USER,
                action="TEST",
                actor_type=AuditActorType.SYSTEM,
                actor_id="test",
                event_timestamp=base_time + timedelta(seconds=i),
                created_at=base_time + timedelta(seconds=i),
                prev_hash=prev_h,
            )
            r.entry_hash = compute_entry_hash_for_entity(r, prev_h)
            records.append(r)
            prev_h = r.entry_hash

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        admin_token = create_access_token(subject=admin_user.id, role=UserRole.ADMIN, username=admin_user.username)

        app.dependency_overrides[get_db_session] = lambda: mock_session

        try:
            resp = test_client.get(
                "/api/v1/audit/verify?limit=100",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

            assert resp.status_code == status.HTTP_200_OK
            data = resp.json()
            assert data["is_valid"] is True
            assert data["total_records"] == 3
            assert data["verified_records"] == 3
            assert data["legacy_records"] == 0
            assert len(data["errors"]) == 0
            assert data["last_verified_hash"] == records[-1].entry_hash
            assert "limitations" in data
            assert len(data["limitations"]) > 0
            assert "AUDIT INTEGRITY VERIFIED" in data["summary"]
        finally:
            app.dependency_overrides.pop(get_db_session, None)

    def test_verify_admin_detects_tampered_record(self, test_client, admin_user):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.new = set()
        mock_session.commit = AsyncMock()

        # Build records with a tampered middle record
        base_time = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
        r1 = AuditLog(
            id=uuid.uuid4(),
            event_type="EVENT_1",
            entity_type=AuditEntityType.USER,
            action="TEST",
            actor_type=AuditActorType.SYSTEM,
            event_timestamp=base_time,
            created_at=base_time,
            prev_hash=GENESIS_PREV_HASH,
        )
        r1.entry_hash = compute_entry_hash_for_entity(r1, GENESIS_PREV_HASH)

        r2 = AuditLog(
            id=uuid.uuid4(),
            event_type="EVENT_2",
            entity_type=AuditEntityType.USER,
            action="ALTERED_ACTION",  # Tampered!
            actor_type=AuditActorType.SYSTEM,
            event_timestamp=base_time + timedelta(seconds=1),
            created_at=base_time + timedelta(seconds=1),
            prev_hash=r1.entry_hash,
            entry_hash="bad_hash" * 4,
        )

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [r1, r2]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        admin_token = create_access_token(subject=admin_user.id, role=UserRole.ADMIN, username=admin_user.username)

        app.dependency_overrides[get_db_session] = lambda: mock_session

        try:
            resp = test_client.get(
                "/api/v1/audit/verify",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

            assert resp.status_code == status.HTTP_200_OK
            data = resp.json()
            assert data["is_valid"] is False
            assert len(data["errors"]) >= 1
            assert any(e["error_type"] == "INVALID_HASH" for e in data["errors"])
        finally:
            app.dependency_overrides.pop(get_db_session, None)
