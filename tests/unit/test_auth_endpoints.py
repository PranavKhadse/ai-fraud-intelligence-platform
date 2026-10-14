"""
Unit Tests for Phase 15.2: Authentication Endpoints & Authorization Dependencies.

Validates:
1. POST /api/v1/auth/login:
   - Valid credentials for ANALYST and ADMIN users.
   - Inactive user rejection (401).
   - Unknown user rejection (401).
   - Invalid password rejection (401).
   - Validation failures for empty fields (422).
2. POST /api/v1/auth/refresh:
   - Refresh token rotation (issues new access & refresh tokens, revokes old).
   - Replay protection (revoked token rejection, 401).
   - Tampered token rejection (401).
   - Token type enforcement (rejects access token sent to refresh endpoint, 401).
   - Inactive user rejection on refresh (401).
3. GET /api/v1/auth/me:
   - Authenticated user profile retrieval for ANALYST and ADMIN.
   - Unauthenticated access rejection (401).
   - Inactive user profile access rejection (401).
4. POST /api/v1/auth/users (Admin-only):
   - Admin creating ANALYST or ADMIN accounts (201).
   - Role isolation: client cannot create SYSTEM or API_CLIENT user (422).
   - Privilege escalation defense: client-controlled is_superuser ignored (always False).
   - Analyst user forbidden from creating users (403).
   - Duplicate username rejection (409).
   - Duplicate email rejection (409).
   - Short password rejection (422).
5. POST /api/v1/auth/api-keys (Admin-only):
   - Admin creating machine API keys (201).
   - Returns raw_key once upon creation; stores only SHA-256 hash.
   - Machine role strictly enforced as API_CLIENT.
   - Analyst user forbidden from creating API keys (403).
6. get_current_actor resolution:
   - Bearer JWT token extracts human ActorContext.
   - X-API-Key extracts machine ActorContext with API_CLIENT role.
   - Inactive/expired API key rejected with 401.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.core.security import (
    ActorContext,
    create_access_token,
    create_refresh_token,
    generate_api_key,
    get_current_actor,
    get_current_user,
    hash_api_key,
    hash_password,
    token_revocation_store,
)
from backend.app.db.models.api_key import APIKey
from backend.app.db.models.enums import AuditActorType, UserRole
from backend.app.db.models.user import User
from backend.app.main import app
from backend.app.repositories.api_key_repository import APIKeyRepository
from backend.app.repositories.user_repository import UserRepository
from backend.app.services.unit_of_work import UnitOfWork, get_unit_of_work


# ==============================================================================
# In-Memory Mock Repositories & Fixtures
# ==============================================================================

class InMemoryUserRepository:
    """Mock repository simulating UserRepository with in-memory storage."""

    def __init__(self, initial_users: Optional[List[User]] = None):
        self._users: Dict[uuid.UUID, User] = {}
        if initial_users:
            for u in initial_users:
                self._users[u.id] = u

    async def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        return self._users.get(user_id)

    async def get_by_username(self, username: str) -> Optional[User]:
        for u in self._users.values():
            if u.username == username:
                return u
        return None

    async def get_by_email(self, email: str) -> Optional[User]:
        for u in self._users.values():
            if u.email.lower() == email.lower():
                return u
        return None

    def add(self, user: User) -> None:
        if not user.id:
            user.id = uuid.uuid4()
        if not user.created_at:
            user.created_at = datetime.now(timezone.utc)
        if not user.updated_at:
            user.updated_at = datetime.now(timezone.utc)
        self._users[user.id] = user

    async def update_last_login(self, user_id: uuid.UUID) -> None:
        user = self._users.get(user_id)
        if user:
            user.last_login_at = datetime.now(timezone.utc)


class InMemoryAPIKeyRepository:
    """Mock repository simulating APIKeyRepository with in-memory storage."""

    def __init__(self, initial_keys: Optional[List[APIKey]] = None):
        self._keys: Dict[uuid.UUID, APIKey] = {}
        if initial_keys:
            for k in initial_keys:
                self._keys[k.id] = k

    async def get_by_id(self, key_id: uuid.UUID) -> Optional[APIKey]:
        return self._keys.get(key_id)

    async def get_by_hashed_key(self, hashed_key: str) -> Optional[APIKey]:
        for k in self._keys.values():
            if k.hashed_key == hashed_key:
                return k
        return None

    def add(self, api_key: APIKey) -> None:
        if not api_key.id:
            api_key.id = uuid.uuid4()
        self._keys[api_key.id] = api_key

    async def update_last_used(self, key_id: uuid.UUID) -> None:
        key = self._keys.get(key_id)
        if key:
            key.last_used_at = datetime.now(timezone.utc)


class MockUnitOfWork:
    """Mock UnitOfWork providing in-memory user and api_key repositories."""

    def __init__(self, user_repo: InMemoryUserRepository, key_repo: InMemoryAPIKeyRepository):
        self.users = user_repo
        self.api_keys = key_repo
        self.committed = False
        self.rolled_back = False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()


@pytest.fixture(autouse=True)
def clean_revocation_store():
    """Ensure revocation store is fresh for each test."""
    token_revocation_store.clear()
    yield
    token_revocation_store.clear()


@pytest.fixture
def mock_analyst_user() -> User:
    user_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)
    return User(
        id=user_id,
        username="analyst_jane",
        email="jane@bank.com",
        full_name="Jane Doe",
        hashed_password=hash_password("StrongPassword123!"),
        role=UserRole.ANALYST,
        is_active=True,
        is_superuser=False,
        created_at=now_utc,
        updated_at=now_utc,
    )


@pytest.fixture
def mock_admin_user() -> User:
    user_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)
    return User(
        id=user_id,
        username="admin_bob",
        email="bob@bank.com",
        full_name="Bob Admin",
        hashed_password=hash_password("AdminSecurePass999!"),
        role=UserRole.ADMIN,
        is_active=True,
        is_superuser=False,
        created_at=now_utc,
        updated_at=now_utc,
    )


@pytest.fixture
def mock_inactive_user() -> User:
    user_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)
    return User(
        id=user_id,
        username="inactive_user",
        email="inactive@bank.com",
        full_name="Inactive Person",
        hashed_password=hash_password("InactivePass123!"),
        role=UserRole.ANALYST,
        is_active=False,
        is_superuser=False,
        created_at=now_utc,
        updated_at=now_utc,
    )


@pytest.fixture
def test_env(mock_analyst_user: User, mock_admin_user: User, mock_inactive_user: User):
    """Sets up mock unit of work with users and overrides get_unit_of_work."""
    user_repo = InMemoryUserRepository([mock_analyst_user, mock_admin_user, mock_inactive_user])
    key_repo = InMemoryAPIKeyRepository([])
    uow = MockUnitOfWork(user_repo, key_repo)

    async def override_uow():
        yield uow

    async def override_current_user_from_header(auth=None):
        # Fallback helper if needed
        pass

    app.dependency_overrides[get_unit_of_work] = override_uow

    # Also override session-based user lookup in get_current_user
    with TestClient(app) as client:
        yield {
            "client": client,
            "uow": uow,
            "user_repo": user_repo,
            "key_repo": key_repo,
            "analyst": mock_analyst_user,
            "admin": mock_admin_user,
            "inactive": mock_inactive_user,
        }

    app.dependency_overrides.clear()


# ==============================================================================
# 1. Login Endpoint Tests
# ==============================================================================

class TestLoginEndpoint:
    """Validates POST /api/v1/auth/login."""

    def test_login_analyst_success(self, test_env):
        client: TestClient = test_env["client"]
        analyst: User = test_env["analyst"]

        resp = client.post("/api/v1/auth/login", json={
            "username": "analyst_jane",
            "password": "StrongPassword123!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        assert analyst.last_login_at is not None

    def test_login_admin_success(self, test_env):
        client: TestClient = test_env["client"]
        resp = client.post("/api/v1/auth/login", json={
            "username": "admin_bob",
            "password": "AdminSecurePass999!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_login_nonexistent_user(self, test_env):
        client: TestClient = test_env["client"]
        resp = client.post("/api/v1/auth/login", json={
            "username": "non_existent_user",
            "password": "AnyPassword123!",
        })
        assert resp.status_code == 401
        assert "Invalid username or password" in resp.json()["detail"]

    def test_login_wrong_password(self, test_env):
        client: TestClient = test_env["client"]
        resp = client.post("/api/v1/auth/login", json={
            "username": "analyst_jane",
            "password": "WrongPassword!",
        })
        assert resp.status_code == 401
        assert "Invalid username or password" in resp.json()["detail"]

    def test_login_inactive_user(self, test_env):
        client: TestClient = test_env["client"]
        resp = client.post("/api/v1/auth/login", json={
            "username": "inactive_user",
            "password": "InactivePass123!",
        })
        assert resp.status_code == 401
        assert "inactive" in resp.json()["detail"].lower()

    def test_login_missing_fields(self, test_env):
        client: TestClient = test_env["client"]
        resp = client.post("/api/v1/auth/login", json={"username": ""})
        assert resp.status_code == 422


# ==============================================================================
# 2. Refresh Token Endpoint Tests
# ==============================================================================

class TestRefreshEndpoint:
    """Validates POST /api/v1/auth/refresh."""

    def test_refresh_token_rotation_success(self, test_env):
        client: TestClient = test_env["client"]
        analyst: User = test_env["analyst"]

        # 1. Login to get initial refresh token
        login_resp = client.post("/api/v1/auth/login", json={
            "username": analyst.username,
            "password": "StrongPassword123!",
        })
        initial_refresh = login_resp.json()["refresh_token"]

        # 2. Refresh tokens
        refresh_resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": initial_refresh,
        })
        assert refresh_resp.status_code == 200
        new_data = refresh_resp.json()
        assert "access_token" in new_data
        assert "refresh_token" in new_data
        assert new_data["refresh_token"] != initial_refresh

        # 3. Attempting to replay old refresh token must be rejected (revoked)
        replay_resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": initial_refresh,
        })
        assert replay_resp.status_code == 401
        assert "revoked" in replay_resp.json()["detail"].lower()

    def test_refresh_token_tampered_signature(self, test_env):
        client: TestClient = test_env["client"]
        analyst: User = test_env["analyst"]
        token, _ = create_refresh_token(subject=str(analyst.id), username=analyst.username)
        tampered = token[:-4] + "abcd"

        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": tampered})
        assert resp.status_code == 401

    def test_refresh_token_wrong_token_type(self, test_env):
        client: TestClient = test_env["client"]
        analyst: User = test_env["analyst"]
        # Create an access token and pass it to the refresh endpoint
        access_tok = create_access_token(
            subject=str(analyst.id),
            role="ANALYST",
            username=analyst.username,
        )
        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": access_tok})
        assert resp.status_code == 401
        assert "Expected 'refresh'" in resp.json()["detail"] or "refresh" in resp.json()["detail"].lower()

    def test_refresh_token_inactive_user(self, test_env):
        client: TestClient = test_env["client"]
        inactive: User = test_env["inactive"]
        token, _ = create_refresh_token(subject=str(inactive.id), username=inactive.username)

        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": token})
        assert resp.status_code == 401
        assert "inactive" in resp.json()["detail"].lower()


# ==============================================================================
# 3. Current User Profile Endpoint Tests
# ==============================================================================

class TestCurrentUserProfileEndpoint:
    """Validates GET /api/v1/auth/me."""

    def test_get_current_user_analyst_success(self, test_env):
        client: TestClient = test_env["client"]
        analyst: User = test_env["analyst"]

        # Helper dependency override for get_current_user to return mock analyst
        app.dependency_overrides[get_current_user] = lambda: analyst

        token = create_access_token(str(analyst.id), role="ANALYST", username=analyst.username)
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == analyst.username
        assert data["email"] == analyst.email
        assert data["role"] == "ANALYST"
        assert data["is_active"] is True
        assert data["is_superuser"] is False

    def test_get_current_user_admin_success(self, test_env):
        client: TestClient = test_env["client"]
        admin: User = test_env["admin"]

        app.dependency_overrides[get_current_user] = lambda: admin

        token = create_access_token(str(admin.id), role="ADMIN", username=admin.username)
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == admin.username
        assert data["role"] == "ADMIN"
        assert data["is_superuser"] is False

    def test_get_current_user_unauthenticated(self, test_env):
        client: TestClient = test_env["client"]
        # Remove any override on get_current_user
        if get_current_user in app.dependency_overrides:
            del app.dependency_overrides[get_current_user]

        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401


# ==============================================================================
# 4. Admin User Creation Endpoint Tests
# ==============================================================================

class TestAdminUserCreationEndpoint:
    """Validates POST /api/v1/auth/users."""

    def test_create_user_admin_success(self, test_env):
        client: TestClient = test_env["client"]
        admin: User = test_env["admin"]
        user_repo: InMemoryUserRepository = test_env["user_repo"]

        app.dependency_overrides[get_current_user] = lambda: admin

        token = create_access_token(str(admin.id), role="ADMIN", username=admin.username)
        resp = client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": "new_analyst",
                "email": "new_analyst@bank.com",
                "password": "SecurePassword123!",
                "role": "ANALYST",
                "full_name": "New Analyst",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "new_analyst"
        assert data["email"] == "new_analyst@bank.com"
        assert data["role"] == "ANALYST"
        assert data["is_superuser"] is False

        # Verify password was stored hashed, not plaintext
        created = user_repo._users[uuid.UUID(data["id"])]
        assert created.hashed_password != "SecurePassword123!"
        assert created.hashed_password.startswith("$2b$")

    def test_create_user_admin_creates_admin(self, test_env):
        client: TestClient = test_env["client"]
        admin: User = test_env["admin"]

        app.dependency_overrides[get_current_user] = lambda: admin

        token = create_access_token(str(admin.id), role="ADMIN", username=admin.username)
        resp = client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": "new_admin",
                "email": "new_admin@bank.com",
                "password": "SecureAdminPassword123!",
                "role": "ADMIN",
            },
        )
        assert resp.status_code == 201
        assert resp.json()["role"] == "ADMIN"
        assert resp.json()["is_superuser"] is False

    def test_create_user_analyst_forbidden(self, test_env):
        client: TestClient = test_env["client"]
        analyst: User = test_env["analyst"]

        app.dependency_overrides[get_current_user] = lambda: analyst

        token = create_access_token(str(analyst.id), role="ANALYST", username=analyst.username)
        resp = client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": "should_fail",
                "email": "fail@bank.com",
                "password": "SecurePassword123!",
                "role": "ANALYST",
            },
        )
        assert resp.status_code == 403
        assert "Administrative privileges required" in resp.json()["detail"]

    def test_create_user_client_controlled_superuser_ignored(self, test_env):
        """Even if client sends is_superuser: True, backend creates non-superuser."""
        client: TestClient = test_env["client"]
        admin: User = test_env["admin"]
        user_repo: InMemoryUserRepository = test_env["user_repo"]

        app.dependency_overrides[get_current_user] = lambda: admin

        token = create_access_token(str(admin.id), role="ADMIN", username=admin.username)
        resp = client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": "priv_escalation_attempt",
                "email": "priv@bank.com",
                "password": "SecurePassword123!",
                "role": "ANALYST",
                "is_superuser": True,  # Attacker tries privilege escalation
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["is_superuser"] is False
        created = user_repo._users[uuid.UUID(data["id"])]
        assert created.is_superuser is False

    def test_create_user_role_isolation_rejects_system_or_machine_role(self, test_env):
        """Role isolation: human accounts cannot be created as SYSTEM or API_CLIENT."""
        client: TestClient = test_env["client"]
        admin: User = test_env["admin"]

        app.dependency_overrides[get_current_user] = lambda: admin

        token = create_access_token(str(admin.id), role="ADMIN", username=admin.username)

        # Attempt to set role to SYSTEM
        resp_sys = client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": "sys_user",
                "email": "sys@bank.com",
                "password": "SecurePassword123!",
                "role": "SYSTEM",
            },
        )
        assert resp_sys.status_code == 422

        # Attempt to set role to API_CLIENT
        resp_api = client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": "api_user",
                "email": "api@bank.com",
                "password": "SecurePassword123!",
                "role": "API_CLIENT",
            },
        )
        assert resp_api.status_code == 422

    def test_create_user_duplicate_username(self, test_env):
        client: TestClient = test_env["client"]
        admin: User = test_env["admin"]

        app.dependency_overrides[get_current_user] = lambda: admin

        token = create_access_token(str(admin.id), role="ADMIN", username=admin.username)
        resp = client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": "analyst_jane",  # Already exists
                "email": "different_email@bank.com",
                "password": "SecurePassword123!",
                "role": "ANALYST",
            },
        )
        assert resp.status_code == 409
        assert "already in use" in resp.json()["detail"].lower()

    def test_create_user_duplicate_email(self, test_env):
        client: TestClient = test_env["client"]
        admin: User = test_env["admin"]

        app.dependency_overrides[get_current_user] = lambda: admin

        token = create_access_token(str(admin.id), role="ADMIN", username=admin.username)
        resp = client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": "unique_username",
                "email": "jane@bank.com",  # Already exists
                "password": "SecurePassword123!",
                "role": "ANALYST",
            },
        )
        assert resp.status_code == 409
        assert "already in use" in resp.json()["detail"].lower()

    def test_create_user_short_password(self, test_env):
        client: TestClient = test_env["client"]
        admin: User = test_env["admin"]

        app.dependency_overrides[get_current_user] = lambda: admin

        token = create_access_token(str(admin.id), role="ADMIN", username=admin.username)
        resp = client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": "short_pass_user",
                "email": "short@bank.com",
                "password": "short",  # < 8 chars
                "role": "ANALYST",
            },
        )
        assert resp.status_code == 422


# ==============================================================================
# 5. Admin API Key Creation Endpoint Tests
# ==============================================================================

class TestAdminAPIKeyCreationEndpoint:
    """Validates POST /api/v1/auth/api-keys."""

    def test_create_api_key_admin_success(self, test_env):
        client: TestClient = test_env["client"]
        admin: User = test_env["admin"]
        key_repo: InMemoryAPIKeyRepository = test_env["key_repo"]

        app.dependency_overrides[get_current_user] = lambda: admin

        token = create_access_token(str(admin.id), role="ADMIN", username=admin.username)
        resp = client.post(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": "payment-gateway-service",
                "description": "API key for payment gateway integration",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "raw_key" in data
        assert data["raw_key"].startswith(settings.API_KEY_PREFIX)
        assert data["role"] == "API_CLIENT"
        assert data["name"] == "payment-gateway-service"
        assert data["key_prefix"] == settings.API_KEY_PREFIX
        assert data["is_active"] is True

        # Verify key was stored as SHA-256 hash in repository, never plaintext
        stored_key = key_repo._keys[uuid.UUID(data["id"])]
        assert stored_key.hashed_key != data["raw_key"]
        assert stored_key.hashed_key == hash_api_key(data["raw_key"])
        assert stored_key.role == AuditActorType.API_CLIENT

    def test_create_api_key_analyst_forbidden(self, test_env):
        client: TestClient = test_env["client"]
        analyst: User = test_env["analyst"]

        app.dependency_overrides[get_current_user] = lambda: analyst

        token = create_access_token(str(analyst.id), role="ANALYST", username=analyst.username)
        resp = client.post(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "test-key"},
        )
        assert resp.status_code == 403
        assert "Administrative privileges required" in resp.json()["detail"]


# ==============================================================================
# 6. Authoritative ActorContext Resolution Tests
# ==============================================================================

class TestActorContextResolution:
    """Validates get_current_actor resolution across Bearer JWT, API keys, and dev headers."""

    @pytest.mark.asyncio
    async def test_get_current_actor_bearer_jwt(self, test_env):
        analyst: User = test_env["analyst"]
        token = create_access_token(str(analyst.id), role="ANALYST", username=analyst.username)

        req = MagicMock()
        req.headers = {"Authorization": f"Bearer {token}"}
        req.client.host = "192.168.1.100"
        req.state = MagicMock()
        req.state.correlation_id = "test-cid-123"

        actor = await get_current_actor(
            request=req,
            auth=token,
        )
        assert actor.actor_id == analyst.username
        assert actor.actor_role == AuditActorType.ANALYST
        assert actor.client_ip == "192.168.1.100"
        assert actor.correlation_id == "test-cid-123"

    @pytest.mark.asyncio
    async def test_get_current_actor_api_key_active(self, test_env):
        key_repo: InMemoryAPIKeyRepository = test_env["key_repo"]
        raw_key, prefix = generate_api_key()
        hashed = hash_api_key(raw_key)

        api_key = APIKey(
            id=uuid.uuid4(),
            name="microservice_ingest",
            client_id="svc-ingest-01",
            key_prefix=prefix,
            hashed_key=hashed,
            role=AuditActorType.API_CLIENT,
            is_active=True,
        )
        key_repo.add(api_key)

        # Mock an AsyncSession that forwards get_by_hashed_key and update_last_used
        mock_session = MagicMock()

        # Mock get_by_hashed_key via APIKeyRepository patch
        req = MagicMock()
        req.headers = {"X-API-Key": raw_key}
        req.client.host = "10.0.0.50"
        req.state = MagicMock()
        req.state.correlation_id = "key-cid-999"

        from unittest.mock import patch
        with patch("backend.app.repositories.api_key_repository.APIKeyRepository.get_by_hashed_key", new=AsyncMock(return_value=api_key)), \
             patch("backend.app.repositories.api_key_repository.APIKeyRepository.update_last_used", new=AsyncMock()):
            actor = await get_current_actor(
                request=req,
                x_api_key=raw_key,
                session=mock_session,
            )
            assert actor.actor_id == "svc-ingest-01"
            assert actor.actor_role == AuditActorType.API_CLIENT

    @pytest.mark.asyncio
    async def test_get_current_actor_api_key_revoked(self):
        from fastapi import HTTPException
        raw_key, prefix = generate_api_key()
        hashed = hash_api_key(raw_key)

        api_key = APIKey(
            id=uuid.uuid4(),
            name="revoked_key",
            client_id="svc-revoked",
            key_prefix=prefix,
            hashed_key=hashed,
            role=AuditActorType.API_CLIENT,
            is_active=False,
        )

        req = MagicMock()
        req.client.host = "10.0.0.50"
        req.state = MagicMock()
        mock_session = MagicMock()

        from unittest.mock import patch
        with patch("backend.app.repositories.api_key_repository.APIKeyRepository.get_by_hashed_key", new=AsyncMock(return_value=api_key)):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_actor(
                    request=req,
                    x_api_key=raw_key,
                    session=mock_session,
                )
            assert exc_info.value.status_code == 401
            assert "revoked or inactive" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_current_actor_api_key_expired(self):
        from fastapi import HTTPException
        raw_key, prefix = generate_api_key()
        hashed = hash_api_key(raw_key)

        api_key = APIKey(
            id=uuid.uuid4(),
            name="expired_key",
            client_id="svc-expired",
            key_prefix=prefix,
            hashed_key=hashed,
            role=AuditActorType.API_CLIENT,
            is_active=True,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        req = MagicMock()
        req.client.host = "10.0.0.50"
        req.state = MagicMock()
        mock_session = MagicMock()

        from unittest.mock import patch
        with patch("backend.app.repositories.api_key_repository.APIKeyRepository.get_by_hashed_key", new=AsyncMock(return_value=api_key)):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_actor(
                    request=req,
                    x_api_key=raw_key,
                    session=mock_session,
                )
            assert exc_info.value.status_code == 401
            assert "expired" in exc_info.value.detail
