"""
Unit Tests for Phase 15.1: UserRepository and APIKeyRepository.

Validates:
- Query operations by ID, username, email, hashed_key, and client_id.
- Case-insensitive email lookups.
- Paginated listing with active-status and role filtering.
- Last-login and last-used timestamp updates.
- Transaction-boundary neutrality (never commits or flushes implicitly).
- Error wrapping: SQLAlchemyError translation into domain PersistenceError.
- Defensive validation of pagination bounds.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.api_key import APIKey
from backend.app.db.models.enums import AuditActorType, UserRole
from backend.app.db.models.user import User
from backend.app.repositories.api_key_repository import APIKeyRepository
from backend.app.repositories.exceptions import PersistenceError
from backend.app.repositories.user_repository import UserRepository


@pytest.fixture
def mock_session():
    """Mock AsyncSession for isolated repository testing."""
    session = MagicMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    return session


class TestUserRepository:
    """Validates asynchronous UserRepository operations."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session):
        user_id = uuid.uuid4()
        user = User(
            id=user_id,
            username="investigator_1",
            email="inv1@bank.com",
            hashed_password="hash",
            role=UserRole.ANALYST,
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        repo = UserRepository(mock_session)
        result = await repo.get_by_id(user_id)

        assert result is user
        assert result.username == "investigator_1"
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = UserRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_raises_persistence_error(self, mock_session):
        mock_session.execute.side_effect = SQLAlchemyError("DB connection dropped")
        repo = UserRepository(mock_session)

        with pytest.raises(PersistenceError, match="Failed to retrieve user by ID"):
            await repo.get_by_id(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_get_by_username_found(self, mock_session):
        user = User(
            id=uuid.uuid4(),
            username="admin_user",
            email="admin@bank.com",
            hashed_password="hash",
            role=AuditActorType.ADMIN,
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        repo = UserRepository(mock_session)
        result = await repo.get_by_username("admin_user")

        assert result is user
        assert result.role == AuditActorType.ADMIN

    @pytest.mark.asyncio
    async def test_get_by_email_found(self, mock_session):
        user = User(
            id=uuid.uuid4(),
            username="analyst_2",
            email="analyst@bank.com",
            hashed_password="hash",
            role=AuditActorType.ANALYST,
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        repo = UserRepository(mock_session)
        result = await repo.get_by_email("ANALYST@BANK.COM")

        assert result is user

    @pytest.mark.asyncio
    async def test_list_users_pagination_and_filter(self, mock_session):
        user1 = User(id=uuid.uuid4(), username="u1", email="u1@b.com", hashed_password="h", role=UserRole.ANALYST)
        user2 = User(id=uuid.uuid4(), username="u2", email="u2@b.com", hashed_password="h", role=UserRole.ANALYST)

        count_result = MagicMock()
        count_result.scalar_one.return_value = 2

        users_result = MagicMock()
        users_result.scalars.return_value.all.return_value = [user1, user2]

        mock_session.execute.side_effect = [count_result, users_result]

        repo = UserRepository(mock_session)
        users, total = await repo.list_users(limit=10, offset=0, is_active=True, role=UserRole.ANALYST)

        assert total == 2
        assert len(users) == 2
        assert users[0] is user1

    @pytest.mark.asyncio
    async def test_list_users_invalid_bounds(self, mock_session):
        repo = UserRepository(mock_session)

        with pytest.raises(ValueError, match="limit must be positive"):
            await repo.list_users(limit=0)

        with pytest.raises(ValueError, match="offset cannot be negative"):
            await repo.list_users(offset=-1)

    @pytest.mark.asyncio
    async def test_add_and_add_many(self, mock_session):
        repo = UserRepository(mock_session)
        user = User(id=uuid.uuid4(), username="u3", email="u3@b.com", hashed_password="h")

        res = await repo.add(user)
        assert res is user
        mock_session.add.assert_called_once_with(user)
        mock_session.commit.assert_not_called()

        # Add many empty
        empty_res = await repo.add_many([])
        assert empty_res == []

        # Add many with items
        multi_res = await repo.add_many([user])
        assert multi_res == [user]
        mock_session.add_all.assert_called_once_with([user])

    @pytest.mark.asyncio
    async def test_update_last_login(self, mock_session):
        user_id = uuid.uuid4()
        user = User(id=user_id, username="u4", email="u4@b.com", hashed_password="h", last_login_at=None)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        repo = UserRepository(mock_session)
        custom_time = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
        updated = await repo.update_last_login(user_id, login_time=custom_time)

        assert updated is user
        assert user.last_login_at == custom_time


class TestAPIKeyRepository:
    """Validates asynchronous APIKeyRepository operations."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session):
        key_id = uuid.uuid4()
        key = APIKey(
            id=key_id,
            key_prefix="pk_live_",
            hashed_key="sha256_hash",
            name="Checkout Gateway",
            client_id="checkout_svc",
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = key
        mock_session.execute.return_value = mock_result

        repo = APIKeyRepository(mock_session)
        result = await repo.get_by_id(key_id)

        assert result is key
        assert result.client_id == "checkout_svc"

    @pytest.mark.asyncio
    async def test_get_by_hashed_key_found(self, mock_session):
        key = APIKey(
            id=uuid.uuid4(),
            key_prefix="pk_test_",
            hashed_key="test_digest_12345",
            name="Testing Key",
            client_id="test_runner",
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = key
        mock_session.execute.return_value = mock_result

        repo = APIKeyRepository(mock_session)
        result = await repo.get_by_hashed_key("test_digest_12345")

        assert result is key
        assert result.key_prefix == "pk_test_"

    @pytest.mark.asyncio
    async def test_list_by_client_id(self, mock_session):
        k1 = APIKey(id=uuid.uuid4(), key_prefix="pk_1", hashed_key="h1", name="k1", client_id="c1", is_active=True)
        k2 = APIKey(id=uuid.uuid4(), key_prefix="pk_2", hashed_key="h2", name="k2", client_id="c1", is_active=False)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [k1]
        mock_session.execute.return_value = mock_result

        repo = APIKeyRepository(mock_session)
        results = await repo.list_by_client_id("c1", active_only=True)

        assert len(results) == 1
        assert results[0] is k1

    @pytest.mark.asyncio
    async def test_list_by_prefix(self, mock_session):
        k = APIKey(id=uuid.uuid4(), key_prefix="pk_live_", hashed_key="h", name="k", client_id="c", is_active=True)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [k]
        mock_session.execute.return_value = mock_result

        repo = APIKeyRepository(mock_session)
        results = await repo.list_by_prefix("pk_live_")

        assert len(results) == 1
        assert results[0] is k

    @pytest.mark.asyncio
    async def test_list_by_user_id(self, mock_session):
        user_id = uuid.uuid4()
        k = APIKey(id=uuid.uuid4(), key_prefix="pk_u_", hashed_key="h", name="k", client_id="c", user_id=user_id)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [k]
        mock_session.execute.return_value = mock_result

        repo = APIKeyRepository(mock_session)
        results = await repo.list_by_user_id(user_id)

        assert len(results) == 1
        assert results[0] is k

    @pytest.mark.asyncio
    async def test_add_and_update_last_used(self, mock_session):
        repo = APIKeyRepository(mock_session)
        key_id = uuid.uuid4()
        key = APIKey(id=key_id, key_prefix="pk_live_", hashed_key="h", name="k", client_id="c", last_used_at=None)

        # Add
        await repo.add(key)
        mock_session.add.assert_called_once_with(key)

        # Update last used
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = key
        mock_session.execute.return_value = mock_result

        now = datetime.now(timezone.utc)
        updated = await repo.update_last_used(key_id, used_time=now)
        assert updated is key
        assert key.last_used_at == now
