"""
Unit Tests for Phase 9 Milestone 9.4 Increment 1: Transaction Repository Foundation.

Validates:
- Repository and exception class imports.
- Exception hierarchy inheritance.
- Session injection and transaction-boundary neutrality (no commit/rollback).
- Query execution (get_by_id, get_by_external_id, exists_by_external_id, add).
- Error translation and exception chaining.
- Statement structure and field targeting.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import Select

from backend.app.db.models.transaction import Transaction
from backend.app.repositories import (
    TransactionRepository,
    PersistenceError,
    PersistenceConflictError,
    PersistenceNotFoundError,
)


class TestRepositoryImportsAndHierarchy:
    """Validates public package exports and exception inheritance."""

    def test_repository_and_exceptions_import(self):
        assert TransactionRepository is not None
        assert PersistenceError is not None
        assert PersistenceConflictError is not None
        assert PersistenceNotFoundError is not None

    def test_exception_inheritance_hierarchy(self):
        assert issubclass(PersistenceConflictError, PersistenceError)
        assert issubclass(PersistenceNotFoundError, PersistenceError)
        assert issubclass(PersistenceError, Exception)

    def test_exception_custom_attributes(self):
        err = PersistenceError("Custom message", details={"field": "test"})
        assert err.message == "Custom message"
        assert err.details == {"field": "test"}
        assert str(err) == "Custom message"


class TestTransactionRepositorySessionContract:
    """Validates session injection and boundary rules."""

    def test_repository_accepts_existing_session(self):
        mock_session = MagicMock(spec=AsyncSession)
        repo = TransactionRepository(mock_session)
        assert repo._session is mock_session

    def test_repository_does_not_create_own_session(self):
        mock_session = MagicMock(spec=AsyncSession)
        with patch("backend.app.db.session.get_async_engine") as mock_engine:
            TransactionRepository(mock_session)
            mock_engine.assert_not_called()


class TestTransactionRepositoryOperations:
    """Validates repository CRUD methods against async session doubles."""

    @pytest.fixture
    def mock_session(self):
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def sample_transaction(self):
        return Transaction(
            id=uuid.uuid4(),
            external_transaction_id="TX_123456789",
            account_id="acc_98765",
            merchant_category="grocery_pos",
            job_category="engineer",
            amount=Decimal("150.00"),
            currency="USD",
            cardholder_lat=Decimal("40.712800"),
            cardholder_long=Decimal("-74.006000"),
            merchant_lat=Decimal("40.713000"),
            merchant_long=Decimal("-74.005800"),
            city_pop=500000,
            transaction_timestamp=datetime.now(timezone.utc),
            features_snapshot={"amt": 150.00},
        )

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session, sample_transaction):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_transaction
        mock_session.execute.return_value = mock_result

        repo = TransactionRepository(mock_session)
        result = await repo.get_by_id(sample_transaction.id)

        assert result is sample_transaction
        mock_session.execute.assert_awaited_once()

        # Verify SQL statement structure
        call_args = mock_session.execute.call_args[0]
        stmt = call_args[0]
        assert isinstance(stmt, Select)

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = TransactionRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())

        assert result is None
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_external_id_found(self, mock_session, sample_transaction):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_transaction
        mock_session.execute.return_value = mock_result

        repo = TransactionRepository(mock_session)
        result = await repo.get_by_external_id("TX_123456789")

        assert result is sample_transaction
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_external_id_not_found(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = TransactionRepository(mock_session)
        result = await repo.get_by_external_id("TX_NON_EXISTENT")

        assert result is None
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exists_by_external_id_true(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = True
        mock_session.execute.return_value = mock_result

        repo = TransactionRepository(mock_session)
        result = await repo.exists_by_external_id("TX_123456789")

        assert result is True
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exists_by_external_id_false(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = False
        mock_session.execute.return_value = mock_result

        repo = TransactionRepository(mock_session)
        result = await repo.exists_by_external_id("TX_NON_EXISTENT")

        assert result is False
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_attaches_and_returns_entity(self, mock_session, sample_transaction):
        repo = TransactionRepository(mock_session)
        result = await repo.add(sample_transaction)

        assert result is sample_transaction
        mock_session.add.assert_called_once_with(sample_transaction)

    @pytest.mark.asyncio
    async def test_add_does_not_commit_or_rollback(self, mock_session, sample_transaction):
        repo = TransactionRepository(mock_session)
        await repo.add(sample_transaction)

        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_id_raises_persistence_error_on_failure(self, mock_session):
        mock_session.execute.side_effect = SQLAlchemyError("Connection timeout")

        repo = TransactionRepository(mock_session)
        target_id = uuid.uuid4()

        with pytest.raises(PersistenceError) as exc_info:
            await repo.get_by_id(target_id)

        assert "Failed to retrieve transaction by ID" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, SQLAlchemyError)

    @pytest.mark.asyncio
    async def test_get_by_external_id_raises_persistence_error_on_failure(self, mock_session):
        mock_session.execute.side_effect = SQLAlchemyError("Query execution failed")

        repo = TransactionRepository(mock_session)
        with pytest.raises(PersistenceError) as exc_info:
            await repo.get_by_external_id("TX_ERROR")

        assert "Failed to retrieve transaction by external ID" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_exists_by_external_id_raises_persistence_error_on_failure(self, mock_session):
        mock_session.execute.side_effect = SQLAlchemyError("DB unreachable")

        repo = TransactionRepository(mock_session)
        with pytest.raises(PersistenceError) as exc_info:
            await repo.exists_by_external_id("TX_ERROR")

        assert "Failed to check existence for external ID" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
