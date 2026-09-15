"""
Unit Tests for Phase 9 Milestone 9.5 Increment 1: Unit of Work Foundation.

Validates:
- Unit of Work imports and alias exports.
- Constructor injection and session validation.
- Repository wiring and session sharing across all 6 persistence repositories.
- Async context manager enter/exit semantics.
- Non-committing successful context exit.
- Automatic defensive rollback on escaping exception.
- Explicit transaction boundary methods (commit, rollback, flush).
- Exception propagation through transaction methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.repositories import (
    AuditLogRepository,
    FeatureAttributionRepository,
    ReasonCodeRepository,
    RiskEvaluationRepository,
    RuleMatchRepository,
    TransactionRepository,
)
from backend.app.services import (
    FraudPersistenceUnitOfWork,
    UnitOfWork,
)


@pytest.fixture
def mock_session():
    """Mock AsyncSession for verifying Unit of Work transaction delegation."""
    session = MagicMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.close = AsyncMock()
    session.add = MagicMock()
    session.execute = AsyncMock()
    return session


class TestUnitOfWorkWiringAndInstantiation:
    """Validates constructor injection, repository wiring, and session containment."""

    def test_imports_and_alias(self):
        assert FraudPersistenceUnitOfWork is not None
        assert UnitOfWork is FraudPersistenceUnitOfWork

    def test_constructor_accepts_injected_session(self, mock_session):
        uow = FraudPersistenceUnitOfWork(mock_session)
        assert uow.session is mock_session
        assert uow._session is mock_session

    def test_constructor_rejects_none_session(self):
        with pytest.raises(ValueError, match="AsyncSession must not be None"):
            FraudPersistenceUnitOfWork(None)  # type: ignore[arg-type]

    def test_unit_of_work_never_creates_own_engine(self, mock_session):
        with patch("backend.app.db.session.get_async_engine") as mock_engine:
            FraudPersistenceUnitOfWork(mock_session)
            mock_engine.assert_not_called()

    def test_all_repositories_wired_with_same_session(self, mock_session):
        uow = FraudPersistenceUnitOfWork(mock_session)

        # Verify all 6 repositories are instantiated
        assert isinstance(uow.transactions, TransactionRepository)
        assert isinstance(uow.risk_evaluations, RiskEvaluationRepository)
        assert isinstance(uow.rule_matches, RuleMatchRepository)
        assert isinstance(uow.reason_codes, ReasonCodeRepository)
        assert isinstance(uow.feature_attributions, FeatureAttributionRepository)
        assert isinstance(uow.audit_logs, AuditLogRepository)

        # Verify all 6 repositories share the exact same session instance
        assert uow.transactions._session is mock_session
        assert uow.risk_evaluations._session is mock_session
        assert uow.rule_matches._session is mock_session
        assert uow.reason_codes._session is mock_session
        assert uow.feature_attributions._session is mock_session
        assert uow.audit_logs._session is mock_session


class TestUnitOfWorkTransactionMethods:
    """Validates explicit commit, rollback, and flush delegation."""

    @pytest.mark.asyncio
    async def test_commit_delegates_to_session(self, mock_session):
        uow = FraudPersistenceUnitOfWork(mock_session)
        await uow.commit()

        mock_session.commit.assert_awaited_once()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_delegates_to_session(self, mock_session):
        uow = FraudPersistenceUnitOfWork(mock_session)
        await uow.rollback()

        mock_session.rollback.assert_awaited_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_delegates_to_session(self, mock_session):
        uow = FraudPersistenceUnitOfWork(mock_session)
        await uow.flush()

        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_propagates_exceptions(self, mock_session):
        mock_session.commit.side_effect = SQLAlchemyError("Commit deadlock")
        uow = FraudPersistenceUnitOfWork(mock_session)

        with pytest.raises(SQLAlchemyError, match="Commit deadlock"):
            await uow.commit()

    @pytest.mark.asyncio
    async def test_rollback_propagates_exceptions(self, mock_session):
        mock_session.rollback.side_effect = SQLAlchemyError("Connection lost")
        uow = FraudPersistenceUnitOfWork(mock_session)

        with pytest.raises(SQLAlchemyError, match="Connection lost"):
            await uow.rollback()

    @pytest.mark.asyncio
    async def test_flush_propagates_exceptions(self, mock_session):
        mock_session.flush.side_effect = SQLAlchemyError("Foreign key violation on flush")
        uow = FraudPersistenceUnitOfWork(mock_session)

        with pytest.raises(SQLAlchemyError, match="Foreign key violation on flush"):
            await uow.flush()

    @pytest.mark.asyncio
    async def test_repeated_commits_allowed(self, mock_session):
        uow = FraudPersistenceUnitOfWork(mock_session)
        await uow.commit()
        await uow.commit()

        assert mock_session.commit.await_count == 2

    @pytest.mark.asyncio
    async def test_repeated_rollbacks_allowed(self, mock_session):
        uow = FraudPersistenceUnitOfWork(mock_session)
        await uow.rollback()
        await uow.rollback()

        assert mock_session.rollback.await_count == 2


class TestUnitOfWorkContextManager:
    """Validates async context manager semantics and exception safety."""

    @pytest.mark.asyncio
    async def test_context_enter_returns_self_and_tracks_state(self, mock_session):
        uow = FraudPersistenceUnitOfWork(mock_session)
        assert uow.in_context is False

        async with uow as ctx:
            assert ctx is uow
            assert uow.in_context is True

        assert uow.in_context is False

    @pytest.mark.asyncio
    async def test_context_exit_does_not_auto_commit(self, mock_session):
        """Validates that exiting the context normally does NOT trigger an automatic commit."""
        uow = FraudPersistenceUnitOfWork(mock_session)

        async with uow:
            # Perform operations without explicit commit
            pass

        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_commit_inside_context_persists(self, mock_session):
        """Validates the standard pattern: explicit commit inside context block."""
        async with FraudPersistenceUnitOfWork(mock_session) as uow:
            await uow.commit()

        mock_session.commit.assert_awaited_once()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_escaping_exception_triggers_rollback(self, mock_session):
        """Validates defensive rollback when an exception escapes the context block."""
        uow = FraudPersistenceUnitOfWork(mock_session)

        with pytest.raises(RuntimeError, match="Service computation failed"):
            async with uow:
                raise RuntimeError("Service computation failed")

        mock_session.rollback.assert_awaited_once()
        mock_session.commit.assert_not_called()
        assert uow.in_context is False

    @pytest.mark.asyncio
    async def test_escaping_custom_exception_is_not_swallowed(self, mock_session):
        """Validates that custom exceptions propagate out of the context cleanly."""
        class CustomDomainError(Exception):
            pass

        with pytest.raises(CustomDomainError):
            async with FraudPersistenceUnitOfWork(mock_session):
                raise CustomDomainError("Business rule violation")

        mock_session.rollback.assert_awaited_once()
