"""
Unit Tests for Database Engine & Session Management in Phase 9.

Validates:
1. Lazy AsyncEngine initialization.
2. async_sessionmaker factory creation and configuration.
3. get_db_session dependency lifecycle (yield, safe closure).
4. No unconditional commit inside get_db_session dependency.
5. Defensive rollback on unhandled exceptions in get_db_session.
6. Engine disposal via close_db_engine().
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine

from backend.app.db.session import (
    get_async_engine,
    get_session_factory,
    get_db_session,
    close_db_engine,
    reset_db_engine,
)
import backend.app.db.session as session_module


@pytest.fixture(autouse=True)
def cleanup_engine():
    """Reset the global database engine state before and after each test."""
    reset_db_engine()
    yield
    reset_db_engine()


def test_lazy_engine_initialization():
    """Verify that get_async_engine returns an AsyncEngine instance lazily."""
    engine = get_async_engine()
    assert isinstance(engine, AsyncEngine)
    # Consecutive calls return the same singleton instance
    engine2 = get_async_engine()
    assert engine is engine2


def test_session_factory_configuration():
    """Verify that get_session_factory configures AsyncSession with expire_on_commit=False."""
    factory = get_session_factory()
    assert factory.class_ == AsyncSession
    assert factory.kw.get("expire_on_commit") is False
    assert factory.kw.get("autoflush") is False


@pytest.mark.asyncio
async def test_get_db_session_lifecycle():
    """Verify that get_db_session dependency yields a session and closes it on exit."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_factory = MagicMock(return_value=mock_session)
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    with patch("backend.app.db.session.get_session_factory", return_value=mock_factory):
        session_gen = get_db_session()
        session = await anext(session_gen)
        assert session is mock_session

        # Finish generator normally
        try:
            await anext(session_gen)
        except StopAsyncIteration:
            pass

        # Verify close was called
        mock_session.close.assert_awaited_once()
        # Verify NO unconditional commit was executed
        mock_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_get_db_session_defensive_rollback_on_exception():
    """Verify that unhandled exceptions inside consumer trigger session rollback."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_factory = MagicMock(return_value=mock_session)
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    with patch("backend.app.db.session.get_session_factory", return_value=mock_factory):
        session_gen = get_db_session()
        session = await anext(session_gen)
        assert session is mock_session

        # Send exception into generator using athrow
        with pytest.raises(RuntimeError, match="Simulated consumer failure"):
            await session_gen.athrow(RuntimeError("Simulated consumer failure"))

        # Rollback and close must be awaited on exception
        mock_session.rollback.assert_awaited_once()
        mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_db_engine_disposal():
    """Verify that close_db_engine properly disposes of active engine connections."""
    mock_engine = AsyncMock(spec=AsyncEngine)
    session_module._async_engine = mock_engine
    session_module._async_session_factory = MagicMock()

    await close_db_engine()
    mock_engine.dispose.assert_awaited_once()
    assert session_module._async_engine is None
    assert session_module._async_session_factory is None
