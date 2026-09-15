"""
Asynchronous Database Engine and Session Management for PostgreSQL.

Provides:
- Lazy AsyncEngine initialization with connection pooling.
- async_sessionmaker factory configured for async SQLAlchemy 2.0.
- FastAPI dependency `get_db_session` with exception rollback and safe closure.
- Explicit engine disposal on application shutdown.

Transaction Boundary Rule:
The `get_db_session` dependency does NOT commit transactions automatically.
Services or persistence handlers must explicitly invoke `await session.commit()`.
"""

from typing import AsyncGenerator, Optional, Dict, Any
import logging
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine,
)
from backend.app.core.config import settings

logger = logging.getLogger("fraud_api.db.session")

# Global singleton holders for lazy engine and sessionmaker
_async_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_async_engine(
    custom_url: Optional[str] = None,
    echo: Optional[bool] = None,
    **kwargs: Any,
) -> AsyncEngine:
    """
    Get or lazily initialize the application's global AsyncEngine.

    If `custom_url` is provided, a new independent engine is instantiated
    without modifying the global singleton.
    """
    global _async_engine

    target_url = custom_url or settings.DATABASE_URL
    is_echo = settings.DB_ECHO if echo is None else echo

    # Handle custom URL instantiation
    if custom_url is not None:
        return _create_engine_instance(target_url, is_echo, **kwargs)

    # Lazy singleton instantiation for global application engine
    if _async_engine is None:
        logger.info(f"Initializing AsyncEngine for '{settings.safe_database_url}'...")
        _async_engine = _create_engine_instance(target_url, is_echo, **kwargs)

    return _async_engine


def _create_engine_instance(
    url: str,
    echo: bool,
    **kwargs: Any,
) -> AsyncEngine:
    """
    Internal factory to construct an AsyncEngine with appropriate pooling configurations.
    """
    engine_kwargs: Dict[str, Any] = {
        "echo": echo,
        **kwargs,
    }

    # SQLite (used in specific lightweight test scenarios) does not support QueuePool parameters
    if "sqlite" not in url:
        engine_kwargs.update(
            {
                "pool_size": settings.DB_POOL_SIZE,
                "max_overflow": settings.DB_MAX_OVERFLOW,
                "pool_timeout": settings.DB_POOL_TIMEOUT,
                "pool_recycle": settings.DB_POOL_RECYCLE,
                "pool_pre_ping": settings.DB_POOL_PRE_PING,
            }
        )

    return create_async_engine(url, **engine_kwargs)


def get_session_factory(
    engine: Optional[AsyncEngine] = None,
) -> async_sessionmaker[AsyncSession]:
    """
    Get or create the async session factory.
    """
    global _async_session_factory

    target_engine = engine or get_async_engine()

    if engine is not None:
        return async_sessionmaker(
            bind=target_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=target_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    return _async_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an asynchronous database session.

    Guarantees:
    - Yields a scoped AsyncSession per request.
    - Automatically rolls back uncommitted changes if an unhandled exception occurs.
    - Closes the session cleanly upon exit.
    - Does NOT automatically commit transactions; callers own explicit commits.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception as exc:
            logger.warning(f"Database session encountered exception; executing rollback: {exc}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db_engine() -> None:
    """
    Explicitly dispose of all connection pool sockets on application shutdown.
    """
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        logger.info("Disposing application AsyncEngine connection pool...")
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
        logger.info("AsyncEngine disposed successfully.")


def reset_db_engine() -> None:
    """
    Synchronously reset internal singleton engine references (primarily for test teardown).
    """
    global _async_engine, _async_session_factory
    _async_engine = None
    _async_session_factory = None
