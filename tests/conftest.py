"""
Pytest Global Configuration and Integration Test Fixtures.

Provides asynchronous database engine, session, and Unit of Work fixtures
for running real PostgreSQL persistence integration tests.
"""

import asyncio
import os
from typing import AsyncGenerator, Optional
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from backend.app.db.models import Base
from backend.app.services.persistence_service import FraudPersistenceService
from backend.app.services.unit_of_work import FraudPersistenceUnitOfWork


def get_test_database_url() -> str:
    """
    Resolve and validate the PostgreSQL test database URL with strict anti-corruption safeguards.

    Priority:
    1. TEST_DATABASE_URL environment variable
    2. Default local test database URL: 'postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_intelligence_test_db'

    Safety Guards:
    - Rejects any URL targeting the primary development or production database ('fraud_intelligence_db').
    - Requires the database name to explicitly indicate a test environment (e.g. contain 'test').
    """
    raw_url = (
        os.environ.get("TEST_DATABASE_URL")
        or "postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_intelligence_test_db"
    )
    # Ensure asyncpg driver scheme
    if raw_url.startswith("postgres://"):
        raw_url = "postgresql+asyncpg://" + raw_url[len("postgres://"):]
    elif raw_url.startswith("postgresql://"):
        raw_url = "postgresql+asyncpg://" + raw_url[len("postgresql://"):]

    # Strict Safety Guard: Never allow automated tests/truncation against non-test databases
    db_name = raw_url.rsplit("/", 1)[-1].split("?")[0].lower()
    if db_name in ("fraud_intelligence_db", "production", "prod", "main") or "test" not in db_name:
        raise RuntimeError(
            f"CRITICAL SAFETY ERROR: Test database URL '{raw_url}' targets a non-test database ('{db_name}'). "
            "Automated test execution with table truncation is strictly prohibited against non-test databases."
        )

    return raw_url



async def _check_postgres_connection(url: str) -> bool:
    """Check if PostgreSQL database is reachable."""
    try:
        engine = create_async_engine(url, poolclass=NullPool)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception:
        return False


@pytest_asyncio.fixture(scope="function")
async def pg_engine() -> AsyncGenerator[Optional[AsyncEngine], None]:
    """
    Function-scoped asynchronous PostgreSQL engine with NullPool for integration tests.

    Creates all ORM schema tables on initialization and disposes connections on teardown.
    Yields None if PostgreSQL is unreachable.
    """
    test_url = get_test_database_url()
    is_available = await _check_postgres_connection(test_url)
    if not is_available:
        yield None
        return

    engine = create_async_engine(test_url, poolclass=NullPool, echo=False)

    # Initialize schema tables and indexes declaratively from Base.metadata
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(pg_engine: Optional[AsyncEngine]) -> AsyncGenerator[AsyncSession, None]:
    """
    Function-scoped AsyncSession for isolated database integration test execution.

    Truncates all tables after each test run to ensure strict test isolation.
    Skips the test cleanly if PostgreSQL is unavailable.
    """
    if pg_engine is None:
        pytest.skip(
            "PostgreSQL test database not available. "
            "Set TEST_DATABASE_URL (e.g. postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_intelligence_test_db) "
            "or ensure PostgreSQL is running to execute integration tests."
        )

    # Ensure clean state before test execution
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE TABLE "
                "model_registry_entries, "
                "model_monitoring_snapshots, "
                "case_notes, "
                "cases, "
                "evaluation_feature_attributions, "
                "evaluation_reason_codes, "
                "evaluation_rule_matches, "
                "audit_logs, "
                "risk_evaluations, "
                "transactions "
                "CASCADE"
            )
        )

    session_factory = async_sessionmaker(
        bind=pg_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session

    # Deterministic cleanup after each test
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE TABLE "
                "model_monitoring_snapshots, "
                "case_notes, "
                "cases, "
                "evaluation_feature_attributions, "
                "evaluation_reason_codes, "
                "evaluation_rule_matches, "
                "audit_logs, "
                "risk_evaluations, "
                "transactions "
                "CASCADE"
            )
        )



@pytest_asyncio.fixture(scope="function")
async def uow(db_session: AsyncSession) -> FraudPersistenceUnitOfWork:
    """Provide a FraudPersistenceUnitOfWork instance bound to the active test database session."""
    return FraudPersistenceUnitOfWork(db_session)


@pytest_asyncio.fixture(scope="function")
async def persistence_service(uow: FraudPersistenceUnitOfWork) -> FraudPersistenceService:
    """Provide a FraudPersistenceService instance bound to the test Unit of Work."""
    return FraudPersistenceService(uow)
