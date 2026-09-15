"""
Alembic Environment Configuration for Asynchronous PostgreSQL Migrations.

Integrates SQLAlchemy 2.0 DeclarativeBase metadata with Alembic migration runner,
supporting both offline SQL script generation and online asyncpg migrations.
"""

import asyncio
from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also ensure backend directory is on sys.path
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Import settings and target metadata
from backend.app.core.config import settings
from backend.app.db.models import Base

# Access Alembic Config object safely
try:
    config = context.config
except (AttributeError, NameError):
    config = None

# Interpret the config file for Python logging.
if config is not None and getattr(config, "config_file_name", None) is not None:
    fileConfig(config.config_file_name)

# Set target metadata for 'autogenerate' support
target_metadata = Base.metadata


def get_url() -> str:
    """
    Retrieve the configured asynchronous PostgreSQL database URL.
    Prefers application settings, with safe fallback to configuration.
    """
    return settings.DATABASE_URL


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """
    Execute migrations synchronously within an async connection context.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Create an asynchronous engine and run migrations online.
    """
    url = get_url()
    connectable = create_async_engine(
        url,
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode using the async event loop.
    """
    asyncio.run(run_async_migrations())


# Execute migration runner when invoked by Alembic command runtime
try:
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()
except (NameError, AttributeError, RuntimeError):
    # env.py was imported directly outside of an active Alembic execution context
    pass
