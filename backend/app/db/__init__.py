"""
Database and Persistence Package for PostgreSQL.

Exposes:
- `get_async_engine`: Global asynchronous database engine provider.
- `get_session_factory`: Async sessionmaker provider.
- `get_db_session`: FastAPI dependency providing scoped AsyncSession instances.
- `close_db_engine`: Application shutdown cleanup function.
- `reset_db_engine`: Test teardown reset helper.
- `check_db_health`: Asynchronous database connectivity probe.
"""

from backend.app.db.session import (
    get_async_engine,
    get_session_factory,
    get_db_session,
    close_db_engine,
    reset_db_engine,
)
from backend.app.db.health import check_db_health

__all__ = [
    "get_async_engine",
    "get_session_factory",
    "get_db_session",
    "close_db_engine",
    "reset_db_engine",
    "check_db_health",
]
