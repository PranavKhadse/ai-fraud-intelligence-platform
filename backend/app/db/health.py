"""
Database Health Helper Module for Asynchronous PostgreSQL.

Provides:
- Lightweight `check_db_health` function executing a minimal `SELECT 1` probe.
- Error sanitization to avoid leaking credentials or driver connection strings.
- Pure lazy execution without import-time side effects.
"""

from typing import Dict, Any, Optional
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text

from backend.app.db.session import get_async_engine

logger = logging.getLogger("fraud_api.db.health")


async def check_db_health(
    engine: Optional[AsyncEngine] = None,
    timeout_seconds: float = 3.0,
) -> Dict[str, Any]:
    """
    Probe the database with a lightweight 'SELECT 1' query to verify connectivity.

    Args:
        engine: Optional AsyncEngine instance. If None, the application engine is used.
        timeout_seconds: Maximum seconds to wait for the probe query to complete.

    Returns:
        Dict[str, Any]: Health status dictionary containing:
            - status: "healthy" | "unhealthy"
            - database_connected: bool
            - error: Optional sanitized error message
    """
    target_engine = engine or get_async_engine()

    try:
        # Wrap connection and execution in explicit timeout
        async with asyncio.timeout(timeout_seconds):
            async with target_engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                row = result.scalar()
                if row == 1:
                    return {
                        "status": "healthy",
                        "database_connected": True,
                        "error": None,
                    }
                return {
                    "status": "unhealthy",
                    "database_connected": False,
                    "error": f"Unexpected health probe scalar result: {row}",
                }
    except TimeoutError:
        logger.warning(f"Database health check timed out after {timeout_seconds}s.")
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": f"Database health probe timed out after {timeout_seconds}s.",
        }
    except Exception as exc:
        sanitized_error = f"{type(exc).__name__}: {str(exc).split('@')[-1] if '@' in str(exc) else str(exc)}"
        # Strip any accidental user credentials from error string
        clean_error = sanitized_error.split("://")[-1] if "://" in sanitized_error else sanitized_error
        logger.warning(f"Database health check failed: {type(exc).__name__}")
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": f"Database connection probe failed ({type(exc).__name__}).",
        }
