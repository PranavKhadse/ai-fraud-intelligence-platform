"""
Unit Tests for Database Health Check Helper in Phase 9.

Validates:
1. Successful SELECT 1 probe query returns healthy status.
2. Connection exception returns sanitized unhealthy response without leaking credentials.
3. Query timeout handling.
4. Unexpected query result handling.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.app.db.health import check_db_health


@pytest.mark.asyncio
async def test_db_health_success():
    """Verify that successful SELECT 1 execution returns healthy status."""
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_conn.execute.return_value = mock_result

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_conn
    mock_cm.__aexit__.return_value = None

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.connect.return_value = mock_cm

    health = await check_db_health(engine=mock_engine)
    assert health["status"] == "healthy"
    assert health["database_connected"] is True
    assert health["error"] is None


@pytest.mark.asyncio
async def test_db_health_unexpected_scalar():
    """Verify that unexpected scalar result is flagged as unhealthy."""
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 0
    mock_conn.execute.return_value = mock_result

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_conn
    mock_cm.__aexit__.return_value = None

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.connect.return_value = mock_cm

    health = await check_db_health(engine=mock_engine)
    assert health["status"] == "unhealthy"
    assert health["database_connected"] is False
    assert "Unexpected health probe scalar result" in health["error"]


@pytest.mark.asyncio
async def test_db_health_connection_failure():
    """Verify that connection failures return sanitized error messages without leaking secrets."""
    mock_cm = AsyncMock()
    mock_cm.__aenter__.side_effect = ConnectionRefusedError(
        "Connection refused to postgresql+asyncpg://admin:SuperSecretPass123!@10.0.0.1:5432/db"
    )

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.connect.return_value = mock_cm

    health = await check_db_health(engine=mock_engine)
    assert health["status"] == "unhealthy"
    assert health["database_connected"] is False
    assert "SuperSecretPass123!" not in health["error"]
    assert "ConnectionRefusedError" in health["error"]


@pytest.mark.asyncio
async def test_db_health_timeout():
    """Verify that probe query timeout is handled gracefully."""
    mock_cm = AsyncMock()
    mock_cm.__aenter__.side_effect = TimeoutError()

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.connect.return_value = mock_cm

    health = await check_db_health(engine=mock_engine, timeout_seconds=0.01)
    assert health["status"] == "unhealthy"
    assert health["database_connected"] is False
    assert "timed out" in health["error"].lower()
