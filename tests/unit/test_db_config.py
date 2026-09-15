"""
Unit Tests for Database Configuration & Settings in Phase 9.

Validates:
1. Valid PostgreSQL async URL parsing (postgresql+asyncpg://).
2. Auto-normalization of postgresql:// and postgres:// to postgresql+asyncpg://.
3. Rejection of invalid/unsupported database schemes.
4. Default pool parameters and environment variable overrides.
5. Password masking in safe_database_url.
6. Zero database connection requirement during configuration loading.
"""

import pytest
from pydantic import ValidationError
from backend.app.core.config import Settings


def test_default_database_settings():
    """Verify default database configuration values."""
    settings = Settings()
    assert settings.DATABASE_URL.startswith("postgresql+asyncpg://")
    assert settings.DB_POOL_SIZE == 20
    assert settings.DB_MAX_OVERFLOW == 10
    assert settings.DB_POOL_TIMEOUT == 30.0
    assert settings.DB_POOL_RECYCLE == 1800
    assert settings.DB_POOL_PRE_PING is True
    assert settings.DB_ECHO is False


def test_database_url_normalization_postgresql():
    """Verify that 'postgresql://' is automatically normalized to 'postgresql+asyncpg://'."""
    settings = Settings(DATABASE_URL="postgresql://myuser:mypass@localhost:5432/mydb")
    assert settings.DATABASE_URL == "postgresql+asyncpg://myuser:mypass@localhost:5432/mydb"


def test_database_url_normalization_postgres():
    """Verify that 'postgres://' is automatically normalized to 'postgresql+asyncpg://'."""
    settings = Settings(DATABASE_URL="postgres://admin:secret@db.internal:5432/risk_db")
    assert settings.DATABASE_URL == "postgresql+asyncpg://admin:secret@db.internal:5432/risk_db"


def test_database_url_rejection_invalid_scheme():
    """Verify that unsupported database driver schemes are rejected with ValueError."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(DATABASE_URL="mysql://user:pass@localhost/db")
    assert "Invalid or unsupported database URL scheme" in str(excinfo.value)


def test_safe_database_url_masking():
    """Verify that safe_database_url masks passwords properly."""
    settings = Settings(DATABASE_URL="postgresql+asyncpg://fraud_user:SuperSecretPassword123!@localhost:5432/fraud_db")
    masked = settings.safe_database_url
    assert "SuperSecretPassword123!" not in masked
    assert "postgresql+asyncpg://fraud_user:***@localhost:5432/fraud_db" == masked


def test_pool_settings_env_override(monkeypatch):
    """Verify that pool settings can be overridden via environment variables."""
    monkeypatch.setenv("DB_POOL_SIZE", "50")
    monkeypatch.setenv("DB_MAX_OVERFLOW", "25")
    monkeypatch.setenv("DB_ECHO", "true")

    settings = Settings()
    assert settings.DB_POOL_SIZE == 50
    assert settings.DB_MAX_OVERFLOW == 25
    assert settings.DB_ECHO is True
