"""
Application Configuration Module for FastAPI Fraud Detection & Risk Intelligence API.

Provides strongly typed configuration settings backed by Pydantic v2 / pydantic-settings,
with environment variable overrides, connection string normalization, password masking,
and database connection pooling parameters.
"""

from pathlib import Path
from typing import List, Optional
import re
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings for the FastAPI Risk Intelligence API service.
    Loads values from environment variables or .env file with safe defaults.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Service Information
    APP_NAME: str = "AI-Powered Fraud Detection & Risk Intelligence Platform API"
    APP_DESCRIPTION: str = (
        "Enterprise-grade, real-time fraud risk scoring and explainability API powered by "
        "gradient-boosted models, behavioral features, decision policy engine, and TreeSHAP."
    )
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Model Artifact Paths
    MODEL_PATH: Path = Path("ml/models/artifacts/champion_model.joblib")
    PREPROCESSOR_PATH: Path = Path("ml/models/artifacts/champion_preprocessor.joblib")
    METADATA_PATH: Optional[Path] = Path("ml/models/artifacts/model_metadata.json")

    # CORS and Security Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origin patterns for frontend web dashboard and microservices.",
    )
    TRUST_PROXY_HEADERS: bool = Field(
        default=False,
        description="Whether to trust X-Forwarded-For and X-Real-IP reverse proxy headers for client IP extraction.",
    )
    ALLOW_DEV_ACTOR_HEADERS: bool = Field(
        default=False,
        description="Allow X-Actor-ID and X-Actor-Role headers in development/test environments.",
    )

    # API Hardening & Payload Limits (Phase 15.6)
    MAX_REQUEST_BODY_BYTES: int = Field(
        default=1_048_576,
        ge=1024,
        description="Maximum permitted request body size in bytes (default 1MB).",
    )

    # Rate Limiting Configuration (Phase 15.6)
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable or disable in-memory rate limiting across endpoints.",
    )
    RATE_LIMIT_LOGIN_PER_MINUTE: int = Field(
        default=10,
        ge=1,
        description="Rate limit for login attempts per client IP per minute.",
    )
    RATE_LIMIT_API_PER_MINUTE: int = Field(
        default=100,
        ge=1,
        description="Rate limit for general API requests per client key/IP per minute.",
    )

    # Security Response Headers Configuration (Phase 15.6)
    SECURITY_HEADERS_ENABLED: bool = Field(
        default=True,
        description="Enable standard security response headers (X-Content-Type-Options, CSP, etc.).",
    )
    SECURITY_HSTS_ENABLED: bool = Field(
        default=False,
        description="Enable Strict-Transport-Security response header (recommended for production HTTPS).",
    )
    SECURITY_CSP_HEADER: str = Field(
        default="default-src 'self'; frame-ancestors 'none'; object-src 'none'",
        description="Content-Security-Policy header value.",
    )
    SECURITY_REFERRER_POLICY: str = Field(
        default="strict-origin-when-cross-origin",
        description="Referrer-Policy header value.",
    )

    # JWT & Authentication Configuration (Phase 15.2)
    JWT_SECRET_KEY: str = Field(
        default="dev-insecure-jwt-secret-key-change-in-production-min-32-chars",
        description="Cryptographic secret key for signing and verifying JWT tokens.",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="Cryptographic signing algorithm for JWT tokens.",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        description="Access token lifetime in minutes.",
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
        description="Refresh token lifetime in days.",
    )
    API_KEY_PREFIX: str = Field(
        default="pk_live_",
        description="Public prefix for issued machine API keys.",
    )


    # Risk Engine Configuration Defaults
    DEFAULT_POLICY_MODE: str = "TRI_TIER"
    DEFAULT_REVIEW_THRESHOLD: float = 0.35
    DEFAULT_BLOCK_THRESHOLD: float = 0.78
    DEFAULT_MAX_REASON_CODES: int = 5
    DEFAULT_TOP_K_RISK_FACTORS: int = 5
    DEFAULT_TOP_K_MITIGATING_FACTORS: int = 3

    # Database Configuration (Phase 9)
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "fraud_intelligence_db"

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_intelligence_db",
        description="Asynchronous PostgreSQL connection URL using the asyncpg driver.",
    )
    DB_POOL_SIZE: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum persistent connections in the connection pool.",
    )
    DB_MAX_OVERFLOW: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum overflow connections allowed beyond DB_POOL_SIZE under high load.",
    )
    DB_POOL_TIMEOUT: float = Field(
        default=30.0,
        ge=0.0,
        description="Seconds to wait before timing out on acquiring a connection from the pool.",
    )
    DB_POOL_RECYCLE: int = Field(
        default=1800,
        ge=-1,
        description="Recycle connections after N seconds to prevent stale dropped sockets.",
    )
    DB_POOL_PRE_PING: bool = Field(
        default=True,
        description="Ping connection prior to checkout to discard stale or disconnected sockets.",
    )
    DB_ECHO: bool = Field(
        default=False,
        description="Enable SQLAlchemy SQL query echo logging for debugging.",
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_and_normalize_database_url(cls, v: Optional[str]) -> str:
        """
        Validate and normalize database connection URLs.
        Automatically converts synchronous 'postgresql://' or 'postgres://' schemes
        to the asynchronous 'postgresql+asyncpg://' driver scheme.
        Rejects unsupported non-PostgreSQL drivers.
        """
        if v is None or not str(v).strip():
            return "postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_intelligence_db"

        url_str = str(v).strip()

        # Handle standard PostgreSQL prefix normalization
        if url_str.startswith("postgres://"):
            url_str = "postgresql+asyncpg://" + url_str[len("postgres://"):]
        elif url_str.startswith("postgresql://"):
            url_str = "postgresql+asyncpg://" + url_str[len("postgresql://"):]

        # Validate driver scheme
        allowed_schemes = ("postgresql+asyncpg://", "sqlite+aiosqlite://")
        if not any(url_str.startswith(scheme) for scheme in allowed_schemes):
            raise ValueError(
                f"Invalid or unsupported database URL scheme in '{cls._mask_url(url_str)}'. "
                f"Asynchronous PostgreSQL requires the 'postgresql+asyncpg://' driver scheme."
            )

        return url_str

    @staticmethod
    def _mask_url(url: str) -> str:
        """
        Sanitize database connection string by replacing passwords with asterisks.
        """
        return re.sub(r":([^:@/]+)@", r":***@", url)

    @property
    def safe_database_url(self) -> str:
        """
        Return the database URL with masked password for safe logging and telemetry.
        """
        return self._mask_url(self.DATABASE_URL)

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        """
        Ensure high-entropy secrets and secure configurations in production.
        Fails fast if production uses placeholder secrets or weak keys.
        """
        env_lower = self.ENVIRONMENT.lower().strip()
        is_production = env_lower in ("production", "prod")

        if is_production:
            insecure_placeholders = {
                "dev-insecure-jwt-secret-key-change-in-production-min-32-chars",
                "secret",
                "changeme",
                "password",
                "123456",
                "admin",
            }
            key_stripped = self.JWT_SECRET_KEY.strip()
            if key_stripped in insecure_placeholders or len(key_stripped) < 32:
                raise ValueError(
                    "Production Security Violation: JWT_SECRET_KEY must be a cryptographically secure, "
                    "non-default secret of at least 32 characters in production environments."
                )
            if self.ALLOW_DEV_ACTOR_HEADERS:
                raise ValueError(
                    "Production Security Violation: ALLOW_DEV_ACTOR_HEADERS must be False in production."
                )
            if any(origin.strip() == "*" for origin in self.CORS_ORIGINS):
                raise ValueError(
                    "Production Security Violation: Wildcard CORS origin '*' is strictly prohibited in production. "
                    "Explicitly specify permitted origins in CORS_ORIGINS."
                )
        return self


settings = Settings()
