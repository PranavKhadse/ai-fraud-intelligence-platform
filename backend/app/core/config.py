"""
Application Configuration Module for FastAPI Fraud Detection & Risk Intelligence API.

Provides strongly typed configuration settings backed by Pydantic v2 / pydantic-settings,
with environment variable overrides, connection string normalization, password masking,
and database connection pooling parameters.
"""

from pathlib import Path
from typing import List, Optional
import re
from pydantic import Field, field_validator
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


settings = Settings()
