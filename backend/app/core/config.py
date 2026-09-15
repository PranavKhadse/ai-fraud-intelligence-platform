"""
Application Configuration Module for FastAPI Fraud Detection API.

Provides strongly typed configuration settings backed by Pydantic / dataclass patterns
and environment variable overrides.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """
    Application Settings for the FastAPI Risk Intelligence API service.
    """
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

    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origin patterns for frontend web dashboard and microservices.",
    )

    # Risk Engine Configuration Defaults
    DEFAULT_POLICY_MODE: str = "TRI_TIER"
    DEFAULT_REVIEW_THRESHOLD: float = 0.35
    DEFAULT_BLOCK_THRESHOLD: float = 0.78
    DEFAULT_MAX_REASON_CODES: int = 5
    DEFAULT_TOP_K_RISK_FACTORS: int = 5
    DEFAULT_TOP_K_MITIGATING_FACTORS: int = 3


settings = Settings()
