"""
Health check and telemetry response schemas for FastAPI API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Standardized health status response model.
    """
    status: str = Field(..., description="Overall service status ('healthy', 'degraded', etc.)", json_schema_extra={"example": "healthy"})
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="API semantic version")
    model_loaded: bool = Field(..., description="Whether the ML champion model and preprocessor are loaded in memory")
    model_version: Optional[str] = Field(None, description="Provenance champion model version identifier")
    rules_loaded_count: int = Field(..., description="Number of registered business rules loaded in the rule engine")
    timestamp: str = Field(..., description="Current ISO 8601 server timestamp")
