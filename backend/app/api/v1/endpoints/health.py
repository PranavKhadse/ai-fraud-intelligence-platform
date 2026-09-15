"""
Health Check and Telemetry Endpoint for FastAPI Fraud Detection API.
"""

from fastapi import APIRouter, Depends, status
from backend.app.schemas.health import HealthResponse
from backend.app.services.risk_service import RiskService, get_risk_service

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health & Model Readiness Check",
    description=(
        "Returns the active operational status of the fraud detection API service, "
        "including model readiness, champion model provenance version, and loaded rule catalog count."
    ),
)
async def check_health(
    service: RiskService = Depends(get_risk_service),
) -> HealthResponse:
    """
    Execute health check and return service telemetry.
    """
    return service.get_health_status()
