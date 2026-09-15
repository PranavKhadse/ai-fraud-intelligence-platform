"""
Services Package for FastAPI Fraud Detection & Risk Intelligence API.
"""

from backend.app.services.risk_service import (
    RiskService,
    get_risk_service,
    set_risk_service,
)

__all__ = [
    "RiskService",
    "get_risk_service",
    "set_risk_service",
]
