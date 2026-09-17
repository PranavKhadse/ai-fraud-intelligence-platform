"""
Central Router for API Version 1.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import dashboard, health, predict

api_v1_router = APIRouter()

# Include versioned health, prediction, and dashboard endpoints
api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(predict.router, tags=["Predictions"])
api_v1_router.include_router(dashboard.router, tags=["Dashboard"])
