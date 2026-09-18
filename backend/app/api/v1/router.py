"""
Central Router for API Version 1.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import cases, dashboard, health, predict

api_v1_router = APIRouter()

# Include versioned health, prediction, dashboard, and case management endpoints
api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(predict.router, tags=["Predictions"])
api_v1_router.include_router(dashboard.router, tags=["Dashboard"])
api_v1_router.include_router(cases.router, tags=["Cases"])
