"""
Central Router for API Version 1.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, predict

api_v1_router = APIRouter()

# Include versioned health and prediction endpoints
api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(predict.router, tags=["Predictions"])
