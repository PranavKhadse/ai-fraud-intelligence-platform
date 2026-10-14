"""
Central Router for API Version 1.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import audit, auth, cases, dashboard, health, monitoring, predict

api_v1_router = APIRouter()

# Include versioned health, auth, prediction, dashboard, case management, model monitoring, and audit endpoints
api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_v1_router.include_router(predict.router, tags=["Predictions"])
api_v1_router.include_router(dashboard.router, tags=["Dashboard"])
api_v1_router.include_router(cases.router, tags=["Cases"])
api_v1_router.include_router(monitoring.router, tags=["Model Monitoring"])
api_v1_router.include_router(audit.router, prefix="/audit", tags=["Audit"])

