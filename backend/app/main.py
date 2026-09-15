"""
FastAPI Fraud Detection & Risk Intelligence API Application Entrypoint.

Initializes the FastAPI application, manages service lifespans, configures CORS middleware,
and exposes root and versioned fraud prediction and health endpoints.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.api.v1.router import api_v1_router
from backend.app.api.v1.endpoints import health, predict
from backend.app.services.risk_service import get_risk_service

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraud_api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager for startup pre-warming and graceful shutdown.
    Pre-loads the frozen champion model, preprocessor, and rule catalog into memory.
    """
    logger.info("Initializing FastAPI Fraud Detection & Risk Intelligence API...")
    try:
        # Pre-warm the RiskEvaluator and load frozen model artifacts
        service = get_risk_service()
        logger.info(
            f"Model readiness verified: version='{service.model_version}', rules={service.rules_count} loaded."
        )
    except Exception as e:
        logger.error(f"Failed to initialize RiskEvaluator during startup: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down Fraud Detection & Risk Intelligence API...")


def create_application() -> FastAPI:
    """
    Application factory creating and configuring the FastAPI instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount direct root endpoints for GET /health and POST /predict
    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, tags=["Predictions"])

    # Mount versioned API router under /api/v1
    app.include_router(api_v1_router, prefix=settings.API_V1_STR)

    @app.get(
        "/",
        summary="API Root Information",
        description="Returns API service information and active documentation routes.",
        tags=["Root"],
    )
    async def root() -> Dict[str, Any]:
        return {
            "name": settings.APP_NAME,
            "version": settings.VERSION,
            "status": "online",
            "documentation": "/docs",
            "health_endpoint": "/health",
            "predict_endpoint": "/predict",
            "api_v1_prefix": settings.API_V1_STR,
        }

    return app


app = create_application()
