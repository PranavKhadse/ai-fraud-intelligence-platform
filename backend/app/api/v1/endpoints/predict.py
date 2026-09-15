"""
Fraud Prediction and Explainability Endpoint for FastAPI Fraud Detection API.

Orchestrates ML inference, rule-based decisioning, TreeSHAP local explainability,
and atomic relational persistence under a unified request lifecycle.
"""

import ipaddress
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.core.config import settings
from backend.app.repositories.exceptions import PersistenceConflictError, PersistenceError
from backend.app.schemas.predict import PredictionResponse, TransactionPredictRequest
from backend.app.services.persistence_service import (
    FraudPersistenceService,
    get_persistence_service,
)
from backend.app.services.risk_persistence_mapper import (
    RiskEvaluationContext,
    RiskPersistenceMapper,
)
from backend.app.services.risk_service import RiskService, get_risk_service

logger = logging.getLogger("fraud_api.predict")

router = APIRouter()


def extract_client_ip(http_request: Request, trust_proxy_headers: bool = False) -> Optional[str]:
    """
    Safely extract and validate client IP address from HTTP request lifecycle.

    Security & Anti-Spoofing Policy:
    - If trust_proxy_headers is False (default):
      Extracts strictly from `http_request.client.host` to prevent arbitrary client IP spoofing.
    - If trust_proxy_headers is True:
      1. Inspects 'X-Forwarded-For' (evaluating comma-separated IPs from left-to-right, returning the first valid IP).
      2. If absent or invalid, inspects 'X-Real-IP'.
      3. Validates each candidate string with `ipaddress.ip_address` to prevent injection attacks.
      4. Falls back to `http_request.client.host` if all forwarding headers are missing or invalid.
    """
    if trust_proxy_headers:
        # Check X-Forwarded-For (format: client, proxy1, proxy2)
        raw_forwarded = http_request.headers.get("X-Forwarded-For", "")
        if raw_forwarded.strip():
            for part in raw_forwarded.split(","):
                candidate = part.strip()
                if candidate:
                    try:
                        ipaddress.ip_address(candidate)
                        return candidate
                    except ValueError:
                        continue

        # Check X-Real-IP
        raw_real = http_request.headers.get("X-Real-IP", "").strip()
        if raw_real:
            try:
                ipaddress.ip_address(raw_real)
                return raw_real
            except ValueError:
                pass

    # Direct connection host fallback
    if http_request.client and http_request.client.host:
        return http_request.client.host

    return None


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Single Transaction Fraud Risk & Explainability",
    description=(
        "Processes a financial transaction payload conforming to the 55-feature behavioral contract, "
        "runs champion gradient-boosted ML inference, evaluates deterministic business rules, "
        "computes local TreeSHAP reason codes and risk factors, and persists the complete evaluation aggregate."
    ),
)
async def predict_transaction(
    request: TransactionPredictRequest,
    http_request: Request,
    risk_service: RiskService = Depends(get_risk_service),
    persistence_service: FraudPersistenceService = Depends(get_persistence_service),
) -> PredictionResponse:
    """
    Evaluate fraud risk and return calibrated score, risk tier, action, reason codes, and rule telemetry,
    while atomically persisting the evaluation record into the relational persistence layer.
    """
    # 1. Monotonic latency measurement around ML inference & explainability
    eval_start_time = time.perf_counter()
    try:
        prediction_response = risk_service.predict_transaction(request)
    except ValueError as e:
        logger.warning(f"Transaction validation error during risk evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Transaction validation error during evaluation: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error processing transaction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error processing transaction: {str(e)}",
        )

    eval_latency_ms = round((time.perf_counter() - eval_start_time) * 1000.0, 2)

    # 2. Extract verified request lifecycle context
    correlation_id = (
        http_request.headers.get("X-Correlation-ID")
        or http_request.headers.get("X-Request-ID")
    )
    client_ip = extract_client_ip(http_request, trust_proxy_headers=settings.TRUST_PROXY_HEADERS)

    context = RiskEvaluationContext(
        correlation_id=correlation_id,
        client_ip=client_ip,
        actor_id="fastapi_predict_api",
        evaluation_latency_ms=eval_latency_ms,
        external_transaction_id=request.transaction_id,
        account_id=request.account_id,
        transaction_timestamp=request.timestamp,
    )

    # 3. Map prediction into persistence command
    try:
        command = RiskPersistenceMapper.map_prediction_to_command(
            request=request,
            response=prediction_response,
            context=context,
        )
    except ValueError as e:
        logger.warning(f"Validation error mapping transaction to persistence command: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Persistence mapping validation error: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error mapping transaction to persistence command: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error preparing transaction persistence.",
        )

    # 4. Atomically persist evaluation aggregate
    try:
        await persistence_service.persist_evaluation(command)
    except PersistenceConflictError as e:
        logger.warning(f"Duplicate transaction conflict for external ID '{request.transaction_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Transaction with external_transaction_id '{request.transaction_id}' already exists.",
        )
    except PersistenceError as e:
        logger.error(f"Database persistence failure for transaction '{request.transaction_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Persistence failure storing evaluation result.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected persistence error for transaction '{request.transaction_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error persisting evaluation.",
        )

    # 5. Return unmodified PredictionResponse preserving 100% public schema contract
    return prediction_response
