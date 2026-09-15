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
    while atomically persisting the evaluation record into the relational persistence layer with full idempotency.
    """
    # 1. Normalize and validate optional external_transaction_id
    ext_tx_id: Optional[str] = None
    raw_tx_id = getattr(request, "transaction_id", None)
    if raw_tx_id is not None:
        cleaned_id = str(raw_tx_id).strip()
        if not cleaned_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Transaction identifier 'transaction_id' cannot be empty or whitespace-only.",
            )
        if len(cleaned_id) > 128:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Transaction identifier 'transaction_id' exceeds maximum length of 128 characters.",
            )
        ext_tx_id = cleaned_id

    # 2. Pre-inference Idempotency Fast Path: Check if external transaction was already evaluated
    if ext_tx_id:
        try:
            existing_tx = await persistence_service.get_existing_evaluation(ext_tx_id)
            if existing_tx:
                if RiskPersistenceMapper.is_payload_equivalent(request, existing_tx):
                    logger.info(
                        "Idempotent request replay for external_transaction_id='%s'",
                        ext_tx_id,
                    )
                    return RiskPersistenceMapper.reconstruct_prediction_response(existing_tx)
                else:
                    logger.warning(
                        "Conflicting payload for existing external_transaction_id='%s'",
                        ext_tx_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Transaction with external_transaction_id '{ext_tx_id}' already exists with conflicting request payload.",
                    )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Persistence failure during idempotency lookup for transaction '%s': %s",
                ext_tx_id,
                e,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Persistence failure checking transaction idempotency.",
            )

    # 3. Monotonic latency measurement around ML inference & explainability
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

    # 4. Extract verified request lifecycle context
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
        external_transaction_id=ext_tx_id,
        account_id=request.account_id,
        transaction_timestamp=request.timestamp,
    )

    # 5. Map prediction into persistence command
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

    # 6. Atomically persist evaluation aggregate
    try:
        await persistence_service.persist_evaluation(command)
    except PersistenceConflictError as e:
        # Resolve concurrent worker race condition: re-query committed transaction
        if ext_tx_id:
            try:
                existing_tx = await persistence_service.get_existing_evaluation(ext_tx_id)
                if existing_tx and RiskPersistenceMapper.is_payload_equivalent(request, existing_tx):
                    logger.info(
                        "Resolved concurrent race for external_transaction_id='%s' via idempotent replay",
                        ext_tx_id,
                    )
                    return RiskPersistenceMapper.reconstruct_prediction_response(existing_tx)
            except Exception:
                pass

        logger.warning(f"Duplicate transaction conflict for external ID '{ext_tx_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Transaction with external_transaction_id '{ext_tx_id}' already exists with conflicting request payload.",
        )
    except PersistenceError as e:
        logger.error(f"Database persistence failure for transaction '{ext_tx_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Persistence failure storing evaluation result.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected persistence error for transaction '{ext_tx_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error persisting evaluation.",
        )

    # 7. Return unmodified PredictionResponse preserving 100% public schema contract
    return prediction_response
