"""
Fraud Prediction and Explainability Endpoint for FastAPI Fraud Detection API.
"""

from fastapi import APIRouter, Depends, status, HTTPException
from backend.app.schemas.predict import TransactionPredictRequest, PredictionResponse
from backend.app.services.risk_service import RiskService, get_risk_service

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Single Transaction Fraud Risk & Explainability",
    description=(
        "Processes a financial transaction payload conforming to the 55-feature behavioral contract, "
        "runs champion gradient-boosted ML inference, evaluates deterministic business rules, "
        "and computes local TreeSHAP reason codes and risk factors."
    ),
)
async def predict_transaction(
    request: TransactionPredictRequest,
    service: RiskService = Depends(get_risk_service),
) -> PredictionResponse:
    """
    Evaluate fraud risk and return calibrated score, risk tier, action, reason codes, and rule telemetry.
    """
    try:
        return service.predict_transaction(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Transaction validation error during evaluation: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error processing transaction: {str(e)}",
        )
