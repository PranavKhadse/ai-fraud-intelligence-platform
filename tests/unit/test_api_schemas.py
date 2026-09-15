"""
Unit Tests for Pydantic v2 Request and Response Schemas in FastAPI Fraud API.

Validates:
1. TransactionPredictRequest:
   - Full 55-feature valid payload parses successfully.
   - Optional metadata attributes (transaction_id, account_id, timestamp) are preserved.
   - Numerical bound validations (amount >= 0, lat/long ranges, hour 0-23, month 1-12).
   - Extra metadata fields are allowed and retained.
2. Response Schemas:
   - HealthResponse schema validation.
   - ReasonCodeResponse, RuleMatchResponse, FeatureAttributionResponse, PredictionResponse.
"""

import pytest
from pydantic import ValidationError
from datetime import datetime, timezone

from backend.app.schemas.health import HealthResponse
from backend.app.schemas.predict import (
    TransactionPredictRequest,
    PredictionResponse,
    ReasonCodeResponse,
    RuleMatchResponse,
    FeatureAttributionResponse,
)


@pytest.fixture
def valid_transaction_payload() -> dict:
    """Return a complete dictionary containing all 55 predictive features."""
    return {
        "transaction_id": "tx_test_12345",
        "account_id": "acc_user_987",
        "timestamp": "2026-09-15T09:30:00Z",
        # 6 Canonical Numeric Predictors
        "amount": 49.99,
        "cardholder_lat": 40.7128,
        "cardholder_long": -74.0060,
        "merchant_lat": 40.7306,
        "merchant_long": -73.9352,
        "city_pop": 8336817.0,
        # 2 Categorical Predictors
        "merchant_category": "shopping_net",
        "job_category": "engineer",
        # 11 Temporal
        "transaction_hour": 14,
        "day_of_week": 2,
        "day_of_month": 15,
        "month": 9,
        "week_of_year": 38,
        "is_weekend": 0,
        "is_night": 0,
        "hour_sin": 0.5,
        "hour_cos": -0.866,
        "day_of_week_sin": 0.9749,
        "day_of_week_cos": -0.2225,
        # 7 Velocity
        "txn_count_1h": 1.0,
        "txn_count_6h": 2.0,
        "txn_count_24h": 3.0,
        "txn_count_7d": 10.0,
        "txn_count_30d": 35.0,
        "time_since_prev_txn_seconds": 3600.0,
        "is_first_account_txn": 0,
        # 8 Spending
        "amt_sum_1h": 49.99,
        "amt_sum_24h": 120.50,
        "amt_sum_7d": 450.00,
        "amt_sum_30d": 1800.00,
        "amt_mean_24h": 40.17,
        "amt_mean_7d": 45.00,
        "amt_max_24h": 60.00,
        "amt_median_30d": 38.50,
        # 5 Deviation
        "historical_amount_mean": 42.00,
        "historical_amount_std": 15.50,
        "historical_amount_median": 38.50,
        "amount_zscore": 0.515,
        "amount_ratio_to_historical_mean": 1.19,
        # 6 Account History
        "account_txn_count_before": 35.0,
        "account_total_spend_before": 1800.00,
        "account_avg_amount_before": 51.43,
        "account_max_amount_before": 150.00,
        "account_unique_merchant_count_before": 20.0,
        "account_unique_category_count_before": 8.0,
        # 6 Merchant Interaction
        "account_merchant_txn_count_before": 3.0,
        "account_category_txn_count_before": 12.0,
        "account_merchant_spend_before": 150.00,
        "account_category_spend_before": 520.00,
        "merchant_txn_count_before": 2500.0,
        "category_txn_count_before": 15000.0,
        # 4 Geographic
        "cardholder_merchant_distance_km": 6.35,
        "distance_from_prev_merchant_km": 2.10,
        "implied_travel_speed_kmh": 2.10,
        "is_impossible_travel_speed": 0,
    }


def test_transaction_predict_request_valid(valid_transaction_payload: dict):
    """Verify that a complete, valid dictionary parses into TransactionPredictRequest."""
    req = TransactionPredictRequest(**valid_transaction_payload)
    assert req.amount == 49.99
    assert req.merchant_category == "shopping_net"
    assert req.transaction_id == "tx_test_12345"
    assert req.account_id == "acc_user_987"
    assert req.cardholder_lat == 40.7128


def test_transaction_predict_request_negative_amount(valid_transaction_payload: dict):
    """Verify that negative amounts are rejected with ValidationError."""
    payload = valid_transaction_payload.copy()
    payload["amount"] = -1.0
    with pytest.raises(ValidationError) as excinfo:
        TransactionPredictRequest(**payload)
    assert "greater than or equal to 0" in str(excinfo.value)


def test_transaction_predict_request_invalid_latitude(valid_transaction_payload: dict):
    """Verify that latitude outside [-90, 90] is rejected."""
    payload = valid_transaction_payload.copy()
    payload["cardholder_lat"] = 95.0
    with pytest.raises(ValidationError) as excinfo:
        TransactionPredictRequest(**payload)
    assert "less than or equal to 90" in str(excinfo.value)


def test_transaction_predict_request_invalid_hour(valid_transaction_payload: dict):
    """Verify that transaction_hour outside [0, 23] is rejected."""
    payload = valid_transaction_payload.copy()
    payload["transaction_hour"] = 25
    with pytest.raises(ValidationError) as excinfo:
        TransactionPredictRequest(**payload)
    assert "less than or equal to 23" in str(excinfo.value)


def test_transaction_predict_request_missing_field(valid_transaction_payload: dict):
    """Verify that missing a required predictive field raises ValidationError."""
    payload = valid_transaction_payload.copy()
    del payload["amount"]
    with pytest.raises(ValidationError) as excinfo:
        TransactionPredictRequest(**payload)
    assert "Field required" in str(excinfo.value)


def test_health_response_schema():
    """Verify HealthResponse model validation."""
    resp = HealthResponse(
        status="healthy",
        app_name="Test Fraud API",
        version="1.0.0",
        model_loaded=True,
        model_version="1.0.0",
        rules_loaded_count=6,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    assert resp.status == "healthy"
    assert resp.model_loaded is True
    assert resp.rules_loaded_count == 6


def test_prediction_response_schema():
    """Verify PredictionResponse model serialization and validation."""
    rule_match = RuleMatchResponse(
        rule_id="RULE_VELOCITY_BURST_REVIEW",
        description="Velocity burst test rule",
        feature_name="txn_count_1h",
        operator=">",
        comparison_value=4.0,
        outcome="REVIEW",
        rule_type="VELOCITY",
        priority=30,
    )
    reason_code = ReasonCodeResponse(
        code="VELOCITY_BURST_1H",
        headline="Velocity Burst",
        description="High velocity in past hour.",
        category="VELOCITY",
        source="MODEL",
        severity="HIGH",
        rank=1,
    )
    feature_attr = FeatureAttributionResponse(
        feature_name="txn_count_1h",
        display_name="1-Hour Transaction Count",
        raw_value=5.0,
        shap_value=0.85,
        direction="RISK_INCREASING",
        relative_contribution_pct=45.5,
        rank=1,
    )
    resp = PredictionResponse(
        transaction_id="tx_123",
        model_score=0.72,
        risk_score=72,
        risk_tier="HIGH",
        decision_action="REVIEW",
        policy_mode="TRI_TIER",
        reason="Test policy action",
        is_overridden=True,
        rule_action="REVIEW",
        rules_triggered=["RULE_VELOCITY_BURST_REVIEW"],
        rule_matches=[rule_match],
        reason_codes=[reason_code],
        top_risk_factors=[feature_attr],
        top_mitigating_factors=[],
        model_version="1.0.0",
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )
    assert resp.risk_score == 72
    assert resp.is_overridden is True
    assert len(resp.rules_triggered) == 1
    assert len(resp.reason_codes) == 1
    assert len(resp.top_risk_factors) == 1
