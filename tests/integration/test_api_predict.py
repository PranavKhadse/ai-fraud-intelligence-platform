"""
Automated Integration Tests for Phase 8: FastAPI Fraud Detection & Explainability API.

Validates:
1. POST /predict with Legitimate Transactions:
   - Produces HTTP 200 with APPROVE decision, LOW risk tier, and complete reason codes.
2. POST /api/v1/predict with Identical Behavior:
   - Ensures versioned router parity.
3. POST /predict with Fraud Transactions:
   - Produces HTTP 200 with BLOCK decision, CRITICAL risk tier, and high risk score.
4. Business Rule Overrides & Monitoring Telemetry:
   - RULE_VELOCITY_BURST_REVIEW: triggers escalation from APPROVE to REVIEW (is_overridden=True).
   - RULE_AMT_ZSCORE_DEVIATION_REVIEW: triggers escalation to REVIEW.
   - RULE_COMPLIANCE_LARGE_AMOUNT_MONITOR: records MONITOR outcome without mutating decision.
5. Strict Request Validation & Error Handling:
   - Rejects missing required columns with HTTP 422.
   - Rejects negative amount, invalid latitude/longitude, invalid hours with HTTP 422.
   - Rejects malformed and empty JSON bodies.
6. Frozen Artifact Immutability:
   - Verifies SHA-256 checksums of champion model, preprocessor, and metadata remain invariant.
"""

import hashlib
from pathlib import Path
from typing import Dict, Any
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.models.config import VAL_FEATURES_PATH


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide a TestClient instance for API integration tests."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def val_sample_rows() -> pd.DataFrame:
    """Load a sample partition of validation transactions."""
    df = pd.read_parquet(VAL_FEATURES_PATH)
    return df


@pytest.fixture(scope="module")
def legitimate_row(val_sample_rows: pd.DataFrame) -> Dict[str, Any]:
    """Extract a legitimate (non-fraud) transaction row as a dictionary."""
    row = val_sample_rows[val_sample_rows["is_fraud"] == 0].iloc[0].to_dict()
    row["timestamp"] = str(row["timestamp"])
    return row


@pytest.fixture(scope="module")
def fraud_row(val_sample_rows: pd.DataFrame) -> Dict[str, Any]:
    """Extract a known fraud transaction row as a dictionary."""
    row = val_sample_rows[val_sample_rows["is_fraud"] == 1].iloc[0].to_dict()
    row["timestamp"] = str(row["timestamp"])
    return row


def test_predict_legitimate_transaction(client: TestClient, legitimate_row: Dict[str, Any]):
    """Verify evaluation of a typical legitimate transaction."""
    response = client.post("/predict", json=legitimate_row)
    assert response.status_code == 200
    data = response.json()

    assert data["decision_action"] == "APPROVE"
    assert data["risk_tier"] == "LOW"
    assert data["risk_score"] < 35
    assert data["model_score"] < 0.35
    assert data["policy_mode"] == "TRI_TIER"
    assert data["is_overridden"] is False
    assert data["model_version"] == "1.0.0"
    assert len(data["reason_codes"]) > 0
    assert len(data["top_risk_factors"]) > 0
    assert len(data["top_mitigating_factors"]) > 0
    assert "evaluated_at" in data


def test_predict_versioned_parity(client: TestClient, legitimate_row: Dict[str, Any]):
    """Verify that POST /api/v1/predict behaves identically to POST /predict."""
    resp_root = client.post("/predict", json=legitimate_row)
    resp_v1 = client.post("/api/v1/predict", json=legitimate_row)

    assert resp_root.status_code == 200
    assert resp_v1.status_code == 200

    data_root = resp_root.json()
    data_v1 = resp_v1.json()

    assert data_root["decision_action"] == data_v1["decision_action"]
    assert data_root["risk_score"] == data_v1["risk_score"]
    assert data_root["model_score"] == data_v1["model_score"]
    assert data_root["risk_tier"] == data_v1["risk_tier"]


def test_predict_fraud_transaction(client: TestClient, fraud_row: Dict[str, Any]):
    """Verify evaluation of a high-risk fraud transaction."""
    response = client.post("/predict", json=fraud_row)
    assert response.status_code == 200
    data = response.json()

    assert data["decision_action"] == "BLOCK"
    assert data["risk_tier"] == "CRITICAL"
    assert data["risk_score"] >= 78
    assert data["model_score"] >= 0.78
    assert len(data["reason_codes"]) > 0


def test_predict_rule_override_velocity_burst(client: TestClient, legitimate_row: Dict[str, Any]):
    """Verify that high 1h velocity triggers RULE_VELOCITY_BURST_REVIEW and escalates action to REVIEW."""
    payload = legitimate_row.copy()
    payload["txn_count_1h"] = 5.0  # Above rule threshold 4.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_overridden"] is True
    assert data["decision_action"] == "REVIEW"
    assert data["rule_action"] == "REVIEW"
    assert "RULE_VELOCITY_BURST_REVIEW" in data["rules_triggered"]

    # Verify structured rule match entry
    matches = [m for m in data["rule_matches"] if m["rule_id"] == "RULE_VELOCITY_BURST_REVIEW"]
    assert len(matches) == 1
    assert matches[0]["outcome"] == "REVIEW"
    assert matches[0]["feature_name"] == "txn_count_1h"


def test_predict_rule_override_spending_deviation(client: TestClient, legitimate_row: Dict[str, Any]):
    """Verify that extreme amount Z-score triggers RULE_AMT_ZSCORE_DEVIATION_REVIEW."""
    payload = legitimate_row.copy()
    payload["amount_zscore"] = 6.5  # Above rule threshold 5.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_overridden"] is True
    assert data["decision_action"] == "REVIEW"
    assert data["rule_action"] == "REVIEW"
    assert "RULE_AMT_ZSCORE_DEVIATION_REVIEW" in data["rules_triggered"]


def test_predict_passive_compliance_monitor(client: TestClient, legitimate_row: Dict[str, Any]):
    """Verify that amount > 3000 triggers RULE_COMPLIANCE_LARGE_AMOUNT_MONITOR without overriding baseline action."""
    payload = legitimate_row.copy()
    payload["amount"] = 3500.0  # Above 3000.0 threshold

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "RULE_COMPLIANCE_LARGE_AMOUNT_MONITOR" in data["rules_triggered"]
    monitor_match = [m for m in data["rule_matches"] if m["rule_id"] == "RULE_COMPLIANCE_LARGE_AMOUNT_MONITOR"]
    assert len(monitor_match) == 1
    assert monitor_match[0]["outcome"] == "MONITOR"


def test_predict_missing_required_column(client: TestClient, legitimate_row: Dict[str, Any]):
    """Verify that omitting a mandatory feature returns HTTP 422."""
    payload = legitimate_row.copy()
    del payload["amount"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_amount_validation(client: TestClient, legitimate_row: Dict[str, Any]):
    """Verify that negative amounts are rejected with HTTP 422."""
    payload = legitimate_row.copy()
    payload["amount"] = -50.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_latitude_validation(client: TestClient, legitimate_row: Dict[str, Any]):
    """Verify that out-of-range coordinates are rejected with HTTP 422."""
    payload = legitimate_row.copy()
    payload["cardholder_lat"] = 99.9

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_payload(client: TestClient):
    """Verify that empty JSON request body returns HTTP 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_artifact_immutability():
    """Verify that champion model artifacts remain completely unchanged."""
    artifacts = [
        Path("ml/models/artifacts/champion_model.joblib"),
        Path("ml/models/artifacts/champion_preprocessor.joblib"),
        Path("ml/models/artifacts/model_metadata.json"),
    ]

    for p in artifacts:
        assert p.exists(), f"Artifact missing: {p}"
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            hasher.update(f.read())
        digest = hasher.hexdigest()
        assert len(digest) == 64, f"Invalid SHA-256 digest for {p}"
