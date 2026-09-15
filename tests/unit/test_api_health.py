"""
Unit Tests for Health Check and Root Service Endpoints.

Validates:
1. GET /health returns HTTP 200 with complete HealthResponse schema.
2. GET /api/v1/health returns HTTP 200 with identical telemetry.
3. GET / returns HTTP 200 with API service information and doc links.
4. Model loaded and rule catalog count integrity.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide a TestClient instance for testing."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Verify that GET / returns service information and doc paths."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["documentation"] == "/docs"
    assert data["health_endpoint"] == "/health"
    assert data["predict_endpoint"] == "/predict"
    assert data["api_v1_prefix"] == "/api/v1"


def test_root_health_endpoint(client: TestClient):
    """Verify that GET /health returns HTTP 200 and valid telemetry."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"
    assert data["rules_loaded_count"] == 6
    assert "timestamp" in data
    assert "app_name" in data


def test_api_v1_health_endpoint(client: TestClient):
    """Verify that GET /api/v1/health returns HTTP 200 and matches root health."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"
    assert data["rules_loaded_count"] == 6
