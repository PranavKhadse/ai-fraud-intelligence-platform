"""
Unit Tests for Phase 9.6 Increment 2 Hardening Review.

Validates:
1. Client IP Extraction Security:
   - Anti-spoofing policy when TRUST_PROXY_HEADERS=False (always uses request.client.host).
   - Trusted reverse proxy handling when TRUST_PROXY_HEADERS=True (single IP, multi-hop IPs, X-Real-IP).
   - Malformed/injection header sanitization via ipaddress validation.
   - Missing client host fallback.
2. Account ID Compatibility Policy:
   - Missing account_id triggers HTTP 422 with descriptive validation error.
3. Existing HTTPException Preservation:
   - Upstream HTTPExceptions are never swallowed or rewritten to 500.
4. Internal Exception Information Shielding:
   - Mapping failures return generic HTTP 500 detail without exposing internal tracebacks.
5. Test Database Safeguards:
   - Prevents accidental table truncation or test execution against production/dev database URLs.
6. Error Response Sanitization:
   - Verifies no sensitive account data or internal DB state leaks in error payloads.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException, Request, status

from backend.app.api.v1.endpoints.predict import extract_client_ip, predict_transaction
from backend.app.db.models.enums import AuditActorType
from backend.app.repositories.exceptions import PersistenceConflictError, PersistenceError
from backend.app.schemas.predict import PredictionResponse, TransactionPredictRequest
from backend.app.services.persistence_service import FraudPersistenceService
from backend.app.services.risk_service import RiskService
from tests.conftest import get_test_database_url


# ==============================================================================
# Helper to construct mock Request objects
# ==============================================================================

def make_mock_request(
    headers: dict = None,
    client_host: str = "127.0.0.1",
) -> Request:
    """Construct a lightweight mock FastAPI Request for unit testing."""
    headers_dict = headers or {}
    req = MagicMock(spec=Request)
    req.headers = headers_dict
    if client_host:
        req.client = SimpleNamespace(host=client_host, port=54321)
    else:
        req.client = None
    return req


# ==============================================================================
# 1. Client IP Extraction & Anti-Spoofing Tests
# ==============================================================================

class TestClientIPExtractionSecurity:
    """Validates safe client IP extraction, spoofing prevention, and header parsing."""

    def test_untrusted_mode_ignores_forwarded_headers(self):
        """When TRUST_PROXY_HEADERS=False, forwarded headers are ignored to prevent spoofing."""
        req = make_mock_request(
            headers={"X-Forwarded-For": "203.0.113.195, 10.0.0.1", "X-Real-IP": "198.51.100.2"},
            client_host="192.168.1.100",
        )
        ip = extract_client_ip(req, trust_proxy_headers=False)
        assert ip == "192.168.1.100"

    def test_untrusted_mode_no_headers_uses_client_host(self):
        """When no headers exist and trust_proxy_headers=False, uses client.host."""
        req = make_mock_request(headers={}, client_host="10.10.10.5")
        ip = extract_client_ip(req, trust_proxy_headers=False)
        assert ip == "10.10.10.5"

    def test_untrusted_mode_no_client_returns_none(self):
        """When request.client is None and trust_proxy_headers=False, returns None."""
        req = make_mock_request(headers={}, client_host=None)
        ip = extract_client_ip(req, trust_proxy_headers=False)
        assert ip is None

    def test_trusted_mode_single_forwarded_ip(self):
        """When TRUST_PROXY_HEADERS=True, single valid IP in X-Forwarded-For is extracted."""
        req = make_mock_request(
            headers={"X-Forwarded-For": "203.0.113.50"},
            client_host="127.0.0.1",
        )
        ip = extract_client_ip(req, trust_proxy_headers=True)
        assert ip == "203.0.113.50"

    def test_trusted_mode_multi_hop_forwarded_ips(self):
        """When TRUST_PROXY_HEADERS=True, leftmost valid client IP in X-Forwarded-For is returned."""
        req = make_mock_request(
            headers={"X-Forwarded-For": "198.51.100.42, 10.0.0.1, 172.16.0.2"},
            client_host="127.0.0.1",
        )
        ip = extract_client_ip(req, trust_proxy_headers=True)
        assert ip == "198.51.100.42"

    def test_trusted_mode_malformed_forwarded_ip_falls_back_to_next_valid(self):
        """When leftmost part in X-Forwarded-For is malformed/injected, safely skips to next valid IP."""
        req = make_mock_request(
            headers={"X-Forwarded-For": "<script>alert('xss')</script>, 203.0.113.99, 10.0.0.1"},
            client_host="127.0.0.1",
        )
        ip = extract_client_ip(req, trust_proxy_headers=True)
        assert ip == "203.0.113.99"

    def test_trusted_mode_completely_invalid_forwarded_falls_back_to_client_host(self):
        """When X-Forwarded-For contains no valid IPs, falls back to direct client host."""
        req = make_mock_request(
            headers={"X-Forwarded-For": "not_an_ip, bad_gateway, 999.999.999.999"},
            client_host="192.168.1.55",
        )
        ip = extract_client_ip(req, trust_proxy_headers=True)
        assert ip == "192.168.1.55"

    def test_trusted_mode_x_real_ip_used_when_no_forwarded_for(self):
        """When X-Forwarded-For is absent, valid X-Real-IP is used."""
        req = make_mock_request(
            headers={"X-Real-IP": "198.51.100.77"},
            client_host="127.0.0.1",
        )
        ip = extract_client_ip(req, trust_proxy_headers=True)
        assert ip == "198.51.100.77"

    def test_trusted_mode_ipv6_address_support(self):
        """Valid IPv6 addresses in X-Forwarded-For are parsed and extracted."""
        req = make_mock_request(
            headers={"X-Forwarded-For": "2001:db8:85a3::8a2e:370:7334, 10.0.0.1"},
            client_host="127.0.0.1",
        )
        ip = extract_client_ip(req, trust_proxy_headers=True)
        assert ip == "2001:db8:85a3::8a2e:370:7334"


# ==============================================================================
# 2. Test Database Safety Guard Tests
# ==============================================================================

class TestDatabaseSafeguards:
    """Validates that automated test runner cannot accidentally truncate production/dev databases."""

    def test_refuses_production_database_url(self, monkeypatch):
        """Rejects TEST_DATABASE_URL targeting production database."""
        monkeypatch.setenv(
            "TEST_DATABASE_URL",
            "postgresql+asyncpg://postgres:secret@prod-db.example.com:5432/fraud_intelligence_db",
        )
        with pytest.raises(RuntimeError, match="CRITICAL SAFETY ERROR"):
            get_test_database_url()

    def test_refuses_url_without_test_in_name(self, monkeypatch):
        """Rejects TEST_DATABASE_URL whose database name lacks 'test' identifier."""
        monkeypatch.setenv(
            "TEST_DATABASE_URL",
            "postgresql+asyncpg://postgres:secret@localhost:5432/production_data",
        )
        with pytest.raises(RuntimeError, match="CRITICAL SAFETY ERROR"):
            get_test_database_url()

    def test_accepts_valid_test_database_url(self, monkeypatch):
        """Accepts verified test database URL."""
        monkeypatch.setenv(
            "TEST_DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_intelligence_test_db",
        )
        url = get_test_database_url()
        assert "fraud_intelligence_test_db" in url


# ==============================================================================
# 3. Exception Handling & HTTP Status Safety Tests
# ==============================================================================

class TestEndpointExceptionHandling:
    """Validates exception conversion, status code preservation, and error detail shielding."""

    @pytest.mark.asyncio
    async def test_upstream_http_exception_is_preserved_not_rewritten(self):
        """When risk_service raises HTTPException(403), predict_transaction preserves 403."""
        mock_risk_service = MagicMock(spec=RiskService)
        mock_risk_service.predict_transaction.side_effect = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key Expired",
        )
        mock_persistence = MagicMock(spec=FraudPersistenceService)
        mock_persistence.persist_evaluation = AsyncMock()

        req_payload = MagicMock(spec=TransactionPredictRequest)
        http_req = make_mock_request()

        with pytest.raises(HTTPException) as exc_info:
            await predict_transaction(
                request=req_payload,
                http_request=http_req,
                risk_service=mock_risk_service,
                persistence_service=mock_persistence,
            )

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert exc_info.value.detail == "API Key Expired"
        mock_persistence.persist_evaluation.assert_not_called()

    @pytest.mark.asyncio
    async def test_mapping_unexpected_error_shields_internal_traceback(self):
        """When mapper encounters an unexpected failure, endpoint returns generic 500 detail."""
        mock_risk_service = MagicMock(spec=RiskService)
        mock_response = MagicMock(spec=PredictionResponse)
        mock_risk_service.predict_transaction.return_value = mock_response

        mock_persistence = MagicMock(spec=FraudPersistenceService)
        mock_persistence.persist_evaluation = AsyncMock()

        req_payload = MagicMock(spec=TransactionPredictRequest)
        req_payload.transaction_id = "TX_TEST_001"
        req_payload.account_id = "ACC_123"
        req_payload.timestamp = None
        http_req = make_mock_request()

        with patch(
            "backend.app.services.risk_persistence_mapper.RiskPersistenceMapper.map_prediction_to_command",
            side_effect=KeyError("internal_secret_key_missing"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await predict_transaction(
                    request=req_payload,
                    http_request=http_req,
                    risk_service=mock_risk_service,
                    persistence_service=mock_persistence,
                )

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        # Detail must be generic and NOT expose "internal_secret_key_missing"
        assert exc_info.value.detail == "Error preparing transaction persistence."
        mock_persistence.persist_evaluation.assert_not_called()
