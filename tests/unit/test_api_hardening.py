"""
Unit Tests for Phase 15.6: Rate Limiting & API Hardening.

Validates:
1. Rate Limiting:
   - Dedicated login rate limit (429 on exceeding RATE_LIMIT_LOGIN_PER_MINUTE).
   - General API rate limit (429 on exceeding RATE_LIMIT_API_PER_MINUTE).
   - Rate limit headers (Retry-After, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset).
   - Exemption of documentation and health check endpoints.
   - API key quota separation (different keys get independent buckets).
   - Anti-spoofing client IP resolution:
     - TRUST_PROXY_HEADERS=False: X-Forwarded-For is ignored, spoofing fails.
     - TRUST_PROXY_HEADERS=True: X-Forwarded-For is respected.
   - RATE_LIMIT_ENABLED=False bypasses rate limiting.
2. Security Response Headers:
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: DENY
   - X-XSS-Protection: 0
   - Referrer-Policy
   - Content-Security-Policy
   - Permissions-Policy
   - Strict-Transport-Security (when HSTS enabled)
   - Headers present on successful (200) and error (401, 413, 429) responses.
3. CORS Hardening:
   - Rejects wildcard origins ('*') in production.
   - Accepts explicit origins in production.
   - Allows wildcard origins in development.
4. Payload Size Limiting:
   - Payloads within limit are processed normally.
   - Content-Length exceeding MAX_REQUEST_BODY_BYTES returns HTTP 413.
   - Streamed/chunked body exceeding limit returns HTTP 413.
"""

import time
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, settings
from backend.app.main import app
from backend.app.middleware.rate_limit import (
    InMemoryRateLimiter,
    RateLimitMiddleware,
    global_rate_limiter,
)
from backend.app.middleware.payload_limit import PayloadLimitMiddleware
from backend.app.services.unit_of_work import get_unit_of_work


class InMemoryUserRepository:
    def __init__(self, users=None):
        self._users = {u.username: u for u in (users or [])}

    async def get_by_username(self, username: str):
        return self._users.get(username)

    async def get_by_id(self, uid):
        return None

    async def update_last_login(self, uid):
        pass


class MockUnitOfWork:
    def __init__(self, user_repo=None):
        self.users = user_repo or InMemoryUserRepository()
        self.session = None

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


from backend.app.db.session import get_db_session
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(autouse=True)
def setup_mock_db_session():
    """Mock get_db_session so endpoint dependencies do not connect to postgres."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalars.return_value.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    async def override_db_session():
        yield mock_session

    app.dependency_overrides[get_db_session] = override_db_session
    yield
    app.dependency_overrides.pop(get_db_session, None)


@pytest.fixture(autouse=True)
def setup_mock_uow():
    """Mock database unit of work so auth requests don't hit postgres."""
    mock_uow = MockUnitOfWork()

    async def override_uow():
        yield mock_uow

    app.dependency_overrides[get_unit_of_work] = override_uow
    yield
    app.dependency_overrides.pop(get_unit_of_work, None)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Ensure in-memory rate limiter buckets are clean before and after every test."""
    global_rate_limiter.clear()
    yield
    global_rate_limiter.clear()


# ==============================================================================
# 1. Rate Limiting Tests
# ==============================================================================

class TestRateLimiting:
    """Validates sliding window rate limiting across login and general endpoints."""

    def test_rate_limit_headers_on_allowed_requests(self):
        client = TestClient(app)
        res = client.get("/api/v1/auth/me")
        # Unauthenticated request returns 401, but rate limit headers are still attached
        assert "x-ratelimit-limit" in res.headers
        assert "x-ratelimit-remaining" in res.headers
        assert "x-ratelimit-reset" in res.headers
        assert int(res.headers["x-ratelimit-remaining"]) >= 0

    def test_login_rate_limit_enforced_with_429(self, monkeypatch):
        # Set a small limit of 3 for fast testing
        monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_MINUTE", 3)
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)

        client = TestClient(app)
        payload = {"username": "attacker", "password": "WrongPassword123!"}

        # First 3 attempts should reach auth logic (401 invalid credentials)
        for i in range(3):
            res = client.post("/api/v1/auth/login", json=payload)
            assert res.status_code == 401, f"Attempt {i+1} should return 401"
            assert res.headers["x-ratelimit-limit"] == "3"
            assert int(res.headers["x-ratelimit-remaining"]) == 2 - i

        # 4th attempt should be blocked by rate limiter with 429
        res_blocked = client.post("/api/v1/auth/login", json=payload)
        assert res_blocked.status_code == 429
        data = res_blocked.json()
        assert "Rate limit exceeded" in data["detail"]
        assert "retry_after" in data["detail"].lower() or "retry after" in data["detail"].lower()

        # Check required standard headers on 429 response
        assert "retry-after" in res_blocked.headers
        assert int(res_blocked.headers["retry-after"]) >= 1
        assert res_blocked.headers["x-ratelimit-limit"] == "3"
        assert res_blocked.headers["x-ratelimit-remaining"] == "0"
        assert "x-ratelimit-reset" in res_blocked.headers

    def test_general_api_rate_limit_enforced_with_429(self, monkeypatch):
        # Set general API limit to 4 for testing
        monkeypatch.setattr(settings, "RATE_LIMIT_API_PER_MINUTE", 4)
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)

        client = TestClient(app)

        # 4 allowed requests
        for i in range(4):
            res = client.get("/api/v1/auth/me")
            assert res.status_code in (401, 200)
            assert int(res.headers["x-ratelimit-remaining"]) == 3 - i

        # 5th request blocked
        blocked = client.get("/api/v1/auth/me")
        assert blocked.status_code == 429
        assert blocked.headers["x-ratelimit-remaining"] == "0"
        assert "retry-after" in blocked.headers
        assert "Rate limit exceeded" in blocked.json()["detail"]

    def test_exempt_endpoints_not_rate_limited(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_API_PER_MINUTE", 2)
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)

        client = TestClient(app)

        # Health endpoint should never be rate limited
        for _ in range(10):
            res = client.get("/health")
            assert res.status_code == 200

        # Root information endpoint should not be rate limited
        for _ in range(10):
            res = client.get("/")
            assert res.status_code == 200

    def test_api_key_quota_isolation(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_API_PER_MINUTE", 2)
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)

        client = TestClient(app)

        # Exhaust quota for Key A
        client.get("/api/v1/cases", headers={"X-API-Key": "key_alpha_secret_12345678"})
        client.get("/api/v1/cases", headers={"X-API-Key": "key_alpha_secret_12345678"})
        blocked_a = client.get("/api/v1/cases", headers={"X-API-Key": "key_alpha_secret_12345678"})
        assert blocked_a.status_code == 429

        # Key B should still have full quota despite Key A being rate limited
        ok_b = client.get("/api/v1/cases", headers={"X-API-Key": "key_beta_secret_98765432"})
        assert ok_b.status_code != 429
        assert ok_b.headers["x-ratelimit-remaining"] == "1"

    def test_rate_limit_disabled_flag(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_MINUTE", 2)
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)

        client = TestClient(app)
        payload = {"username": "testuser", "password": "TestPassword123!"}

        # 5 requests with limit 2 but RATE_LIMIT_ENABLED=False
        for _ in range(5):
            res = client.post("/api/v1/auth/login", json=payload)
            assert res.status_code != 429


# ==============================================================================
# 2. Anti-Spoofing & Proxy IP Tests
# ==============================================================================

class TestClientIPAntiSpoofing:
    """Verifies that untrusted clients cannot spoof IPs to bypass rate limits."""

    def test_anti_spoofing_when_trust_proxy_headers_false(self, monkeypatch):
        """
        When TRUST_PROXY_HEADERS is False, X-Forwarded-For headers are ignored.
        Spoofing different IPs does not bypass the rate limit.
        """
        monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_MINUTE", 2)
        monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)

        client = TestClient(app)
        payload = {"username": "victim", "password": "WrongPassword!"}

        # Attacker attempts to spoof different client IPs via X-Forwarded-For
        res1 = client.post(
            "/api/v1/auth/login",
            json=payload,
            headers={"X-Forwarded-For": "198.51.100.1"},
        )
        assert res1.status_code == 401

        res2 = client.post(
            "/api/v1/auth/login",
            json=payload,
            headers={"X-Forwarded-For": "198.51.100.2"},
        )
        assert res2.status_code == 401

        # 3rd request with yet another spoofed IP should still be blocked (all share socket client host)
        res3 = client.post(
            "/api/v1/auth/login",
            json=payload,
            headers={"X-Forwarded-For": "198.51.100.3"},
        )
        assert res3.status_code == 429

    def test_proxy_headers_respected_when_trust_proxy_headers_true(self, monkeypatch):
        """
        When TRUST_PROXY_HEADERS is True (behind trusted load balancer),
        the upstream client IP in X-Forwarded-For is respected.
        """
        monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_MINUTE", 2)
        monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)

        client = TestClient(app)
        payload = {"username": "victim", "password": "WrongPassword!"}

        # IP A makes 2 attempts
        client.post("/api/v1/auth/login", json=payload, headers={"X-Forwarded-For": "203.0.113.10"})
        client.post("/api/v1/auth/login", json=payload, headers={"X-Forwarded-For": "203.0.113.10"})
        blocked_a = client.post(
            "/api/v1/auth/login", json=payload, headers={"X-Forwarded-For": "203.0.113.10"}
        )
        assert blocked_a.status_code == 429

        # IP B should have separate bucket and not be blocked
        ok_b = client.post(
            "/api/v1/auth/login", json=payload, headers={"X-Forwarded-For": "203.0.113.20"}
        )
        assert ok_b.status_code == 401  # Normal credential rejection, not 429 rate limited


# ==============================================================================
# 3. Security Response Headers Tests
# ==============================================================================

class TestSecurityResponseHeaders:
    """Verifies that all standard enterprise security headers are present on responses."""

    def test_security_headers_present_on_success(self):
        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200

        assert res.headers.get("x-content-type-options") == "nosniff"
        assert res.headers.get("x-frame-options") == "DENY"
        assert res.headers.get("x-xss-protection") == "0"
        assert "referrer-policy" in res.headers
        assert "content-security-policy" in res.headers
        assert "permissions-policy" in res.headers
        assert res.headers.get("x-permitted-cross-domain-policies") == "none"

    def test_security_headers_present_on_error_responses(self):
        client = TestClient(app)

        # 401 Unauthorized
        res_401 = client.get("/api/v1/auth/me")
        assert res_401.status_code == 401
        assert res_401.headers.get("x-content-type-options") == "nosniff"
        assert res_401.headers.get("x-frame-options") == "DENY"

        # 404 Not Found
        res_404 = client.get("/api/v1/non-existent-route")
        assert res_404.status_code == 404
        assert res_404.headers.get("x-content-type-options") == "nosniff"
        assert res_404.headers.get("x-frame-options") == "DENY"

    def test_hsts_header_injected_when_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "SECURITY_HSTS_ENABLED", True)

        client = TestClient(app)
        res = client.get("/health")
        assert "strict-transport-security" in res.headers
        assert "max-age=31536000" in res.headers["strict-transport-security"]


# ==============================================================================
# 4. Safe CORS Configuration Tests
# ==============================================================================

class TestSafeCORSConfiguration:
    """Validates production rejection of wildcard origins and safe development behavior."""

    def test_production_rejects_wildcard_cors_origin(self):
        """Production must fail fast if CORS_ORIGINS contains wildcard '*'."""
        with pytest.raises(ValueError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                CORS_ORIGINS=["*"],
                JWT_SECRET_KEY="a" * 32,
            )
        assert "Wildcard CORS origin '*' is strictly prohibited in production" in str(exc_info.value)

    def test_production_accepts_explicit_cors_origins(self):
        """Production must succeed when explicit domain origins are configured."""
        prod_settings = Settings(
            ENVIRONMENT="production",
            CORS_ORIGINS=["https://fraud.example.com", "https://admin.example.com"],
            JWT_SECRET_KEY="a" * 32,
        )
        assert prod_settings.ENVIRONMENT == "production"
        assert "*" not in prod_settings.CORS_ORIGINS

    def test_development_permits_wildcard_cors_origin(self):
        """Development environment is permitted to use wildcard '*' for local development."""
        dev_settings = Settings(
            ENVIRONMENT="development",
            CORS_ORIGINS=["*"],
        )
        assert dev_settings.CORS_ORIGINS == ["*"]


# ==============================================================================
# 5. Request Body Payload Size Limiting Tests
# ==============================================================================

class TestPayloadSizeLimiting:
    """Validates rejection of oversized request bodies (HTTP 413)."""

    def test_normal_payload_accepted(self):
        client = TestClient(app)
        payload = {"username": "analyst_bob", "password": "Password123!"}
        res = client.post("/api/v1/auth/login", json=payload)
        # Should not be rejected with 413
        assert res.status_code != 413

    def test_content_length_exceeding_limit_rejected_with_413(self, monkeypatch):
        # Set max payload size to 256 bytes for testing
        monkeypatch.setattr(settings, "MAX_REQUEST_BODY_BYTES", 256)

        client = TestClient(app)
        oversized_data = b"A" * 512

        res = client.post(
            "/api/v1/auth/login",
            content=oversized_data,
            headers={"Content-Type": "application/json", "Content-Length": "512"},
        )
        assert res.status_code == 413
        assert "Request payload too large" in res.json()["detail"]
        assert "256" in res.json()["detail"]

    def test_streamed_chunked_payload_exceeding_limit_rejected_with_413(self, monkeypatch):
        # Set max payload size to 200 bytes for testing
        monkeypatch.setattr(settings, "MAX_REQUEST_BODY_BYTES", 200)

        client = TestClient(app)
        oversized_data = b"B" * 400

        # Sending without Content-Length triggers streaming byte accumulation
        res = client.post(
            "/api/v1/auth/login",
            content=oversized_data,
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 413
        assert "Request payload too large" in res.json()["detail"]
