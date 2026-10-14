"""
Unit Tests for Phase 15.2: Authentication & Security Utilities.

Validates:
1. Bcrypt password hashing, verification, length constraints, and malformed hash resilience.
2. Secure API-key generation, prefix validation, and constant-time SHA-256 verification.
3. JWT access and refresh token creation, claim encoding, and decode validation.
4. Role isolation: access token creation strictly forbids non-human roles (SYSTEM, API_CLIENT).
5. Refresh token rotation, cryptographic revocation tracking, and replay attack prevention.
6. Token signature tampering and expiration rejection (HTTP 401).
7. Production configuration security validation: prevents weak or default secrets in production.
"""

from datetime import timedelta
import time
import uuid
from fastapi import HTTPException
from pydantic import ValidationError
import pytest

from backend.app.core.config import Settings
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    rotate_refresh_token,
    token_revocation_store,
    verify_api_key,
    verify_password,
)
from backend.app.db.models.enums import AuditActorType, UserRole


# ==============================================================================
# 1. Password Hashing & Verification Tests
# ==============================================================================

class TestPasswordHashing:
    """Validates bcrypt password hashing and verification."""

    def test_hash_password_success(self):
        hashed = hash_password("SuperSecret123!")
        assert hashed.startswith("$2b$")
        assert len(hashed) >= 60

    def test_verify_password_matching(self):
        password = "ValidPassword_2026!"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_password_non_matching(self):
        hashed = hash_password("CorrectPassword123")
        assert verify_password("WrongPassword456", hashed) is False

    def test_verify_password_empty_or_corrupted(self):
        assert verify_password("", "$2b$12$abcdefghijklmnopqrstuv") is False
        assert verify_password("somepassword", "") is False
        assert verify_password("somepassword", "not_a_bcrypt_hash") is False

    def test_hash_password_empty_raises(self):
        with pytest.raises(ValueError, match="Password cannot be empty"):
            hash_password("")

    def test_hash_password_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 8 characters"):
            hash_password("short1")

    def test_hash_password_too_long_raises(self):
        long_password = "A" * 73
        with pytest.raises(ValueError, match="cannot exceed 72 bytes"):
            hash_password(long_password)


# ==============================================================================
# 2. API Key Generation & Verification Tests
# ==============================================================================

class TestAPIKeyUtilities:
    """Validates API-key generation, hashing, and verification."""

    def test_generate_api_key_format(self):
        raw_key, hashed_key = generate_api_key(prefix="pk_live_")
        assert raw_key.startswith("pk_live_")
        assert len(raw_key) > 40
        assert len(hashed_key) == 64
        # Assert hashed_key matches raw_key digest
        assert hash_api_key(raw_key) == hashed_key

    def test_generate_api_key_custom_prefix_len_validation(self):
        with pytest.raises(ValueError, match="must be exactly 8 characters"):
            generate_api_key(prefix="too_long_prefix_")

        with pytest.raises(ValueError, match="must be exactly 8 characters"):
            generate_api_key(prefix="pk_")

    def test_verify_api_key_matching(self):
        raw_key, hashed_key = generate_api_key(prefix="pk_live_")
        assert verify_api_key(raw_key, hashed_key) is True

    def test_verify_api_key_non_matching(self):
        raw_key1, hashed_key1 = generate_api_key(prefix="pk_live_")
        raw_key2, _ = generate_api_key(prefix="pk_live_")
        assert verify_api_key(raw_key2, hashed_key1) is False

    def test_verify_api_key_empty(self):
        assert verify_api_key("", "some_hash") is False
        assert verify_api_key("pk_live_key", "") is False


# ==============================================================================
# 3. JWT Access & Refresh Token Tests
# ==============================================================================

class TestJWTTokenCreationAndValidation:
    """Validates JWT token issuance, claims, and role isolation."""

    def test_create_and_decode_access_token_analyst(self):
        user_id = uuid.uuid4()
        token = create_access_token(subject=user_id, role=UserRole.ANALYST)
        payload = decode_token(token, expected_type="access")

        assert payload["sub"] == str(user_id)
        assert payload["role"] == "ANALYST"
        assert payload["token_type"] == "access"
        assert "exp" in payload
        assert "jti" in payload

    def test_create_and_decode_access_token_admin(self):
        user_id = uuid.uuid4()
        token = create_access_token(subject=user_id, role=UserRole.ADMIN)
        payload = decode_token(token, expected_type="access")

        assert payload["sub"] == str(user_id)
        assert payload["role"] == "ADMIN"
        assert payload["token_type"] == "access"

    def test_create_access_token_role_isolation_rejects_non_human(self):
        user_id = uuid.uuid4()

        # SYSTEM role must be rejected
        with pytest.raises(ValueError, match="not a permitted human role"):
            create_access_token(subject=user_id, role=AuditActorType.SYSTEM)

        # API_CLIENT role must be rejected
        with pytest.raises(ValueError, match="not a permitted human role"):
            create_access_token(subject=user_id, role=AuditActorType.API_CLIENT)

        # Arbitrary roles must be rejected
        with pytest.raises(ValueError, match="not a permitted human role"):
            create_access_token(subject=user_id, role="AUTOMATION")

    def test_decode_token_expired_raises_401(self):
        user_id = uuid.uuid4()
        # Create token already expired in the past
        token = create_access_token(
            subject=user_id,
            role=UserRole.ANALYST,
            expires_delta=timedelta(seconds=-10),
        )

        with pytest.raises(HTTPException) as exc_info:
            decode_token(token, expected_type="access")
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_decode_token_tampered_signature_raises_401(self):
        user_id = uuid.uuid4()
        token = create_access_token(subject=user_id, role=UserRole.ANALYST)
        # Tamper token signature
        tampered = token[:-4] + "abcd"

        with pytest.raises(HTTPException) as exc_info:
            decode_token(tampered, expected_type="access")
        assert exc_info.value.status_code == 401
        assert "invalid" in exc_info.value.detail.lower()

    def test_decode_token_wrong_type_raises_401(self):
        user_id = uuid.uuid4()
        token = create_access_token(subject=user_id, role=UserRole.ANALYST)

        with pytest.raises(HTTPException) as exc_info:
            decode_token(token, expected_type="refresh")
        assert exc_info.value.status_code == 401
        assert "invalid token type" in exc_info.value.detail.lower()


# ==============================================================================
# 4. Refresh Token Rotation & Revocation Tests
# ==============================================================================

class TestRefreshTokenRotationAndRevocation:
    """Validates refresh token rotation, JTI revocation, and replay prevention."""

    def setup_method(self):
        token_revocation_store.reset()

    def teardown_method(self):
        token_revocation_store.reset()

    def test_create_and_decode_refresh_token(self):
        user_id = uuid.uuid4()
        token, jti = create_refresh_token(subject=user_id)
        payload = decode_token(token, expected_type="refresh")

        assert payload["sub"] == str(user_id)
        assert payload["token_type"] == "refresh"
        assert payload["jti"] == jti
        assert token_revocation_store.is_jti_revoked(jti) is False

    def test_rotate_refresh_token_revokes_old_jti(self):
        user_id = uuid.uuid4()
        old_token, old_jti = create_refresh_token(subject=user_id)

        # Rotate token
        new_token, new_jti = rotate_refresh_token(old_token, subject=user_id)

        assert old_jti != new_jti
        assert token_revocation_store.is_jti_revoked(old_jti) is True
        assert token_revocation_store.is_jti_revoked(new_jti) is False

        # Attempting to decode old rotated token raises 401 (revoked)
        with pytest.raises(HTTPException) as exc_info:
            decode_token(old_token, expected_type="refresh")
        assert exc_info.value.status_code == 401
        assert "revoked" in exc_info.value.detail.lower()

        # Decoding new rotated token succeeds
        new_payload = decode_token(new_token, expected_type="refresh")
        assert new_payload["jti"] == new_jti

    def test_replaying_revoked_refresh_token_fails(self):
        user_id = uuid.uuid4()
        token, jti = create_refresh_token(subject=user_id)
        token_revocation_store.revoke_jti(jti)

        with pytest.raises(HTTPException) as exc_info:
            rotate_refresh_token(token, subject=user_id)
        assert exc_info.value.status_code == 401
        assert "revoked" in exc_info.value.detail.lower()


# ==============================================================================
# 5. Production Configuration Security Tests
# ==============================================================================

class TestProductionSettingsSecurity:
    """Validates safe configuration enforcement for production environments."""

    def test_production_rejects_default_insecure_jwt_secret(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                JWT_SECRET_KEY="dev-insecure-jwt-secret-key-change-in-production-min-32-chars",
            )
        assert "JWT_SECRET_KEY must be a cryptographically secure" in str(exc_info.value)

    def test_production_rejects_short_jwt_secret(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                JWT_SECRET_KEY="short-secret-key",
            )
        assert "at least 32 characters" in str(exc_info.value)

    def test_production_rejects_common_placeholder_secret(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                JWT_SECRET_KEY="password",
            )
        assert "Production Security Violation" in str(exc_info.value)

    def test_production_rejects_allow_dev_actor_headers(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                JWT_SECRET_KEY="a" * 40,
                ALLOW_DEV_ACTOR_HEADERS=True,
            )
        assert "ALLOW_DEV_ACTOR_HEADERS must be False in production" in str(exc_info.value)

    def test_production_accepts_strong_secret_and_secure_config(self):
        strong_secret = "c7d8e9f0123456789abcdef0123456789abcdef0123456789abcdef012345678"
        s = Settings(
            ENVIRONMENT="production",
            JWT_SECRET_KEY=strong_secret,
            ALLOW_DEV_ACTOR_HEADERS=False,
            CORS_ORIGINS=["https://fraud-analytics.internal"],
        )
        assert s.JWT_SECRET_KEY == strong_secret
        assert s.ENVIRONMENT == "production"
        assert "*" not in s.CORS_ORIGINS

    def test_development_permits_default_secret_for_local_ergonomics(self):
        s = Settings(
            ENVIRONMENT="development",
            ALLOW_DEV_ACTOR_HEADERS=True,
        )
        assert s.ENVIRONMENT == "development"
        assert len(s.JWT_SECRET_KEY) >= 32
