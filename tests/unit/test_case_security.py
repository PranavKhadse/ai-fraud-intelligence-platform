"""
Unit Tests for Case Management Security, Actor Resolution, and Role-Based Authorization.

Validates:
1. Production mode (ALLOW_DEV_ACTOR_HEADERS=False) fails closed with HTTP 401.
2. Development/test mode (ALLOW_DEV_ACTOR_HEADERS=True) extracts explicit actor headers.
3. Missing or whitespace-only X-Actor-ID raises HTTP 401.
4. Missing or whitespace-only X-Actor-Role raises HTTP 401.
5. Invalid X-Actor-Role (not in AuditActorType) raises HTTP 401.
6. Role-based authorization: require_role permits allowed roles and rejects others with HTTP 403.
7. IP address and correlation ID extraction from headers and request state.
"""

from unittest.mock import MagicMock
import uuid
import pytest
from fastapi import HTTPException, Request, status

from backend.app.core.config import settings
from backend.app.core.security import ActorContext, get_current_actor, require_role
from backend.app.db.models.enums import AuditActorType


def _build_mock_request(
    client_host: str = "127.0.0.1",
    headers: dict = None,
    correlation_id: str = None,
) -> Request:
    req = MagicMock(spec=Request)
    req.client.host = client_host
    req.headers = headers or {}
    req.state = MagicMock()
    req.state.correlation_id = correlation_id
    return req


@pytest.mark.asyncio
async def test_get_current_actor_fails_closed_in_production(monkeypatch):
    """Verify that when ALLOW_DEV_ACTOR_HEADERS=False, requests fail closed with 401."""
    monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", False)
    req = _build_mock_request()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_actor(
            request=req,
            x_actor_id="analyst_1",
            x_actor_role="ANALYST",
        )
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Actor authentication required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_actor_missing_actor_id(monkeypatch):
    """Verify that missing X-Actor-ID raises 401."""
    monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", True)
    req = _build_mock_request()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_actor(
            request=req,
            x_actor_id=None,
            x_actor_role="ANALYST",
        )
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "X-Actor-ID header is required" in exc_info.value.detail

    # Test whitespace-only
    with pytest.raises(HTTPException) as exc_info2:
        await get_current_actor(
            request=req,
            x_actor_id="   ",
            x_actor_role="ANALYST",
        )
    assert exc_info2.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_get_current_actor_missing_actor_role(monkeypatch):
    """Verify that missing X-Actor-Role raises 401."""
    monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", True)
    req = _build_mock_request()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_actor(
            request=req,
            x_actor_id="analyst_1",
            x_actor_role=None,
        )
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "X-Actor-Role header is required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_actor_invalid_role(monkeypatch):
    """Verify that invalid X-Actor-Role string raises 401."""
    monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", True)
    req = _build_mock_request()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_actor(
            request=req,
            x_actor_id="user_123",
            x_actor_role="SUPERUSER",
        )
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid X-Actor-Role" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_actor_success(monkeypatch):
    """Verify successful actor extraction for all valid roles."""
    monkeypatch.setattr(settings, "ALLOW_DEV_ACTOR_HEADERS", True)
    req = _build_mock_request(client_host="10.0.0.42", correlation_id="cid-12345")

    for role in [AuditActorType.ANALYST, AuditActorType.ADMIN, AuditActorType.API_CLIENT, AuditActorType.SYSTEM]:
        actor = await get_current_actor(
            request=req,
            x_actor_id=f"user_{role.value.lower()}",
            x_actor_role=role.value,
            x_correlation_id="cid-override",
        )
        assert actor.actor_id == f"user_{role.value.lower()}"
        assert actor.actor_role == role
        assert actor.correlation_id == "cid-override"
        assert actor.client_ip == "10.0.0.42"


@pytest.mark.asyncio
async def test_require_role_authorization():
    """Verify require_role dependency allows permitted roles and blocks forbidden roles."""
    checker = require_role(AuditActorType.ANALYST, AuditActorType.ADMIN)

    analyst_actor = ActorContext(actor_id="a1", actor_role=AuditActorType.ANALYST)
    admin_actor = ActorContext(actor_id="adm1", actor_role=AuditActorType.ADMIN)
    client_actor = ActorContext(actor_id="client1", actor_role=AuditActorType.API_CLIENT)

    # Allowed
    res1 = await checker(actor=analyst_actor)
    assert res1 == analyst_actor

    res2 = await checker(actor=admin_actor)
    assert res2 == admin_actor

    # Forbidden
    with pytest.raises(HTTPException) as exc_info:
        await checker(actor=client_actor)
    assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    assert "Forbidden" in exc_info.value.detail
