"""
In-Memory Rate Limiting Middleware & Client Identity Extraction.

Provides:
- Thread-safe sliding window log rate limiting for login and general API endpoints.
- Authoritative, anti-spoofing client IP extraction respecting TRUST_PROXY_HEADERS.
- Standard IETF rate limit response headers (Retry-After, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset).
- Asynchronous security audit event emission upon rate-limit rejections (HTTP 429).

================================================================================
Architectural Notice — Single-Instance Limitation:
The InMemoryRateLimiter stores bucket state strictly in memory local to the
current application process. In horizontally scaled production architectures
with multiple workers or distributed container instances, an external
distributed cache (such as Redis or Memcached) MUST be used to share rate-limiting
state across instances.
================================================================================
"""

from collections import deque
from dataclasses import dataclass
import hashlib
import logging
import math
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

from starlette.types import ASGIApp, Scope, Receive, Send, Message
from starlette.responses import JSONResponse
from starlette import status

from backend.app.core.config import settings

logger = logging.getLogger("fraud_api.middleware.rate_limit")


@dataclass(frozen=True)
class RateLimitResult:
    """Outcome of a rate limit check evaluation."""
    allowed: bool
    limit: int
    remaining: int
    reset_timestamp: int
    retry_after: int


class InMemoryRateLimiter:
    """
    Thread-safe sliding-window log rate limiter.

    Maintains a rolling window of request timestamps for each tracked key.
    Automatically purges expired entries upon evaluation.
    """

    def __init__(self) -> None:
        self._buckets: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def check(
        self,
        key: str,
        limit: int,
        window_seconds: float = 60.0,
    ) -> RateLimitResult:
        """
        Evaluate whether the given key is allowed to make a request within the sliding window.

        Args:
            key: Unique bucket key (e.g., 'login:192.168.1.1' or 'apikey:3f12a...').
            limit: Maximum allowed requests within the time window.
            window_seconds: Duration of the sliding window in seconds (default 60s).

        Returns:
            RateLimitResult containing allowance status and header metadata.
        """
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            # Periodic cleanup of empty or stale buckets if cache grows large
            if len(self._buckets) > 5000 and (now - self._last_cleanup) > 300.0:
                self._prune_stale_buckets(cutoff)
                self._last_cleanup = now

            timestamps = self._buckets.setdefault(key, deque())

            # Purge entries outside the current sliding window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            current_count = len(timestamps)

            if current_count >= limit:
                # Rate limit exceeded
                oldest_timestamp = timestamps[0]
                retry_after = max(1, int(math.ceil(oldest_timestamp + window_seconds - now)))
                reset_timestamp = int(oldest_timestamp + window_seconds)
                return RateLimitResult(
                    allowed=False,
                    limit=limit,
                    remaining=0,
                    reset_timestamp=reset_timestamp,
                    retry_after=retry_after,
                )

            # Request permitted: record timestamp
            timestamps.append(now)
            remaining = max(0, limit - len(timestamps))
            reset_timestamp = int(now + window_seconds)
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=remaining,
                reset_timestamp=reset_timestamp,
                retry_after=0,
            )

    def _prune_stale_buckets(self, cutoff: float) -> None:
        """Internal helper to remove dead keys from memory."""
        stale_keys = [
            k for k, dq in self._buckets.items()
            if not dq or dq[-1] <= cutoff
        ]
        for k in stale_keys:
            del self._buckets[k]

    def clear(self) -> None:
        """Reset all rate limiting state (useful for test isolation)."""
        with self._lock:
            self._buckets.clear()

    @property
    def active_bucket_count(self) -> int:
        """Return the number of currently tracked rate limit buckets."""
        with self._lock:
            return len(self._buckets)


# Global singleton in-memory rate limiter instance
global_rate_limiter = InMemoryRateLimiter()


def get_client_ip(scope: Scope) -> str:
    """
    Extract the authoritative client IP address respecting the TRUST_PROXY_HEADERS configuration.

    Anti-Spoofing Security Guarantee:
    - If `TRUST_PROXY_HEADERS` is False (default): Proxy headers (`X-Forwarded-For`, `X-Real-IP`)
      are STRICTLY ignored. The socket TCP peer address (`scope['client']`) is used.
      This prevents untrusted external clients from evading rate limits by sending arbitrary IP headers.
    - If `TRUST_PROXY_HEADERS` is True: The first IP in `X-Forwarded-For` (client origin) or
      `X-Real-IP` is extracted. Falls back to socket client IP if missing or invalid.
    """
    headers = dict(scope.get("headers", []))
    client_ip: Optional[str] = None

    if settings.TRUST_PROXY_HEADERS:
        raw_forwarded = headers.get(b"x-forwarded-for")
        if raw_forwarded:
            decoded = raw_forwarded.decode("latin1", errors="ignore")
            first_ip = decoded.split(",")[0].strip()
            if first_ip:
                client_ip = first_ip
        else:
            raw_real_ip = headers.get(b"x-real-ip")
            if raw_real_ip:
                decoded = raw_real_ip.decode("latin1", errors="ignore").strip()
                if decoded:
                    client_ip = decoded

    if not client_ip:
        client = scope.get("client")
        if client and len(client) > 0 and client[0]:
            client_ip = str(client[0]).strip()

    return client_ip or "127.0.0.1"


class RateLimitMiddleware:
    """
    Pure ASGI rate limiting middleware enforcing separate limits for login and API endpoints.

    Features:
    - Dedicated login rate limit (stricter limit against brute force attacks).
    - General API rate limit (keyed by API key, JWT hash, or client IP).
    - Exemption of documentation and health check endpoints.
    - Injects standard X-RateLimit-* and Retry-After headers.
    - Emits security audit events on rate limit breach.
    """

    EXEMPT_PATHS: Set[str] = {
        "/",
        "/health",
        "/api/v1/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    def __init__(
        self,
        app: ASGIApp,
        limiter: Optional[InMemoryRateLimiter] = None,
    ) -> None:
        self.app = app
        self.limiter = limiter or global_rate_limiter

    def _is_exempt(self, path: str) -> bool:
        """Check if path is exempt from rate limiting."""
        if path in self.EXEMPT_PATHS:
            return True
        for prefix in ("/docs", "/redoc", "/openapi.json"):
            if path.startswith(prefix):
                return True
        return False

    async def _emit_security_audit_event(
        self,
        client_ip: str,
        path: str,
        method: str,
        key: str,
        limit: int,
    ) -> None:
        """Asynchronously and defensively record a RATE_LIMIT_EXCEEDED security audit event."""
        try:
            from backend.app.db.session import _async_engine, get_session_factory
            if _async_engine is None:
                return

            from backend.app.services.security_audit import record_security_event
            from backend.app.db.models.enums import AuditActorType, AuditEntityType

            factory = get_session_factory()
            async with factory() as session:
                await record_security_event(
                    session=session,
                    event_type="RATE_LIMIT_EXCEEDED",
                    action="RATE_LIMIT",
                    actor_type=AuditActorType.SYSTEM,
                    actor_id=client_ip,
                    entity_type=AuditEntityType.SYSTEM,
                    client_ip=client_ip,
                    payload={
                        "path": path,
                        "method": method,
                        "client_key": key,
                        "limit": limit,
                    },
                    commit=True,
                )
        except Exception:
            # Fail-safe: do not mask rate limit response if database is unavailable or mocked
            pass

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not settings.RATE_LIMIT_ENABLED:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()

        # Check path exemption
        if self._is_exempt(path):
            await self.app(scope, receive, send)
            return

        client_ip = get_client_ip(scope)
        headers = dict(scope.get("headers", []))

        # Categorize request and determine rate limit parameters
        login_path = f"{settings.API_V1_STR}/auth/login"
        if path == login_path and method == "POST":
            # Login attempts: strictly keyed on client IP with dedicated login threshold
            limit = settings.RATE_LIMIT_LOGIN_PER_MINUTE
            key = f"login:{client_ip}"
        else:
            # General API requests
            limit = settings.RATE_LIMIT_API_PER_MINUTE
            raw_api_key = headers.get(b"x-api-key")
            raw_auth = headers.get(b"authorization")

            if raw_api_key:
                # Key machine clients by their API key hash digest
                key_str = raw_api_key.decode("latin1", errors="ignore").strip()
                hashed = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]
                key = f"apikey:{hashed}"
            elif raw_auth and raw_auth.lower().startswith(b"bearer "):
                # Key human callers by Bearer token hash
                token_str = raw_auth[7:].decode("latin1", errors="ignore").strip()
                hashed = hashlib.sha256(token_str.encode("utf-8")).hexdigest()[:16]
                key = f"jwt:{hashed}"
            else:
                # Unauthenticated or default API caller
                key = f"api:{client_ip}"

        result = self.limiter.check(key=key, limit=limit, window_seconds=60.0)

        if not result.allowed:
            logger.warning(
                "Rate limit exceeded: key='%s', path='%s', method='%s', limit=%d/min, retry_after=%ds",
                key,
                path,
                method,
                limit,
                result.retry_after,
            )

            # Asynchronously record security audit event
            await self._emit_security_audit_event(
                client_ip=client_ip,
                path=path,
                method=method,
                key=key,
                limit=limit,
            )

            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Please retry after {result.retry_after} seconds."
                },
                headers={
                    "Retry-After": str(result.retry_after),
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(result.reset_timestamp),
                },
            )
            await response(scope, receive, send)
            return

        # Request permitted: wrap send to attach rate-limit metadata headers
        async def send_with_rate_limit_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                resp_headers: List[Tuple[bytes, bytes]] = list(message.get("headers", []))
                resp_headers.append((b"x-ratelimit-limit", str(result.limit).encode("ascii")))
                resp_headers.append((b"x-ratelimit-remaining", str(result.remaining).encode("ascii")))
                resp_headers.append((b"x-ratelimit-reset", str(result.reset_timestamp).encode("ascii")))
                message["headers"] = resp_headers
            await send(message)

        await self.app(scope, receive, send_with_rate_limit_headers)


__all__ = [
    "InMemoryRateLimiter",
    "RateLimitMiddleware",
    "RateLimitResult",
    "get_client_ip",
    "global_rate_limiter",
]
