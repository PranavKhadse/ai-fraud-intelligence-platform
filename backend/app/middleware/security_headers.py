"""
Enterprise Security Response Headers Middleware.

Injects standard enterprise security headers across all HTTP responses,
protecting against MIME-sniffing, clickjacking, XSS exploitation, cross-origin leakage,
and insecure transport.
"""

from typing import List, Tuple
import logging
from starlette.types import ASGIApp, Scope, Receive, Send, Message

from backend.app.core.config import settings

logger = logging.getLogger("fraud_api.middleware.security_headers")


class SecurityHeadersMiddleware:
    """
    Pure ASGI middleware injecting defense-in-depth security headers into all responses.

    Headers Applied:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 0
    - Referrer-Policy: strict-origin-when-cross-origin (configurable)
    - Content-Security-Policy: default-src 'self' ... (configurable)
    - Permissions-Policy: geolocation=(), camera=(), microphone=(), payment=()
    - X-Permitted-Cross-Domain-Policies: none
    - Strict-Transport-Security: max-age=31536000; includeSubDomains (when HSTS enabled or in production)
    """

    def __init__(
        self,
        app: ASGIApp,
        csp: str = settings.SECURITY_CSP_HEADER,
        referrer_policy: str = settings.SECURITY_REFERRER_POLICY,
        hsts_enabled: bool = settings.SECURITY_HSTS_ENABLED,
    ) -> None:
        self.app = app
        self.csp = csp
        self.referrer_policy = referrer_policy
        self.hsts_enabled = hsts_enabled or (settings.ENVIRONMENT.lower() in ("production", "prod"))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not settings.SECURITY_HEADERS_ENABLED:
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                existing_headers: List[Tuple[bytes, bytes]] = list(message.get("headers", []))
                existing_names = {h[0].lower() for h in existing_headers}

                new_headers: List[Tuple[bytes, bytes]] = list(existing_headers)

                if b"x-content-type-options" not in existing_names:
                    new_headers.append((b"x-content-type-options", b"nosniff"))

                if b"x-frame-options" not in existing_names:
                    new_headers.append((b"x-frame-options", b"DENY"))

                if b"x-xss-protection" not in existing_names:
                    new_headers.append((b"x-xss-protection", b"0"))

                if b"referrer-policy" not in existing_names:
                    new_headers.append((b"referrer-policy", self.referrer_policy.encode("utf-8")))

                if b"content-security-policy" not in existing_names:
                    new_headers.append((b"content-security-policy", self.csp.encode("utf-8")))

                if b"permissions-policy" not in existing_names:
                    new_headers.append((
                        b"permissions-policy",
                        b"geolocation=(), camera=(), microphone=(), payment=()",
                    ))

                if b"x-permitted-cross-domain-policies" not in existing_names:
                    new_headers.append((b"x-permitted-cross-domain-policies", b"none"))

                hsts_active = (
                    self.hsts_enabled
                    or settings.SECURITY_HSTS_ENABLED
                    or (settings.ENVIRONMENT.lower() in ("production", "prod"))
                )
                if hsts_active and b"strict-transport-security" not in existing_names:
                    new_headers.append((
                        b"strict-transport-security",
                        b"max-age=31536000; includeSubDomains",
                    ))

                message["headers"] = new_headers

            await send(message)

        await self.app(scope, receive, send_with_security_headers)


__all__ = ["SecurityHeadersMiddleware"]
