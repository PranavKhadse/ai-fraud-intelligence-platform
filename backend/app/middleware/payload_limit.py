"""
Request Payload Size Limiting Middleware.

Rejects request bodies exceeding the configured MAX_REQUEST_BODY_BYTES limit with HTTP 413.
Guarantees early termination on oversized requests without consuming memory or CPU.
"""

from typing import Optional
import logging
from starlette.types import ASGIApp, Scope, Receive, Send, Message
from starlette.responses import JSONResponse
from starlette import status

from backend.app.core.config import settings

logger = logging.getLogger("fraud_api.middleware.payload_limit")


class PayloadTooLargeError(Exception):
    """Internal exception indicating the streaming request payload exceeded max allowed size."""
    pass


class PayloadLimitMiddleware:
    """
    Pure ASGI middleware enforcing maximum request body size limits.

    Checks:
    1. Fast-path: Content-Length header against max_body_bytes.
    2. Streaming-path: Accumulates chunk lengths for chunked or missing Content-Length headers,
       raising PayloadTooLargeError before unbounded memory is consumed.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_body_bytes: Optional[int] = None,
    ) -> None:
        self.app = app
        self._max_body_bytes = max_body_bytes

    @property
    def max_body_bytes(self) -> int:
        if self._max_body_bytes is not None:
            return self._max_body_bytes
        return settings.MAX_REQUEST_BODY_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_limit = self.max_body_bytes

        # 1. Inspect Content-Length header if present
        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                content_length = int(raw_length)
                if content_length > max_limit:
                    logger.warning(
                        "Request Content-Length (%d bytes) exceeded maximum permitted size (%d bytes) on %s",
                        content_length,
                        max_limit,
                        scope.get("path"),
                    )
                    response = JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Request payload too large. Maximum permitted size is {max_limit} bytes."
                        },
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        # 2. Inspect streaming / chunked bodies dynamically
        bytes_received = 0

        async def limited_receive() -> Message:
            nonlocal bytes_received
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                bytes_received += len(chunk)
                if bytes_received > max_limit:
                    logger.warning(
                        "Streaming request body exceeded maximum permitted size (%d bytes) on %s",
                        max_limit,
                        scope.get("path"),
                    )
                    raise PayloadTooLargeError()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except PayloadTooLargeError:
            response = JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "detail": f"Request payload too large. Maximum permitted size is {max_limit} bytes."
                },
            )
            await response(scope, receive, send)


__all__ = ["PayloadLimitMiddleware", "PayloadTooLargeError"]
