from __future__ import annotations

import asyncio
import math
import re
from collections import deque
from collections.abc import Callable, Mapping
from time import monotonic

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyTooLarge(Exception):
    pass


async def _send_too_large(send: Send) -> None:
    body = b'{"detail":{"code":"request_too_large"}}'
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared_length: int | None = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    parsed = int(value)
                except ValueError:
                    break
                if parsed >= 0:
                    declared_length = parsed
                break
        if declared_length is not None and declared_length > self.max_bytes:
            await _send_too_large(send)
            return

        consumed = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await _send_too_large(send)


_SENSITIVE_ROUTES = (
    ("auth", "POST", re.compile(r"^/api/v1/auth/token$")),
    ("ask", "POST", re.compile(r"^/api/v1/dossiers/[^/]+/ask$")),
    ("upload", "POST", re.compile(r"^/api/v1/dossiers/[^/]+/documents$")),
)


async def _send_rate_limited(send: Send, *, retry_after_seconds: int) -> None:
    body = b'{"detail":{"code":"rate_limited"}}'
    await send(
        {
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"retry-after", str(retry_after_seconds).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RateLimitMiddleware:
    """Bound sensitive requests per client for the single-process demo API.

    This fixed-window-free sliding log is deliberately local and bounded. A future
    multi-replica deployment must replace it with a shared limiter at the gateway.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limits: Mapping[str, int],
        window_seconds: float = 60.0,
        max_keys: int = 10_000,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("rate-limit window must be positive")
        if max_keys <= 0:
            raise ValueError("rate-limit key capacity must be positive")
        if any(value <= 0 for value in limits.values()):
            raise ValueError("rate limits must be positive")
        self.app = app
        self.limits = dict(limits)
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self.clock = clock
        self._events: dict[tuple[str, str], deque[float]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _bucket(scope: Scope) -> str | None:
        method = str(scope.get("method", "")).upper()
        path = str(scope.get("path", ""))
        for bucket, expected_method, pattern in _SENSITIVE_ROUTES:
            if method == expected_method and pattern.fullmatch(path):
                return bucket
        return None

    @staticmethod
    def _client(scope: Scope) -> str:
        client = scope.get("client")
        if client is None:
            return "unknown"
        return str(client[0])

    async def _admit(self, *, bucket: str, client: str) -> tuple[bool, int]:
        now = self.clock()
        cutoff = now - self.window_seconds
        key = (bucket, client)
        async with self._lock:
            events = self._events.get(key)
            if events is None:
                if len(self._events) >= self.max_keys:
                    oldest_key = min(
                        self._events,
                        key=lambda candidate: self._events[candidate][-1],
                    )
                    del self._events[oldest_key]
                events = deque()
                self._events[key] = events
            while events and events[0] <= cutoff:
                events.popleft()
            limit = self.limits[bucket]
            if len(events) >= limit:
                retry_after = max(1, math.ceil(events[0] + self.window_seconds - now))
                return False, retry_after
            events.append(now)
            return True, 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        bucket = self._bucket(scope)
        if bucket is None or bucket not in self.limits:
            await self.app(scope, receive, send)
            return
        admitted, retry_after = await self._admit(bucket=bucket, client=self._client(scope))
        if not admitted:
            await _send_rate_limited(send, retry_after_seconds=retry_after)
            return
        await self.app(scope, receive, send)
