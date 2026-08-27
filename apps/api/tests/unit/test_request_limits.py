import asyncio
import json

from evidencedesk_api.request_limits import RateLimitMiddleware, RequestBodyLimitMiddleware
from starlette.types import Message, Receive, Scope, Send


async def _drain_app(
    _scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    while True:
        message = await receive()
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def test_declared_oversize_request_is_rejected_before_downstream() -> None:
    called = False
    sent: list[Message] = []

    async def app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        nonlocal called
        called = True
        await _drain_app(scope, receive, send)

    async def scenario() -> None:
        middleware = RequestBodyLimitMiddleware(app, max_bytes=10)

        async def receive() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: Message) -> None:
            sent.append(message)

        await middleware(
            {
                "type": "http",
                "method": "POST",
                "headers": [(b"content-length", b"11")],
            },
            receive,
            send,
        )

    asyncio.run(scenario())
    assert called is False
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"]["code"] == "request_too_large"


def test_streamed_request_without_length_is_cut_off_at_the_same_limit() -> None:
    chunks = iter(
        [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"789012", "more_body": False},
        ]
    )
    sent: list[Message] = []

    async def scenario() -> None:
        middleware = RequestBodyLimitMiddleware(_drain_app, max_bytes=10)

        async def receive() -> Message:
            return next(chunks)

        async def send(message: Message) -> None:
            sent.append(message)

        await middleware(
            {"type": "http", "method": "POST", "headers": []},
            receive,
            send,
        )

    asyncio.run(scenario())
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"]["code"] == "request_too_large"


def test_sensitive_route_is_rate_limited_per_client_and_bucket() -> None:
    sent: list[Message] = []
    now = 100.0

    async def scenario() -> None:
        middleware = RateLimitMiddleware(
            _drain_app,
            limits={"auth": 2, "ask": 3, "upload": 1},
            window_seconds=60.0,
            clock=lambda: now,
        )

        async def receive() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: Message) -> None:
            sent.append(message)

        scope: Scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/auth/token",
            "headers": [],
            "client": ("192.0.2.10", 45123),
        }
        await middleware(scope, receive, send)
        await middleware(scope, receive, send)
        await middleware(scope, receive, send)

    asyncio.run(scenario())
    statuses = [message["status"] for message in sent if message["type"] == "http.response.start"]
    assert statuses == [204, 204, 429]
    assert json.loads(sent[-1]["body"])["detail"]["code"] == "rate_limited"


def test_rate_limit_expires_and_does_not_limit_health_checks() -> None:
    sent: list[Message] = []
    current = [100.0]

    async def scenario() -> None:
        middleware = RateLimitMiddleware(
            _drain_app,
            limits={"auth": 1, "ask": 1, "upload": 1},
            window_seconds=60.0,
            clock=lambda: current[0],
        )

        async def receive() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: Message) -> None:
            sent.append(message)

        base: Scope = {
            "type": "http",
            "method": "POST",
            "headers": [],
            "client": ("192.0.2.20", 45123),
        }
        await middleware({**base, "path": "/api/v1/dossiers/one/ask"}, receive, send)
        await middleware({**base, "path": "/health", "method": "GET"}, receive, send)
        current[0] = 161.0
        await middleware({**base, "path": "/api/v1/dossiers/one/ask"}, receive, send)

    asyncio.run(scenario())
    statuses = [message["status"] for message in sent if message["type"] == "http.response.start"]
    assert statuses == [204, 204, 204]
