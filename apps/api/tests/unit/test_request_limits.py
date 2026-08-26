import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from evidencedesk_api.request_limits import RequestBodyLimitMiddleware

Message = dict[str, Any]


async def _drain_app(
    _scope: Message,
    receive: Callable[[], Awaitable[Message]],
    send: Callable[[Message], Awaitable[None]],
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
        scope: Message,
        receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
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
