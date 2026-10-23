"""Opt-in, local-only Qwen adapter. This remains an unqualified pilot."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from fastapi.concurrency import run_in_threadpool

from evidencedesk_api.answering import GroundedAnswer
from evidencedesk_api.local_generation_core import GenerativeAnswerProvider
from evidencedesk_api.retrieval import RankedChunk

MODEL_SPECS = {
    "local-qwen8b-extractive-pilot": {
        "id": "Qwen3-8B-Q4_K_M.gguf",
        "repo": "Qwen/Qwen3-8B-GGUF",
        "revision": "7c41481f57cb95916b40956ab2f0b139b296d974",
        "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
    },
}
EXPECTED_CONTEXT_TOKENS = 6144
EXPECTED_SERVER_SLOTS = 1


def _server_json(base_url: str, suffix: str) -> tuple[int, dict[str, Any]]:
    parsed = urllib.parse.urlparse(base_url)
    request = urllib.request.Request(
        f"http://{parsed.netloc}{suffix}",
        headers={"Accept": "application/json"},
    )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(
            self,
            req: urllib.request.Request,
            fp: Any,
            code: int,
            msg: str,
            headers: Any,
            newurl: str,
        ) -> None:
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=5.0) as response:
        raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("server identity response too large")
        body = json.loads(raw.decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("invalid server identity response")
        return response.status, body


def validate_server_identity(
    health: tuple[int, dict[str, Any]],
    models: tuple[int, dict[str, Any]],
    props: tuple[int, dict[str, Any]],
    *,
    expected_model: str,
) -> None:
    health_status, health_body = health
    models_status, models_body = models
    props_status, props_body = props
    if health_status != 200 or health_body.get("status") != "ok":
        raise ValueError("server health contract mismatch")
    if models_status != 200 or props_status != 200:
        raise ValueError("server identity endpoint unavailable")
    entries = models_body.get("data")
    if not isinstance(entries, list):
        raise ValueError("server model list invalid")
    matching = [
        item
        for item in entries
        if isinstance(item, dict)
        and (
            item.get("id") == expected_model
            or (
                isinstance(item.get("aliases"), list)
                and expected_model in item["aliases"]
            )
        )
    ]
    if not matching or matching[0].get("meta", {}).get("n_ctx") != EXPECTED_CONTEXT_TOKENS:
        raise ValueError("server model identity or context mismatch")
    generation = props_body.get("default_generation_settings")
    if (
        props_body.get("model_alias") != expected_model
        or props_body.get("total_slots") != EXPECTED_SERVER_SLOTS
        or not isinstance(generation, dict)
        or generation.get("n_ctx") != EXPECTED_CONTEXT_TOKENS
    ):
        raise ValueError("server runtime context or slot mismatch")


def _verify_server(base_url: str, expected_model: str) -> None:
    records = tuple(
        _server_json(base_url, suffix)
        for suffix in ("/health", "/v1/models", "/props")
    )
    validate_server_identity(*records, expected_model=expected_model)


class LocalGenerationBusy(RuntimeError):
    """The single local inference slot is occupied."""


class LocalGenerationUnavailable(RuntimeError):
    """The local inference backend failed; details are intentionally private."""


class LocalGenerationAnswerProvider:
    estimated_cost_usd = 0.0

    def __init__(
        self,
        mode: str,
        *,
        base_url: str,
        provider_factory: Callable[..., Any] = GenerativeAnswerProvider,
        identity_checker: Callable[[str, str], None] = _verify_server,
    ) -> None:
        if mode not in MODEL_SPECS:
            raise ValueError("unsupported local model mode")
        parsed = urllib.parse.urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.path.rstrip("/") not in {"", "/v1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("local generation endpoint must use the configured loopback host")
        self.mode = mode
        self._base_url = base_url
        self.model_spec = dict(MODEL_SPECS[mode])
        self._model_id = self.model_spec["id"]
        self._provider_factory = provider_factory
        self._identity_checker = identity_checker
        self._inference_slot = threading.Lock()

    def answer(self, question: str, ranked: list[RankedChunk]) -> GroundedAnswer:
        self.reserve()
        try:
            return self.answer_reserved(question, ranked)
        finally:
            self.release_reserved()

    def reserve(self) -> None:
        if not self._inference_slot.acquire(blocking=False):
            raise LocalGenerationBusy("local_model_busy") from None

    def release_reserved(self) -> None:
        self._inference_slot.release()

    def answer_reserved(self, question: str, ranked: list[RankedChunk]) -> GroundedAnswer:
        provider: Any | None = None
        try:
            self._identity_checker(self._base_url, self._model_id)
            provider = self._provider_factory(self._base_url, self._model_id)
            provider.mode = self.mode
            result = provider.answer(question, ranked)
            diagnostics = getattr(provider, "diagnostics", None)
            if isinstance(diagnostics, list) and any(
                isinstance(item, dict)
                and (
                    "model_validation_error" in item
                    or item.get("decision") == "fail_closed_abstention"
                )
                for item in diagnostics
            ):
                raise LocalGenerationUnavailable("local_model_output_invalid") from None
            if not isinstance(result, GroundedAnswer):
                raise LocalGenerationUnavailable("local_model_contract_invalid") from None
            return result
        except LocalGenerationBusy:
            raise
        except Exception:
            # Never include backend exception text, request data or diagnostics.
            raise LocalGenerationUnavailable("local_model_unavailable") from None
        finally:
            if provider is not None:
                diagnostics = getattr(provider, "diagnostics", None)
                if isinstance(diagnostics, list):
                    diagnostics.clear()


def build_local_generation_answer_provider(
    mode: str,
    *,
    base_url: str = "http://127.0.0.1:50193",
) -> LocalGenerationAnswerProvider:
    return LocalGenerationAnswerProvider(mode, base_url=base_url)


async def answer_local_in_threadpool(
    provider: LocalGenerationAnswerProvider,
    question: str,
    ranked: list[RankedChunk],
) -> GroundedAnswer:
    provider.reserve()
    try:
        return await run_in_threadpool(provider.answer_reserved, question, ranked)
    finally:
        provider.release_reserved()
