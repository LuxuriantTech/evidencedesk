from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest
from evidencedesk_api.answering import GroundedAnswer
from evidencedesk_api.config import Settings
from evidencedesk_api.local_generation import (
    MODEL_SPECS,
    LocalGenerationAnswerProvider,
    LocalGenerationBusy,
    LocalGenerationUnavailable,
    validate_server_identity,
)
from evidencedesk_api.local_generation_core import (
    ModelOutputValidationError,
    _validate_generation_envelope,
)
from evidencedesk_api.provider_registry import build_provider_bundle
from pydantic import ValidationError


def _valid_abstention() -> GroundedAnswer:
    return GroundedAnswer(
        status="abstained",
        answerable=False,
        answer="Insufficient evidence in the retrieved sources.",
        confidence=0.0,
        supporting_document=None,
        supporting_page=None,
        supporting_excerpt=None,
        ambiguity_reason=None,
        extracted_fields={},
        supporting_chunk_ids=(),
        citations=(),
        candidate_assessments=(),
    )


def _accept_server_identity(base_url: str, expected_model: str) -> None:
    return None


class FakeEmbedding:
    mode = "fake-embedding"
    dimension = 384
    model_id = "fake-embedding@fixed"
    estimated_cost_usd = 0.0

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(texts)


class FakeAnswerProvider:
    def __init__(self, base_url: str, model_id: str, *, gate: Event | None = None) -> None:
        self.base_url = base_url
        self.model_id = model_id
        self.diagnostics: list[dict[str, str]] = []
        self.gate = gate

    def answer(self, question: str, ranked: list[Any]) -> GroundedAnswer:
        self.diagnostics.append({"question": question, "document": "private source"})
        if self.gate is not None:
            self.gate.wait(timeout=2)
        return _valid_abstention()


def test_default_and_public_demo_modes_do_not_switch_to_local_generation() -> None:
    settings = Settings(
        _env_file=None,
        jwt_secret="x" * 32,
        demo_admin_password="admin-secret",
        demo_analyst_password="analyst-secret",
        demo_reader_password="reader-secret",
    )
    assert settings.answer_mode == "grounded-local-v3"
    assert settings.public_demo_mode is True
    with pytest.raises(ValidationError, match="unavailable in public demo mode"):
        Settings(
            _env_file=None,
            jwt_secret="x" * 32,
            answer_mode="local-qwen8b-extractive-pilot",
            demo_admin_password="admin-secret",
            demo_analyst_password="analyst-secret",
            demo_reader_password="reader-secret",
        )
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            jwt_secret="x" * 32,
            answer_mode="remote-provider",
            demo_admin_password="admin-secret",
            demo_analyst_password="analyst-secret",
            demo_reader_password="reader-secret",
        )


@pytest.mark.parametrize(
    ("mode", "expected_model"),
    [(mode, model["id"]) for mode, model in MODEL_SPECS.items()],
)
def test_registry_preserves_embedding_and_restricts_local_model_identity(
    mode: str, expected_model: str
) -> None:
    embedding = FakeEmbedding()
    made: list[FakeAnswerProvider] = []

    def factory(base_url: str, model_id: str) -> FakeAnswerProvider:
        provider = FakeAnswerProvider(base_url, model_id)
        made.append(provider)
        return provider

    from evidencedesk_api.local_generation import build_local_generation_answer_provider

    bundle = build_provider_bundle(
        mode,
        embedding_factory=lambda: embedding,
        local_generation_base_url="http://127.0.0.1:50193",
    )
    assert bundle.embedding is embedding
    assert bundle.answer.mode == mode
    assert bundle.estimated_cost_usd == 0.0
    answer = build_local_generation_answer_provider(
        mode,
        base_url="http://127.0.0.1:50193",
    )
    answer._provider_factory = factory
    answer._identity_checker = _accept_server_identity
    assert answer.answer("question one", []).status == "abstained"
    assert answer.answer("question two", []).status == "abstained"
    assert [provider.model_id for provider in made] == [expected_model, expected_model]
    assert made[0] is not made[1]
    assert made[0].diagnostics == made[1].diagnostics == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.com:50193",
        "http://192.168.1.8:50193",
        "https://127.0.0.1:50193",
        "http://user:secret@127.0.0.1:50193",
        "http://127.0.0.1:50193/?target=example.com",
    ],
)
def test_remote_or_ambiguous_endpoint_is_rejected_at_construction(base_url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        LocalGenerationAnswerProvider(
            "local-qwen8b-extractive-pilot",
            base_url=base_url,
            provider_factory=FakeAnswerProvider,
        )


def test_arbitrary_model_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported local model mode"):
        LocalGenerationAnswerProvider(
            "Qwen-unknown-model",
            base_url="http://127.0.0.1:50193",
            provider_factory=FakeAnswerProvider,
        )


def test_backend_error_does_not_leak_exception_or_diagnostic_text() -> None:
    def broken_provider(base_url: str, model_id: str) -> Any:
        class Broken:
            def __init__(self) -> None:
                self.diagnostics: list[dict[str, str]] = []

            def answer(self, question: str, ranked: list[Any]) -> None:
                self.diagnostics.append({"prompt": question, "raw": "private raw output"})
                raise RuntimeError("private source: " + question + " private raw output")

        return Broken()

    provider = LocalGenerationAnswerProvider(
        "local-qwen8b-extractive-pilot",
        base_url="http://127.0.0.1:50193",
        provider_factory=broken_provider,
        identity_checker=_accept_server_identity,
    )
    with pytest.raises(LocalGenerationUnavailable) as caught:
        provider.answer("secret question", [])
    assert str(caught.value) == "local_model_unavailable"
    assert "secret" not in str(caught.value)
    assert "private raw" not in str(caught.value)


def test_fail_closed_invalid_model_output_is_reported_as_unavailable() -> None:
    def invalid_provider(base_url: str, model_id: str) -> Any:
        class Invalid:
            def __init__(self) -> None:
                self.diagnostics = [
                    {
                        "decision": "fail_closed_abstention",
                        "model_validation_error": "private raw schema detail",
                    }
                ]

            def answer(self, question: str, ranked: list[Any]) -> object:
                return object()

        return Invalid()

    provider = LocalGenerationAnswerProvider(
        "local-qwen8b-extractive-pilot",
        base_url="http://127.0.0.1:50193",
        provider_factory=invalid_provider,
        identity_checker=_accept_server_identity,
    )
    with pytest.raises(LocalGenerationUnavailable) as caught:
        provider.answer("private question", [])
    assert str(caught.value) == "local_model_unavailable"
    assert "schema" not in str(caught.value)


def test_only_one_concurrent_local_inference_is_admitted() -> None:
    release = Event()
    made: list[FakeAnswerProvider] = []

    def factory(base_url: str, model_id: str) -> FakeAnswerProvider:
        provider = FakeAnswerProvider(base_url, model_id, gate=release)
        made.append(provider)
        return provider

    provider = LocalGenerationAnswerProvider(
        "local-qwen8b-extractive-pilot",
        base_url="http://127.0.0.1:50193",
        provider_factory=factory,
        identity_checker=_accept_server_identity,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(provider.answer, "first private query", [])
        # Wait until the first provider has entered its blocking answer call.
        for _ in range(100):
            if made and made[0].diagnostics:
                break
            Event().wait(0.01)
        with pytest.raises(LocalGenerationBusy):
            provider.answer("second private query", [])
        release.set()
        first.result(timeout=2)
    assert len(made) == 1
    assert made[0].diagnostics == []


@pytest.mark.parametrize("failure", ["wrong_model", "wrong_context", "wrong_alias", "wrong_slots"])
def test_server_identity_requires_pinned_model_context_and_single_slot(failure: str) -> None:
    expected = "Qwen3-8B-Q4_K_M.gguf"
    model = "Qwen3-4B-Q4_K_M.gguf" if failure == "wrong_model" else expected
    model_context = 8192 if failure == "wrong_context" else 6144
    alias = "wrong-model" if failure == "wrong_alias" else expected
    slots = 2 if failure == "wrong_slots" else 1
    health = (200, {"status": "ok"})
    models = (200, {"data": [{"id": model, "meta": {"n_ctx": model_context}}]})
    props = (
        200,
        {
            "model_alias": alias,
            "total_slots": slots,
            "default_generation_settings": {"n_ctx": model_context},
        },
    )
    with pytest.raises(ValueError):
        validate_server_identity(health, models, props, expected_model=expected)


def test_server_identity_failure_is_sanitized_before_http_surface() -> None:
    def failed_check(_base: str, _model: str) -> None:
        raise RuntimeError("private /props response with document text")

    provider = LocalGenerationAnswerProvider(
        "local-qwen8b-extractive-pilot",
        base_url="http://127.0.0.1:50193",
        provider_factory=FakeAnswerProvider,
        identity_checker=failed_check,
    )
    with pytest.raises(LocalGenerationUnavailable) as caught:
        provider.answer("private query", [])
    assert str(caught.value) == "local_model_unavailable"
    assert "document text" not in str(caught.value)


def test_completion_envelope_must_report_the_configured_model() -> None:
    valid = {
        "model": "Qwen3-8B-Q4_K_M.gguf",
        "choices": [{"message": {"reasoning_content": ""}, "finish_reason": "stop"}],
    }
    _validate_generation_envelope(valid, "{}", "Qwen3-8B-Q4_K_M.gguf")
    with pytest.raises(ModelOutputValidationError, match="identity_mismatch"):
        _validate_generation_envelope(
            {**valid, "model": "Qwen3-4B-Q4_K_M.gguf"},
            "{}",
            "Qwen3-8B-Q4_K_M.gguf",
        )


@pytest.mark.parametrize(
    ("finish_reason", "reasoning_content", "content"),
    [
        ("length", "", '{"status":"abstained"}'),
        ("stop", "private reasoning", '{"status":"abstained"}'),
        ("stop", "", '<think>hidden</think>{"status":"abstained"}'),
    ],
)
def test_completion_rejects_partial_or_thinking_output(
    finish_reason: str,
    reasoning_content: str,
    content: str,
) -> None:
    payload = {
        "model": "Qwen3-8B-Q4_K_M.gguf",
        "choices": [
            {
                "message": {"content": content, "reasoning_content": reasoning_content},
                "finish_reason": finish_reason,
            }
        ],
    }
    with pytest.raises(ModelOutputValidationError):
        _validate_generation_envelope(payload, content, "Qwen3-8B-Q4_K_M.gguf")
