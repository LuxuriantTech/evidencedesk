import pytest
from evidencedesk_api.provider_registry import UnsupportedProviderMode, build_provider_bundle


class FakeSemanticEmbeddingProvider:
    mode = "local-semantic-onnx-v1"
    dimension = 384
    model_id = "semantic-test@revision"
    estimated_cost_usd = 0.0

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def test_hash_bundle_is_explicit_and_free() -> None:
    providers = build_provider_bundle("extractive-local-hash")

    assert providers.embedding.mode == "deterministic-hash-v1"
    assert providers.answer.mode == "extractive-local-hash"
    assert providers.estimated_cost_usd == 0.0


def test_onnx_bundle_uses_an_injected_semantic_factory() -> None:
    embedding = FakeSemanticEmbeddingProvider()
    calls: list[bool] = []

    def factory() -> FakeSemanticEmbeddingProvider:
        calls.append(True)
        return embedding

    providers = build_provider_bundle("extractive-local-onnx", embedding_factory=factory)

    assert providers.embedding is embedding
    assert providers.answer.mode == "extractive-local-onnx"
    assert providers.estimated_cost_usd == 0.0
    assert calls == [True]


def test_grounded_v3_bundle_uses_semantic_embedding_and_frozen_decision_config() -> None:
    embedding = FakeSemanticEmbeddingProvider()

    providers = build_provider_bundle(
        "grounded-local-v3",
        embedding_factory=lambda: embedding,
    )

    assert providers.embedding is embedding
    assert providers.answer.mode == "deterministic-evidence-v3"
    assert providers.answer.config.support_threshold == 0.48
    assert providers.answer.config.partial_support_threshold == 0.38
    assert providers.answer.config.contradiction_margin == 0.08
    assert providers.estimated_cost_usd == 0.0


def test_unknown_provider_mode_fails_closed() -> None:
    with pytest.raises(UnsupportedProviderMode, match="unsupported answer mode"):
        build_provider_bundle("openai-compatible")
