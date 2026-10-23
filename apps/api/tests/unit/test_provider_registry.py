from collections.abc import Sequence
from typing import cast

import pytest
from evidencedesk_api.answering import DeterministicGroundedAnswerProvider, GroundedAnswer
from evidencedesk_api.provider_registry import UnsupportedProviderMode, build_provider_bundle
from evidencedesk_api.providers import EmbeddingProvider
from evidencedesk_api.retrieval import EvidenceChunk, hybrid_rank


class FakeSemanticEmbeddingProvider:
    mode = "local-semantic-onnx-v1"
    dimension = 384
    model_id = "semantic-test@revision"
    estimated_cost_usd = 0.0

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(texts)


def test_hash_bundle_is_explicit_and_free() -> None:
    providers = build_provider_bundle("extractive-local-hash")

    assert providers.embedding.mode == "deterministic-hash-v1"
    assert providers.answer.mode == "extractive-local-hash"
    assert providers.estimated_cost_usd == 0.0


def test_onnx_bundle_uses_an_injected_semantic_factory() -> None:
    embedding = FakeSemanticEmbeddingProvider()
    calls: list[bool] = []

    def factory() -> EmbeddingProvider:
        calls.append(True)
        return embedding

    providers = build_provider_bundle("extractive-local-onnx", embedding_factory=factory)

    assert providers.embedding is cast(EmbeddingProvider, embedding)
    assert providers.answer.mode == "extractive-local-onnx"
    assert providers.estimated_cost_usd == 0.0
    assert calls == [True]


def test_grounded_v3_bundle_uses_semantic_embedding_and_frozen_decision_config() -> None:
    embedding = FakeSemanticEmbeddingProvider()

    providers = build_provider_bundle(
        "grounded-local-v3",
        embedding_factory=lambda: cast(EmbeddingProvider, embedding),
    )

    assert providers.embedding is cast(EmbeddingProvider, embedding)
    assert providers.answer.mode == "deterministic-evidence-v3"
    answer = cast(DeterministicGroundedAnswerProvider, providers.answer)
    assert answer.config.support_threshold == 0.48
    assert answer.config.partial_support_threshold == 0.38
    assert answer.config.contradiction_margin == 0.08
    assert providers.estimated_cost_usd == 0.0


def test_grounded_v3_provider_answers_with_scoped_evidence_after_hybrid_retrieval() -> None:
    embedding = FakeSemanticEmbeddingProvider()
    providers = build_provider_bundle(
        "grounded-local-v3",
        embedding_factory=lambda: cast(EmbeddingProvider, embedding),
    )
    question = "What is the effective date in the Northstar service agreement?"
    chunks = [
        EvidenceChunk(
            id="contract-effective",
            document_id="service-contract",
            document_name="northstar_service_agreement.txt",
            page=1,
            section=None,
            text="Effective date: 27 August 2031.",
            embedding=[0.0] * 384,
            embedding_model_id=embedding.model_id,
        ),
        EvidenceChunk(
            id="addendum-effective",
            document_id="service-addendum",
            document_name="northstar_data_addendum.txt",
            page=1,
            section=None,
            text="Effective date: 28 August 2031.",
            embedding=[0.0] * 384,
            embedding_model_id=embedding.model_id,
        ),
    ]
    ranked = hybrid_rank(question, chunks, provider=providers.embedding)

    result = providers.answer.answer(question, ranked)

    assert isinstance(result, GroundedAnswer)
    assert result.status == "answered"
    assert result.answer == "27 August 2031"
    assert result.supporting_document == "service-contract"
    assert [citation.chunk_id for citation in result.citations] == ["contract-effective"]


def test_unknown_provider_mode_fails_closed() -> None:
    with pytest.raises(UnsupportedProviderMode, match="unsupported answer mode"):
        build_provider_bundle("openai-compatible")
