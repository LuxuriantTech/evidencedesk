from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from evidencedesk_api.answering import (
    DecisionConfig,
    DeterministicGroundedAnswerProvider,
    GroundedAnswer,
)
from evidencedesk_api.local_generation import build_local_generation_answer_provider
from evidencedesk_api.providers import (
    CURRENT_EMBEDDING_MANIFEST,
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    LocalSemanticEmbeddingProvider,
)
from evidencedesk_api.retrieval import (
    AnswerResult,
    ExtractiveAnswerProvider,
    RankedChunk,
)

AnswerValue = AnswerResult | GroundedAnswer


class AnswerProvider(Protocol):
    mode: str
    estimated_cost_usd: float

    def answer(self, question: str, ranked: list[RankedChunk]) -> AnswerValue: ...


@dataclass(frozen=True, slots=True)
class ProviderBundle:
    embedding: EmbeddingProvider
    answer: AnswerProvider

    @property
    def estimated_cost_usd(self) -> float:
        return self.embedding.estimated_cost_usd + self.answer.estimated_cost_usd


class UnsupportedProviderMode(ValueError):
    pass


def build_provider_bundle(
    mode: str,
    *,
    embedding_factory: Callable[[], EmbeddingProvider] | None = None,
    model_path: Path = Path("models/paraphrase-multilingual-minilm-l12-v2"),
    manifest_path: Path = CURRENT_EMBEDDING_MANIFEST,
    local_generation_base_url: str = "http://127.0.0.1:50193",
) -> ProviderBundle:
    if mode == "extractive-local-hash":
        embedding: EmbeddingProvider = DeterministicEmbeddingProvider(dimension=384)
    elif mode in {
        "extractive-local-onnx",
        "grounded-local-v3",
        "local-qwen8b-extractive-pilot",
    }:
        embedding = (
            embedding_factory()
            if embedding_factory is not None
            else LocalSemanticEmbeddingProvider(
                manifest_path=manifest_path,
                model_path=model_path,
            )
        )
    else:
        raise UnsupportedProviderMode(f"unsupported answer mode: {mode}")
    if mode == "grounded-local-v3":
        return ProviderBundle(
            embedding=embedding,
            answer=cast(
                AnswerProvider,
                DeterministicGroundedAnswerProvider(
                    embeddings=embedding,
                    config=DecisionConfig(
                        support_threshold=0.48,
                        partial_support_threshold=0.38,
                        contradiction_margin=0.08,
                    ),
                ),
            ),
        )
    if mode == "local-qwen8b-extractive-pilot":
        return ProviderBundle(
            embedding=embedding,
            answer=build_local_generation_answer_provider(mode, base_url=local_generation_base_url),
        )
    return ProviderBundle(embedding=embedding, answer=ExtractiveAnswerProvider(mode=mode))
