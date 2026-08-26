from dataclasses import dataclass
from typing import Protocol

from evidencedesk_api.providers import DeterministicEmbeddingProvider, EmbeddingProvider
from evidencedesk_api.retrieval import (
    AnswerResult,
    ExtractiveAnswerProvider,
    RankedChunk,
)


class AnswerProvider(Protocol):
    mode: str
    estimated_cost_usd: float

    def answer(self, question: str, ranked: list[RankedChunk]) -> AnswerResult: ...


@dataclass(frozen=True, slots=True)
class ProviderBundle:
    embedding: EmbeddingProvider
    answer: AnswerProvider

    @property
    def estimated_cost_usd(self) -> float:
        return self.embedding.estimated_cost_usd + self.answer.estimated_cost_usd


class UnsupportedProviderMode(ValueError):
    pass


def build_provider_bundle(mode: str) -> ProviderBundle:
    if mode != "extractive-local":
        raise UnsupportedProviderMode(f"unsupported answer mode: {mode}")
    return ProviderBundle(
        embedding=DeterministicEmbeddingProvider(dimension=384),
        answer=ExtractiveAnswerProvider(),
    )
