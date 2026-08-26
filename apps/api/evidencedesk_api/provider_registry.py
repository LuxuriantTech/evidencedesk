from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evidencedesk_api.providers import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    LocalSemanticEmbeddingProvider,
)
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


def build_provider_bundle(
    mode: str,
    *,
    embedding_factory: Callable[[], EmbeddingProvider] | None = None,
    model_path: Path = Path("models/paraphrase-multilingual-minilm-l12-v2"),
    manifest_path: Path = Path("infra/models/paraphrase-multilingual-minilm-l12-v2.json"),
) -> ProviderBundle:
    if mode == "extractive-local-hash":
        embedding: EmbeddingProvider = DeterministicEmbeddingProvider(dimension=384)
    elif mode == "extractive-local-onnx":
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
    return ProviderBundle(
        embedding=embedding,
        answer=ExtractiveAnswerProvider(mode=mode),
    )
