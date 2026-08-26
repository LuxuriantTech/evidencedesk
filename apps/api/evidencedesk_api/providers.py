import hashlib
import math
import re
from typing import Protocol


class EmbeddingProvider(Protocol):
    mode: str
    dimension: int
    estimated_cost_usd: float

    def embed(self, text: str) -> list[float]: ...


class DeterministicEmbeddingProvider:
    """Local feature-hash vectors for the no-key, reproducible baseline."""

    mode = "deterministic-hash-v1"
    estimated_cost_usd = 0.0

    def __init__(self, *, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        normalized = re.sub(r"\s+", " ", text.casefold()).strip()
        features = re.findall(r"[a-z0-9]+", normalized)
        for token in tuple(features):
            padded = f"  {token}  "
            features.extend(padded[index : index + 3] for index in range(len(padded) - 2))

        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            vector[index] += -1.0 if value & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
