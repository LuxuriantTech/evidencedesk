import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol, cast

LOCAL_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_ID = "paraphrase-multilingual-minilm-l12-v2-onnx-q"
LOCAL_MODEL_REPOSITORY = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
LOCAL_MODEL_REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
LOCAL_MODEL_LICENSE = "Apache-2.0"


class ModelIntegrityError(RuntimeError):
    """Raised before inference when local model evidence cannot be verified."""


class EmbeddingBackend(Protocol):
    def embed(
        self, documents: list[str], *, batch_size: int
    ) -> Iterable[Sequence[float]]: ...


class EmbeddingProvider(Protocol):
    mode: str
    model_id: str
    dimension: int
    estimated_cost_usd: float

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


class DeterministicEmbeddingProvider:
    """Local feature-hash vectors for the no-key, reproducible baseline."""

    mode = "deterministic-hash-v1"
    estimated_cost_usd = 0.0

    def __init__(self, *, dimension: int = 384) -> None:
        self.dimension = dimension
        self.model_id = f"deterministic-hash-v1:{dimension}"

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

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(texts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fastembed_backend(
    *,
    model_path: Path,
    local_files_only: bool,
    providers: list[str],
) -> EmbeddingBackend:
    from fastembed import TextEmbedding

    return cast(
        EmbeddingBackend,
        TextEmbedding(
            model_name=LOCAL_MODEL_NAME,
            specific_model_path=str(model_path),
            local_files_only=local_files_only,
            providers=providers,
        ),
    )


class LocalSemanticEmbeddingProvider:
    """Verified, CPU-only ONNX embeddings with no network access at runtime."""

    mode = "local-semantic-onnx-v1"
    estimated_cost_usd = 0.0

    def __init__(
        self,
        *,
        manifest_path: Path,
        model_path: Path,
        backend_factory: Callable[..., EmbeddingBackend] | None = None,
    ) -> None:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_name = str(payload["model_id"])
            repository = str(payload["repository"])
            revision = str(payload["repository_revision"])
            license_name = str(payload["license"])
            dimension = int(payload["dimension"])
            files = payload["files"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelIntegrityError("invalid model manifest") from exc
        if (
            model_name != LOCAL_MODEL_ID
            or repository != LOCAL_MODEL_REPOSITORY
            or revision != LOCAL_MODEL_REVISION
            or license_name != LOCAL_MODEL_LICENSE
        ):
            raise ModelIntegrityError("unexpected model identity")
        if dimension != 384 or not isinstance(files, list) or not files:
            raise ModelIntegrityError("invalid model manifest")

        resolved_root = model_path.resolve()
        for item in files:
            if not isinstance(item, dict):
                raise ModelIntegrityError("invalid model manifest")
            relative = Path(str(item.get("path", "")))
            candidate = (resolved_root / relative).resolve()
            if not candidate.is_relative_to(resolved_root):
                raise ModelIntegrityError("model file escapes model directory")
            if not candidate.is_file():
                raise ModelIntegrityError(f"missing model file: {relative.as_posix()}")
            expected_bytes = item.get("bytes")
            if (
                not isinstance(expected_bytes, int)
                or expected_bytes < 1
                or candidate.stat().st_size != expected_bytes
            ):
                raise ModelIntegrityError(f"size mismatch: {relative.as_posix()}")
            expected = str(item.get("sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", expected) or _sha256(candidate) != expected:
                raise ModelIntegrityError(f"sha256 mismatch: {relative.as_posix()}")

        self.dimension = dimension
        self.model_id = f"{model_name}@{revision}"
        factory = backend_factory or _fastembed_backend
        self._backend = factory(
            model_path=model_path,
            local_files_only=True,
            providers=["CPUExecutionProvider"],
        )

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        documents = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in documents):
            raise ValueError("embedding input must contain non-empty strings")
        raw_vectors = list(self._backend.embed(documents, batch_size=32))
        if len(raw_vectors) != len(documents):
            raise ModelIntegrityError("embedding backend returned the wrong batch length")
        vectors: list[list[float]] = []
        for raw in raw_vectors:
            values = [float(value) for value in raw]
            if len(values) != self.dimension or not all(math.isfinite(value) for value in values):
                raise ModelIntegrityError("embedding backend returned an invalid vector")
            norm = math.sqrt(sum(value * value for value in values))
            if norm == 0.0:
                raise ModelIntegrityError("embedding backend returned a zero vector")
            vectors.append([value / norm for value in values])
        return vectors

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(texts)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]
