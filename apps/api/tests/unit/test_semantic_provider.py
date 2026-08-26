from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
from evidencedesk_api import providers


class FakeTextEmbeddingBackend:
    def __init__(self, **options: object) -> None:
        self.options = options
        self.requests: list[tuple[list[str], int]] = []

    def embed(self, documents: list[str], *, batch_size: int) -> list[list[float]]:
        self.requests.append((documents, batch_size))
        return [[float(index + 1) for index in range(384)] for _ in documents]


def _write_model_manifest(
    tmp_path: Path,
    *,
    file_sha256: str | None = None,
    model_id: str = "paraphrase-multilingual-minilm-l12-v2-onnx-q",
    file_bytes: int | None = None,
) -> Path:
    model_file = tmp_path / "model_optimized.onnx"
    model_file.write_bytes(b"synthetic-onnx-model")
    digest = file_sha256 or hashlib.sha256(model_file.read_bytes()).hexdigest()
    manifest = {
        "model_id": model_id,
        "repository": "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        "repository_revision": "faf4aa4225822f3bc6376869cb1164e8e3feedd0",
        "license": "Apache-2.0",
        "dimension": 384,
        "files": [
            {
                "path": model_file.name,
                "bytes": file_bytes if file_bytes is not None else model_file.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_local_semantic_provider_encodes_a_normalized_batch_offline_on_cpu(tmp_path: Path) -> None:
    manifest_path = _write_model_manifest(tmp_path)
    created: list[FakeTextEmbeddingBackend] = []

    def backend_factory(**options: object) -> FakeTextEmbeddingBackend:
        backend = FakeTextEmbeddingBackend(**options)
        created.append(backend)
        return backend

    provider = providers.LocalSemanticEmbeddingProvider(
        manifest_path=manifest_path,
        model_path=tmp_path,
        backend_factory=backend_factory,
    )

    vectors = provider.embed_many(["Renewal date", "Incident notification"])

    assert provider.mode == "local-semantic-onnx-v1"
    assert provider.dimension == 384
    assert provider.estimated_cost_usd == 0.0
    assert provider.model_id == (
        "paraphrase-multilingual-minilm-l12-v2-onnx-q@"
        "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
    )
    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert all(
        math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)
        for vector in vectors
    )
    assert created[0].options == {
        "model_path": tmp_path,
        "local_files_only": True,
        "providers": ["CPUExecutionProvider"],
    }
    assert created[0].requests == [(["Renewal date", "Incident notification"], 32)]
    assert provider.embed("Renewal date") == vectors[0]


def test_local_semantic_provider_model_id_is_immutable_after_manifest_load(tmp_path: Path) -> None:
    manifest_path = _write_model_manifest(tmp_path)
    provider = providers.LocalSemanticEmbeddingProvider(
        manifest_path=manifest_path,
        model_path=tmp_path,
        backend_factory=lambda **_: FakeTextEmbeddingBackend(),
    )

    model_id = provider.model_id
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["repository_revision"] = "different-revision"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert provider.model_id == model_id


def test_local_semantic_provider_fails_closed_when_a_manifest_file_is_missing(
    tmp_path: Path,
) -> None:
    manifest_path = _write_model_manifest(tmp_path)
    (tmp_path / "model_optimized.onnx").unlink()

    with pytest.raises(providers.ModelIntegrityError, match="missing model file"):
        providers.LocalSemanticEmbeddingProvider(
            manifest_path=manifest_path,
            model_path=tmp_path,
            backend_factory=lambda **_: FakeTextEmbeddingBackend(),
        )


def test_local_semantic_provider_fails_closed_when_a_manifest_hash_is_invalid(
    tmp_path: Path,
) -> None:
    manifest_path = _write_model_manifest(tmp_path, file_sha256="0" * 64)

    with pytest.raises(providers.ModelIntegrityError, match="sha256 mismatch"):
        providers.LocalSemanticEmbeddingProvider(
            manifest_path=manifest_path,
            model_path=tmp_path,
            backend_factory=lambda **_: FakeTextEmbeddingBackend(),
        )


def test_local_semantic_provider_rejects_a_different_model_identity(tmp_path: Path) -> None:
    manifest_path = _write_model_manifest(tmp_path, model_id="lookalike-model")

    with pytest.raises(providers.ModelIntegrityError, match="unexpected model identity"):
        providers.LocalSemanticEmbeddingProvider(
            manifest_path=manifest_path,
            model_path=tmp_path,
            backend_factory=lambda **_: FakeTextEmbeddingBackend(),
        )


def test_local_semantic_provider_rejects_a_file_size_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_model_manifest(tmp_path, file_bytes=1)

    with pytest.raises(providers.ModelIntegrityError, match="size mismatch"):
        providers.LocalSemanticEmbeddingProvider(
            manifest_path=manifest_path,
            model_path=tmp_path,
            backend_factory=lambda **_: FakeTextEmbeddingBackend(),
        )


def test_deterministic_provider_has_an_explicit_stable_model_id_and_batch_api() -> None:
    provider = providers.DeterministicEmbeddingProvider(dimension=384)

    assert provider.model_id.startswith("deterministic-hash-v1")
    assert provider.embed_many(["same text", "same text"]) == [
        provider.embed("same text"),
        provider.embed("same text"),
    ]
