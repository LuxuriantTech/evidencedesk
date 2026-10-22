from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from evidencedesk_api.text_onnx import LocalTextOnnxBackend
from numpy.typing import NDArray


@dataclass
class Encoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class Tokenizer:
    def encode_batch(self, inputs: list[str]) -> list[Encoding]:
        # Deliberately padded rows let the test distinguish masked mean pooling.
        return [Encoding([1, len(value), 0], [1, 1, 0], [0, 0, 0]) for value in inputs]


class Input:
    def __init__(self, name: str, shape: list[int | None]) -> None:
        self.name = name
        self.shape = shape


class Session:
    def __init__(self) -> None:
        self.requests: list[dict[str, NDArray[np.int64]]] = []

    def get_inputs(self) -> list[Input]:
        shape: list[int | None] = [None, None]
        return [
            Input("input_ids", shape),
            Input("attention_mask", shape),
            Input("token_type_ids", shape),
        ]

    def run(
        self,
        _outputs: object,
        inputs: dict[str, NDArray[np.int64]],
    ) -> list[NDArray[np.float32]]:
        self.requests.append(inputs)
        batch, width = inputs["input_ids"].shape
        # Token vectors vary by position; the padded final token must be excluded.
        values = np.zeros((batch, width, 384), dtype=np.float32)
        values[:, 0, 0] = 3
        values[:, 1, 1] = 4
        values[:, 2, 2] = 1000
        return [values]


def _backend(
    tmp_path: Path,
    *,
    session: Session | None = None,
    **kwargs: Any,
) -> tuple[LocalTextOnnxBackend, Session]:
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model_optimized.onnx",
    ):
        (tmp_path / name).write_bytes(b"local placeholder")
    current = session or Session()
    backend = LocalTextOnnxBackend(
        model_path=tmp_path,
        tokenizer_factory=lambda _path: Tokenizer(),
        session_factory=lambda _path, _providers: current,
        **kwargs,
    )
    return backend, current


def test_batches_inputs_and_masked_mean_normalizes_vectors(tmp_path: Path) -> None:
    backend, session = _backend(tmp_path)

    vectors = list(backend.embed(["a", "bb", "ccc"], batch_size=2))

    assert len(vectors) == 3
    assert [request["input_ids"].shape[0] for request in session.requests] == [2, 1]
    assert set(session.requests[0]) == {"input_ids", "attention_mask", "token_type_ids"}
    assert session.requests[0]["input_ids"].dtype == np.int64
    assert session.requests[0]["token_type_ids"].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert vectors[0][0] == pytest.approx(0.6)
    assert vectors[0][1] == pytest.approx(0.8)
    assert vectors[0][2] == pytest.approx(0.0)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "providers", [["CUDAExecutionProvider"], ["CPUExecutionProvider", "CUDAExecutionProvider"]]
)
def test_rejects_non_cpu_provider_configuration(tmp_path: Path, providers: list[str]) -> None:
    with pytest.raises(ValueError, match="only CPUExecutionProvider"):
        _backend(tmp_path, providers=providers)


def test_rejects_non_local_mode_and_invalid_batches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local_files_only"):
        _backend(tmp_path, local_files_only=False)
    backend, _ = _backend(tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        list(backend.embed(["text"], batch_size=0))
    with pytest.raises(ValueError, match="non-empty"):
        list(backend.embed(["  "], batch_size=1))


def test_rejects_bad_output_dimensions_and_zero_vectors(tmp_path: Path) -> None:
    class BadSession(Session):
        def run(
            self,
            _outputs: object,
            inputs: dict[str, NDArray[np.int64]],
        ) -> list[NDArray[np.float32]]:
            return [
                np.zeros(
                    (len(inputs["input_ids"]), inputs["input_ids"].shape[1], 383), dtype=np.float32
                )
            ]

    backend, _ = _backend(tmp_path, session=BadSession())
    with pytest.raises(ValueError, match="unsupported embeddings"):
        list(backend.embed(["text"], batch_size=1))

    class ZeroSession(Session):
        def run(
            self,
            _outputs: object,
            inputs: dict[str, NDArray[np.int64]],
        ) -> list[NDArray[np.float32]]:
            return [
                np.zeros(
                    (len(inputs["input_ids"]), inputs["input_ids"].shape[1], 384), dtype=np.float32
                )
            ]

    backend, _ = _backend(tmp_path, session=ZeroSession())
    with pytest.raises(ValueError, match="zero or invalid"):
        list(backend.embed(["text"], batch_size=1))
