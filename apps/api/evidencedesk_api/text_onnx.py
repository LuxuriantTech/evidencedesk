# SPDX-License-Identifier: Apache-2.0
# Tokenizer setup adapted from FastEmbed 0.8.0, Copyright 2024 Qdrant.
# See third_party/fastembed.NOTICE and third_party/fastembed.LICENSE.
"""Text-only, local ONNX embedding backend matching the pinned FastEmbed path."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np


class _Encoding(Protocol):
    ids: Sequence[int]
    attention_mask: Sequence[int]
    type_ids: Sequence[int]


class _Tokenizer(Protocol):
    padding: Any

    def enable_truncation(self, *, max_length: int) -> None: ...

    def enable_padding(self, *, pad_id: int, pad_token: str) -> None: ...

    def add_special_tokens(self, tokens: list[Any]) -> int: ...

    def encode_batch(self, inputs: list[str]) -> list[_Encoding]: ...


class _Input(Protocol):
    name: str
    shape: Sequence[Any]


class _Session(Protocol):
    def get_inputs(self) -> Sequence[_Input]: ...

    def run(
        self, output_names: Sequence[str] | None, input_feed: dict[str, Any]
    ) -> Sequence[Any]: ...


def _load_tokenizer(model_dir: Path) -> _Tokenizer:
    """Configure the tokenizer exactly as FastEmbed 0.8's local loader does."""
    from tokenizers import AddedToken, Tokenizer

    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads(
            (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        tokens_map = json.loads((model_dir / "special_tokens_map.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid local embedding tokenizer configuration") from exc
    model_max = tokenizer_config.get("model_max_length")
    configured_max = tokenizer_config.get("max_length")
    max_length = (
        min(model_max, configured_max)
        if model_max is not None and configured_max is not None
        else model_max or configured_max
    )
    if not isinstance(max_length, int) or not 1 <= max_length <= 512:
        raise ValueError("unsupported tokenizer maximum length")
    pad_id = config.get("pad_token_id", 0)
    pad_token = tokenizer_config.get("pad_token")
    if not isinstance(pad_id, int) or not isinstance(pad_token, str):
        raise ValueError("unsupported tokenizer padding configuration")
    if not isinstance(tokens_map, dict):
        raise ValueError("unsupported special token map")

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=max_length)
    if not tokenizer.padding:
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
    for token in tokens_map.values():
        if isinstance(token, str):
            tokenizer.add_special_tokens([token])
        elif isinstance(token, dict):
            tokenizer.add_special_tokens([AddedToken(**token)])
        else:
            raise ValueError("unsupported special token configuration")
    return cast(_Tokenizer, tokenizer)


def _load_session(model_file: Path, providers: Sequence[str]) -> _Session:
    import onnxruntime  # type: ignore[import-untyped]

    return cast(
        _Session,
        onnxruntime.InferenceSession(str(model_file), providers=list(providers)),
    )


class LocalTextOnnxBackend:
    """Offline text embeddings from verified local tokenizer and ONNX model files."""

    def __init__(
        self,
        *,
        model_path: Path,
        local_files_only: bool = True,
        providers: list[str] | None = None,
        tokenizer_factory: Any = None,
        session_factory: Any = None,
    ) -> None:
        if local_files_only is not True:
            raise ValueError("local_files_only must be true")
        selected_providers = providers or ["CPUExecutionProvider"]
        if selected_providers != ["CPUExecutionProvider"]:
            raise ValueError("only CPUExecutionProvider is supported")
        root = model_path.resolve()
        required = (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model_optimized.onnx",
        )
        paths = {name: (root / name).resolve() for name in required}
        if any(not path.is_relative_to(root) or not path.is_file() for path in paths.values()):
            raise ValueError("required local text model files are missing")
        self._tokenizer = (tokenizer_factory or _load_tokenizer)(root)
        self._session = (session_factory or _load_session)(
            paths["model_optimized.onnx"], selected_providers
        )
        inputs = self._session.get_inputs()
        self._input_shapes = {str(item.name): item.shape for item in inputs}
        self._input_names = set(self._input_shapes)
        if not {"input_ids", "attention_mask"}.issubset(self._input_names):
            raise ValueError("unsupported ONNX input names")
        if self._input_names - {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("unsupported ONNX input names")
        if (
            "token_type_ids" in self._input_names
            and self._input_shapes["token_type_ids"] != self._input_shapes["input_ids"]
        ):
            raise ValueError("unsupported token_type_ids input shape")

    def embed(self, documents: list[str], batch_size: int) -> Iterable[Sequence[float]]:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if any(not isinstance(document, str) or not document.strip() for document in documents):
            raise ValueError("embedding input must contain non-empty strings")
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            encoded = self._tokenizer.encode_batch(batch)
            if len(encoded) != len(batch) or not encoded:
                raise ValueError("tokenizer returned the wrong batch length")
            widths = {len(item.ids) for item in encoded}
            if len(widths) != 1 or any(
                len(item.attention_mask) != len(item.ids) for item in encoded
            ):
                raise ValueError("tokenizer returned invalid padded sequences")
            ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
            if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[1] > 512:
                raise ValueError("tokenizer returned unsupported input shapes")
            feed: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.asarray(
                    [np.zeros(len(item.ids), dtype=np.int64) for item in encoded], dtype=np.int64
                )
            outputs = self._session.run(None, feed)
            if not outputs:
                raise ValueError("ONNX session returned no embeddings")
            token_embeddings = np.asarray(outputs[0], dtype=np.float64)
            if (
                token_embeddings.ndim != 3
                or token_embeddings.shape[:2] != ids.shape
                or token_embeddings.shape[2] != 384
                or not np.isfinite(token_embeddings).all()
            ):
                raise ValueError("ONNX session returned unsupported embeddings")
            weights = mask.astype(np.float64)[..., None]
            pooled = (token_embeddings * weights).sum(axis=1) / np.maximum(
                weights.sum(axis=1), 1e-9
            )
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
                raise ValueError("ONNX session returned zero or invalid vectors")
            normalized = pooled / norms
            for vector in normalized:
                yield vector.astype(np.float32)
