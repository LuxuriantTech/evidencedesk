from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import download_local_model


class ExactLengthResponse:
    status = 200

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.read_calls = 0

    def __enter__(self) -> ExactLengthResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.read_calls += 1
        if not self.payload:
            raise AssertionError("downloader waited for EOF after the attested byte length")
        block, self.payload = self.payload[:size], self.payload[size:]
        return block


def test_downloader_stops_at_the_attested_size_without_waiting_for_server_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified-model-bytes"
    manifest = {
        "repository": "synthetic/repository",
        "repository_revision": "a" * 40,
        "files": [
            {
                "path": "model.onnx",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    response = ExactLengthResponse(payload)
    monkeypatch.setattr(download_local_model, "urlopen", lambda *_args, **_kwargs: response)

    download_local_model.download_model(manifest_path, tmp_path / "model")

    assert (tmp_path / "model/model.onnx").read_bytes() == payload
    assert response.read_calls == 1
