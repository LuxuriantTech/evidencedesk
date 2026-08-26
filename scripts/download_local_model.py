"""Download the pinned public ONNX bundle and verify every runtime file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class ModelDownloadError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ModelDownloadError("invalid model manifest")
    return value


def verify_model(manifest: dict[str, Any], output: Path) -> None:
    root = output.resolve()
    for item in manifest["files"]:
        relative = Path(str(item["path"]))
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ModelDownloadError(f"missing model file: {relative.as_posix()}")
        if target.stat().st_size != int(item["bytes"]):
            raise ModelDownloadError(f"size mismatch: {relative.as_posix()}")
        if _sha256(target) != str(item["sha256"]):
            raise ModelDownloadError(f"sha256 mismatch: {relative.as_posix()}")


def _download_file(
    *,
    repository: str,
    revision: str,
    item: dict[str, Any],
    output: Path,
) -> None:
    relative = Path(str(item["path"]))
    target = (output.resolve() / relative).resolve()
    if not target.is_relative_to(output.resolve()):
        raise ModelDownloadError(f"model file escapes model directory: {relative.as_posix()}")
    expected_bytes = int(item["bytes"])
    expected_sha256 = str(item["sha256"])
    if target.exists():
        if target.stat().st_size == expected_bytes and _sha256(target) == expected_sha256:
            return
        raise ModelDownloadError(f"existing model file is invalid: {relative.as_posix()}")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    if partial.exists() and partial.stat().st_size > expected_bytes:
        partial.unlink()
    url = (
        f"https://huggingface.co/{quote(repository, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/{quote(relative.as_posix(), safe='/')}?download=true"
    )
    last_error: Exception | None = None
    for _attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "EvidenceDesk-model-fetch/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=60) as response:  # noqa: S310 - HTTPS host is fixed
                append = offset > 0 and response.status == 206
                remaining = expected_bytes - offset if append else expected_bytes
                with partial.open("ab" if append else "wb") as handle:
                    while remaining > 0:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        handle.write(block)
                        remaining -= len(block)
        except (HTTPError, TimeoutError, URLError, OSError) as exc:
            last_error = exc
            continue
        if partial.stat().st_size != expected_bytes:
            last_error = ModelDownloadError(
                f"incomplete download: {relative.as_posix()} "
                f"({partial.stat().st_size}/{expected_bytes} bytes)"
            )
            continue
        if _sha256(partial) != expected_sha256:
            raise ModelDownloadError(f"sha256 mismatch after download: {relative.as_posix()}")
        partial.replace(target)
        return
    raise ModelDownloadError(f"download failed: {relative.as_posix()}: {last_error}")


def download_model(manifest_path: Path, output: Path) -> None:
    manifest = _load_manifest(manifest_path)
    try:
        verify_model(manifest, output)
        print(f"verified existing model: {output}")
        return
    except ModelDownloadError:
        pass
    output.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise ModelDownloadError("invalid model manifest")
        _download_file(
            repository=str(manifest["repository"]),
            revision=str(manifest["repository_revision"]),
            item=item,
            output=output,
        )
    verify_model(manifest, output)
    print(f"downloaded and verified model: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("infra/models/paraphrase-multilingual-minilm-l12-v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/paraphrase-multilingual-minilm-l12-v2"),
    )
    args = parser.parse_args()
    download_model(args.manifest, args.output)


if __name__ == "__main__":
    main()
