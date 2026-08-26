import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath


class UploadRejected(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    safe_filename: str
    media_type: str
    sha256: str
    data: bytes


_MEDIA_TYPES: dict[str, tuple[str, frozenset[str]]] = {
    ".pdf": ("application/pdf", frozenset({"application/pdf"})),
    ".txt": ("text/plain", frozenset({"text/plain", "application/octet-stream"})),
    ".md": (
        "text/markdown",
        frozenset({"text/markdown", "text/plain", "application/octet-stream"}),
    ),
}


def load_public_demo_hashes(path: Path) -> frozenset[str]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid public demo allowlist") from exc
    if not isinstance(manifest, dict) or manifest.get("synthetic_only") is not True:
        raise RuntimeError("public demo allowlist must be synthetic-only")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise RuntimeError("public demo allowlist has no documents")
    hashes: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            raise RuntimeError("invalid public demo allowlist document")
        digest = document.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError("invalid public demo allowlist digest")
        hashes.add(digest)
    if len(hashes) != len(documents):
        raise RuntimeError("duplicate public demo allowlist digest")
    return frozenset(hashes)


def validate_upload(
    *, filename: str, declared_content_type: str, data: bytes, max_bytes: int
) -> ValidatedUpload:
    if not data:
        raise UploadRejected("empty_file")
    if len(data) > max_bytes:
        raise UploadRejected("file_too_large")

    safe_filename = PurePosixPath(filename.replace("\\", "/")).name
    suffix = PurePosixPath(safe_filename).suffix.lower()
    media = _MEDIA_TYPES.get(suffix)
    if media is None or declared_content_type not in media[1]:
        raise UploadRejected("unsupported_type")

    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise UploadRejected("invalid_signature")
    else:
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadRejected("invalid_text") from exc
        if "\x00" in decoded or not decoded.strip():
            raise UploadRejected("invalid_text")

    return ValidatedUpload(
        safe_filename=safe_filename,
        media_type=media[0],
        sha256=sha256(data).hexdigest(),
        data=data,
    )
