from pathlib import Path, PurePosixPath
from typing import Protocol
from uuid import UUID


class DocumentStorage(Protocol):
    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> bool: ...


def build_storage_key(document_id: UUID, original_filename: str) -> str:
    safe_name = PurePosixPath(original_filename.replace("\\", "/")).name
    suffix = PurePosixPath(safe_name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise ValueError("unsupported storage extension")
    return f"{document_id}/document{suffix}"


class LocalDocumentStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _resolve(self, key: str) -> Path:
        candidate = (self._root / key).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError("unsafe storage key")
        return candidate

    def put(self, key: str, data: bytes) -> None:
        destination = self._resolve(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def delete(self, key: str) -> bool:
        destination = self._resolve(key)
        if not destination.exists():
            return False
        destination.unlink()
        try:
            destination.parent.rmdir()
        except OSError:
            pass
        return True
