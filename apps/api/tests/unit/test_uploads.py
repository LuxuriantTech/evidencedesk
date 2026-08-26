from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest
from evidencedesk_api.storage import LocalDocumentStorage, build_storage_key
from evidencedesk_api.uploads import (
    UploadRejected,
    load_public_demo_hashes,
    validate_upload,
)

ROOT = Path(__file__).resolve().parents[4]


def test_valid_markdown_is_identified_and_named_safely() -> None:
    upload = validate_upload(
        filename="../../supplier-notes.md",
        declared_content_type="text/markdown",
        data=b"# Supplier notes\n\nRenewal: 2027-05-31\n",
        max_bytes=1_024,
    )

    assert upload.safe_filename == "supplier-notes.md"
    assert upload.media_type == "text/markdown"
    assert len(upload.sha256) == 64


@pytest.mark.parametrize(
    ("filename", "content_type", "data", "code"),
    [
        ("empty.txt", "text/plain", b"", "empty_file"),
        ("fake.pdf", "application/pdf", b"<html>not a pdf</html>", "invalid_signature"),
        ("binary.txt", "text/plain", b"hello\x00world", "invalid_text"),
        ("script.html", "text/html", b"<h1>x</h1>", "unsupported_type"),
    ],
)
def test_invalid_file_content_is_rejected(
    filename: str, content_type: str, data: bytes, code: str
) -> None:
    with pytest.raises(UploadRejected, match=code):
        validate_upload(
            filename=filename,
            declared_content_type=content_type,
            data=data,
            max_bytes=1_024,
        )


def test_size_limit_is_checked_before_storage() -> None:
    with pytest.raises(UploadRejected, match="file_too_large"):
        validate_upload(
            filename="large.txt",
            declared_content_type="text/plain",
            data=b"a" * 11,
            max_bytes=10,
        )


def test_local_storage_uses_only_a_generated_key(tmp_path: Path) -> None:
    document_id = UUID("e8ae755c-124d-4970-a8e6-66acf5709f62")
    key = build_storage_key(document_id, "../../supplier-notes.md")
    storage = LocalDocumentStorage(tmp_path)

    storage.put(key, b"safe")

    assert key == "e8ae755c-124d-4970-a8e6-66acf5709f62/document.md"
    assert (tmp_path / key).read_bytes() == b"safe"
    assert not (tmp_path.parent / "supplier-notes.md").exists()


def test_public_demo_allowlist_matches_only_versioned_synthetic_files() -> None:
    paths = [
        ROOT / "examples/demo-supplier-note.md",
        ROOT / "datasets/generated/northstar_master_services_agreement.pdf",
        ROOT / "datasets/sources/routing_delay_incident.md",
        ROOT / "datasets/sources/supplier_register.txt",
    ]

    assert load_public_demo_hashes(ROOT / "datasets/public_demo_uploads.json") == frozenset(
        sha256(path.read_bytes()).hexdigest() for path in paths
    )
