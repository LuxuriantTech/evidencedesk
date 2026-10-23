"""Strict, standard-library validation for the opt-in local 8B asset bundle."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import shutil
import stat
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "manifest.json"
MANIFEST: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class BundleError(ValueError):
    """An asset bundle is missing, altered, or unsafe to use."""


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def verify_file(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise BundleError(f"required regular file missing: {path.name}")
    size, digest = sha256_file(path)
    if size != expected_size or digest != expected_sha256:
        raise BundleError(f"asset integrity mismatch: {path.name}")


def _ensure_no_symlink(root: Path, relative_path: str) -> Path:
    candidate = root
    for part in Path(relative_path).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise BundleError(f"bundle path contains a symlink: {relative_path}")
    return candidate


def _tar_path(name: str, root_name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        "\\" in name
        or ":" in name
        or "\x00" in name
        or path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != root_name
    ):
        raise BundleError("runtime archive contains an unsafe path")
    return path


def _runtime_inventory(
    archive: tarfile.TarFile, server_binary: str
) -> tuple[
    dict[str, tuple[int, str, bool]],
    set[str],
    dict[str, tarfile.TarInfo],
    dict[str, tarfile.TarInfo],
]:
    root_name = PurePosixPath(server_binary).parts[0]
    members: dict[str, tarfile.TarInfo] = {}
    directories: set[str] = set()
    for member in archive.getmembers():
        path = _tar_path(member.name, root_name)
        name = path.as_posix()
        if name in members:
            raise BundleError("runtime archive has duplicate paths")
        if not (member.isfile() or member.isdir() or member.issym()):
            raise BundleError("runtime archive contains an unsupported entry type")
        members[name] = member
        if member.isdir():
            directories.add(name)
        for parent in path.parents:
            if parent.parts and parent.as_posix() != ".":
                directories.add(parent.as_posix())

    for name in members:
        path = PurePosixPath(name)
        for parent in path.parents:
            parent_name = parent.as_posix()
            if parent_name in members and not members[parent_name].isdir():
                raise BundleError("runtime archive path traverses a non-directory")

    def resolve_file(name: str, visited: frozenset[str] = frozenset()) -> tarfile.TarInfo:
        if name in visited or name not in members:
            raise BundleError("runtime archive link target is missing or cyclic")
        member = members[name]
        if member.isfile():
            return member
        if not member.issym():
            raise BundleError("runtime archive link does not resolve to a regular file")
        target = member.linkname
        if not target or "\\" in target or ":" in target or "\x00" in target:
            raise BundleError("runtime archive link target is unsafe")
        target_path = PurePosixPath(target)
        if target_path.is_absolute() or ".." in target_path.parts:
            # A relative .. component may still remain under the archive root;
            # normalize first, then enforce that the final target stays there.
            if target_path.is_absolute():
                raise BundleError("runtime archive link target is absolute")
        normalized = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
        if normalized == "." or normalized == ".." or normalized.startswith("../"):
            raise BundleError("runtime archive link escapes its root")
        normalized_path = _tar_path(normalized, root_name)
        return resolve_file(normalized_path.as_posix(), visited | {name})

    expected_files: dict[str, tuple[int, str, bool]] = {}
    source_members: dict[str, tarfile.TarInfo] = {}
    for name, member in members.items():
        if member.isdir():
            continue
        target_member = resolve_file(name)
        source_members[name] = target_member
        source = archive.extractfile(target_member)
        if source is None:
            raise BundleError("runtime archive file could not be read")
        digest = hashlib.sha256()
        size = 0
        with source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        if size != target_member.size:
            raise BundleError("runtime archive file size mismatch")
        expected_files[name] = (size, digest.hexdigest(), bool(target_member.mode & 0o111))
    return expected_files, directories, members, source_members


def verify_runtime_tree(root: Path, archive_path: Path, server_binary: str) -> None:
    """Require the extracted tree to equal every file and directory in its archive."""
    root = root.resolve(strict=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        expected_files, expected_dirs, _, _ = _runtime_inventory(archive, server_binary)
        actual_files: dict[str, Path] = {}
        actual_dirs: set[str] = set()
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            parent = Path(directory)
            for dirname in list(dirnames):
                path = parent / dirname
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    raise BundleError("runtime tree contains an unvalidated symlink")
                actual_dirs.add(relative)
            for filename in filenames:
                path = parent / filename
                if path.resolve() == archive_path.resolve():
                    continue
                if path.is_symlink():
                    raise BundleError("runtime tree contains an unvalidated symlink")
                actual_files[path.relative_to(root).as_posix()] = path
    if set(actual_files) != set(expected_files) or actual_dirs != expected_dirs:
        raise BundleError("runtime tree contains missing or unexpected paths")
    for name, path in actual_files.items():
        size, digest = sha256_file(path)
        expected_size, expected_digest, executable = expected_files[name]
        if size != expected_size or digest != expected_digest:
            raise BundleError(f"runtime file integrity mismatch: {name}")
        if bool(path.stat().st_mode & stat.S_IXUSR) != executable:
            raise BundleError(f"runtime file mode mismatch: {name}")


def verify_runtime_archive(archive_path: Path) -> int:
    """Validate and safely extract only the pinned runtime, never the GGUF."""
    runtime = MANIFEST["runtime"]
    verify_file(
        archive_path,
        expected_size=runtime["archive_bytes"],
        expected_sha256=runtime["archive_sha256"],
    )
    with tempfile.TemporaryDirectory(prefix="evidencedesk-llama-runtime-") as temporary:
        extracted = Path(temporary) / "runtime"
        extracted.mkdir()
        _safe_extract(archive_path, extracted)
        verify_runtime_tree(extracted, archive_path, runtime["server_binary"])
        with tarfile.open(archive_path, "r:gz") as archive:
            files, _, _, _ = _runtime_inventory(archive, runtime["server_binary"])
        binary = files.get(runtime["server_binary"])
        if binary is None or binary[1] != runtime["server_binary_sha256"]:
            raise BundleError("pinned runtime server binary hash mismatch")
        return len(files)


def download_runtime_archive(destination: Path) -> int:
    """Explicitly download and validate only the pinned llama.cpp runtime."""
    destination = destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise BundleError("runtime archive destination must not already exist")
    runtime = MANIFEST["runtime"]
    _download(
        runtime["url"],
        destination,
        runtime["archive_bytes"],
        runtime["archive_sha256"],
    )
    return verify_runtime_archive(destination)


def stage_runtime_archive(archive_path: Path, output: Path) -> int:
    """Stage a validated runtime archive into a new runtime-only directory."""
    archive_path = archive_path.expanduser().absolute()
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise BundleError("runtime output directory must not already exist")
    runtime = MANIFEST["runtime"]
    verify_file(
        archive_path,
        expected_size=runtime["archive_bytes"],
        expected_sha256=runtime["archive_sha256"],
    )
    try:
        output.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise BundleError("runtime output directory must not already exist") from exc
    try:
        staged_archive = output / "llama-runtime.tar.gz"
        shutil.copyfile(archive_path, staged_archive)
        verify_file(
            staged_archive,
            expected_size=runtime["archive_bytes"],
            expected_sha256=runtime["archive_sha256"],
        )
        runtime_root = output / "runtime"
        runtime_root.mkdir()
        _safe_extract(staged_archive, runtime_root)
        verify_runtime_tree(runtime_root, staged_archive, runtime["server_binary"])
        with tarfile.open(staged_archive, "r:gz") as archive:
            files, _, _, _ = _runtime_inventory(archive, runtime["server_binary"])
        binary = files.get(runtime["server_binary"])
        if binary is None or binary[1] != runtime["server_binary_sha256"]:
            raise BundleError("pinned runtime server binary hash mismatch")
        return len(files)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def prepare_runtime_directory(output: Path) -> int:
    """Download and stage the pinned runtime only; this never fetches the GGUF."""
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise BundleError("runtime output directory must not already exist")
    runtime = MANIFEST["runtime"]
    with tempfile.TemporaryDirectory(prefix="evidencedesk-runtime-download-") as temporary:
        archive = Path(temporary) / "llama-runtime.tar.gz"
        _download(runtime["url"], archive, runtime["archive_bytes"], runtime["archive_sha256"])
        return stage_runtime_archive(archive, output)


def verify_bundle(root: Path) -> None:
    if root.is_symlink():
        raise BundleError("bundle root may not be a symlink")
    root = root.resolve(strict=True)
    manifest = _ensure_no_symlink(root, "manifest.json")
    if manifest.read_bytes() != MANIFEST_PATH.read_bytes():
        raise BundleError("bundle manifest does not match the pinned manifest")
    model = MANIFEST["model"]
    runtime = MANIFEST["runtime"]
    verify_file(
        _ensure_no_symlink(root, f"models/{model['id']}"),
        expected_size=model["bytes"],
        expected_sha256=model["sha256"],
    )
    verify_file(
        _ensure_no_symlink(root, "runtime/llama-runtime.tar.gz"),
        expected_size=runtime["archive_bytes"],
        expected_sha256=runtime["archive_sha256"],
    )
    binary = _ensure_no_symlink(root, f"runtime/{runtime['server_binary']}")
    if not binary.is_file():
        raise BundleError("runtime server binary is missing")
    if not os.access(binary, os.X_OK):
        raise BundleError("runtime server binary is not executable")
    for section in (model, runtime):
        license_path = _ensure_no_symlink(root, section["license_file"])
        if not license_path.is_file() or license_path.stat().st_size == 0:
            raise BundleError("pinned license text missing from bundle")
    # Prove the staged archive itself expands to the checked binary, without
    # trusting the extracted tree as a substitute for the archived bytes.
    archive_path = root / "runtime" / "llama-runtime.tar.gz"
    with tarfile.open(archive_path, "r:gz") as archive:
        expected_files, _, _, _ = _runtime_inventory(archive, runtime["server_binary"])
    expected_binary = expected_files.get(runtime["server_binary"])
    if expected_binary is None or expected_binary[1] != runtime["server_binary_sha256"]:
        raise BundleError("pinned runtime server binary hash mismatch")
    verify_runtime_tree(root / "runtime", archive_path, runtime["server_binary"])


def _download(url: str, destination: Path, expected_size: int, expected_sha256: str) -> None:
    if not url.startswith("https://"):
        raise BundleError("only pinned HTTPS artifact URLs are allowed")
    request = urllib.request.Request(  # noqa: S310 - pinned HTTPS URLs only
        url, headers={"User-Agent": "EvidenceDesk-local-8B-prep/1"}
    )
    count = 0
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as out:  # noqa: S310
        if response.status != 200:
            raise BundleError("artifact download failed")
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > expected_size:
                raise BundleError("download exceeds pinned byte size")
            digest.update(chunk)
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    if count != expected_size or digest.hexdigest() != expected_sha256:
        raise BundleError("download does not match pinned size and SHA-256")


def _download_license(url: str, destination: Path, license_id: str) -> None:
    if not url.startswith("https://"):
        raise BundleError("only pinned HTTPS license URLs are allowed")
    request = urllib.request.Request(  # noqa: S310 - manifest pins the HTTPS source revision
        url, headers={"User-Agent": "EvidenceDesk-local-8B-prep/1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        if response.status != 200:
            raise BundleError("license text download failed")
        content = response.read(65_537)
    if not content or len(content) > 65_536:
        raise BundleError("license text size is invalid")
    text = content.decode("utf-8")
    markers = {
        "Apache-2.0": ("Apache License", "Version 2.0"),
        "MIT": ("Permission is hereby granted", "THE SOFTWARE IS PROVIDED"),
    }.get(license_id)
    if markers is None or any(marker not in text for marker in markers):
        raise BundleError("license text does not match its declared license")
    destination.write_text(text, encoding="utf-8")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    allowed_root = destination.resolve(strict=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        _, directories, members, source_members = _runtime_inventory(
            archive, str(PurePosixPath(MANIFEST["runtime"]["server_binary"]))
        )
        for name in sorted(directories, key=lambda value: (value.count("/"), value)):
            target = allowed_root / Path(*PurePosixPath(name).parts)
            resolved = target.resolve()
            if resolved != allowed_root and allowed_root not in resolved.parents:
                raise BundleError("runtime archive path escapes extraction root")
            if target.exists() or target.is_symlink():
                if not target.is_dir() or target.is_symlink():
                    raise BundleError("runtime archive directory collision")
            else:
                target.mkdir()

        for name, member in members.items():
            if not member.isfile() and not member.issym():
                continue
            target = allowed_root / Path(*PurePosixPath(name).parts)
            if target.exists() or target.is_symlink():
                raise BundleError("runtime archive file collision")
            source_member = source_members[name]
            source = archive.extractfile(source_member)
            if source is None:
                raise BundleError("runtime archive file could not be read")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if source_member.mode & 0o111 else 0o644)


def prepare_bundle(output: Path) -> None:
    """Download and validate assets into a new, previously absent directory."""
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    required = MANIFEST["resources"]["disk_free_before_download_bytes"]
    if free < required:
        raise BundleError("insufficient free disk space for pinned model")
    try:
        output.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise BundleError("output directory must not already exist") from exc
    try:
        (output / "models").mkdir()
        (output / "runtime").mkdir()
        (output / "licenses").mkdir()
        (output / "manifest.json").write_bytes(MANIFEST_PATH.read_bytes())
        model = MANIFEST["model"]
        runtime = MANIFEST["runtime"]
        _download(
            model["url"],
            output / "models" / model["id"],
            model["bytes"],
            model["sha256"],
        )
        for section in (model, runtime):
            _download_license(
                section["license_url"],
                output / section["license_file"],
                section["license"],
            )
        archive_path = output / "runtime" / "llama-runtime.tar.gz"
        _download(runtime["url"], archive_path, runtime["archive_bytes"], runtime["archive_sha256"])
        _safe_extract(archive_path, output / "runtime")
        binary = output / "runtime" / runtime["server_binary"]
        if not binary.is_file():
            raise BundleError("pinned runtime binary missing from archive")
        binary.chmod(binary.stat().st_mode | 0o111)
        verify_bundle(output)
        (output / "PROVENANCE.md").write_text(
            "# Local 8B pilot asset provenance\n\n"
            "Model: Qwen3-8B-GGUF at the immutable revision in `manifest.json`; "
            "license Apache-2.0 in `licenses/Apache-2.0.txt`.\n\n"
            "Runtime: llama.cpp release b11798 at the commit in `manifest.json`; "
            "license MIT in `licenses/MIT.txt`.\n\n"
            "This bundle contains no application documents or user sessions. "
            "Keep the license terms with any redistribution.\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
