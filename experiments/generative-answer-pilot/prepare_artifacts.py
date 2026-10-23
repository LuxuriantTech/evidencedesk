#!/usr/bin/env python3
"""Prepare hash-pinned local QA artifacts for GitHub-hosted CI.

This script is intentionally stdlib-only. It downloads only the fixed Qwen
GGUF and llama.cpp release archive, then invokes the source checkout's pinned
EvidenceDesk MiniLM downloader after verifying its commit and file hash.
It does not start a model or server. Every failure leaves the exclusive output
directory and its log in place for inspection; reruns need a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
ARTIFACT_MANIFEST = HERE / "ARTIFACT_MANIFEST.json"
PREPARATION_NAME = "PREPARATION_MANIFEST.json"
LOG_NAME = "logs/preparation.log"
EXPECTED_SOURCE_COMMIT = "120e233bb8ff09cabc1bd3c5004a8545039459aa"
MINILM_MANIFEST_REL = "infra/models/paraphrase-multilingual-minilm-l12-v2-text-onnx.json"
MINILM_MANIFEST_SHA256 = "69b870fae9bae6060e43aace617bf8cc17d3e200bfbaf06172c3219f79eb49ec"
MINILM_DOWNLOADER_REL = "scripts/download_local_model.py"
MINILM_DOWNLOADER_SHA256 = "45379801a871598b304b8033e05555f66986afaa8a60994c97fb71cdcca86c82"
MINILM_REPOSITORY = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
MINILM_REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
MINILM_LICENSE = "Apache-2.0"
SOURCE_MANIFEST_SHA256 = "69b870fae9bae6060e43aace617bf8cc17d3e200bfbaf06172c3219f79eb49ec"
MIN_FREE_BYTES = 6 * 1024**3
CHUNK_BYTES = 1024 * 1024


class PreparationError(RuntimeError):
    pass


def sha256_stream(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    while True:
        block = stream.read(CHUNK_BYTES)
        if not block:
            break
        count += len(block)
        digest.update(block)
    return count, digest.hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def log_line(log: Path, message: str) -> None:
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(message.rstrip("\n") + "\n")


def safe_rel(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or "\\" in value or any(part in ("", ".", "..") for part in path.parts):
        raise PreparationError(f"unsafe_relative_path:{value!r}")
    return path.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreparationError("expected_json_object")
    return value


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def download(url: str, target: Path, expected_bytes: int, expected_sha: str, log: Path, label: str) -> None:
    if target.exists() or target.with_name(target.name + ".part").exists():
        raise PreparationError(f"refuse_existing_output:{label}")
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    log_line(log, f"download_start artifact={label} expected_bytes={expected_bytes}")
    request = Request(url, headers={"User-Agent": "EvidenceDesk-CI-artifact-prepare/1.0"})
    try:
        with urlopen(request, timeout=90) as response, part.open("xb") as out:
            status = getattr(response, "status", None)
            if status not in (200, 206):
                raise PreparationError(f"download_http_status:{label}:{status}")
            digest = hashlib.sha256()
            size = 0
            while True:
                block = response.read(CHUNK_BYTES)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                size += len(block)
            out.flush()
            os.fsync(out.fileno())
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        # Keep .part and report only a sanitized exception class/message.
        log_line(log, f"download_failed artifact={label} error={type(exc).__name__}")
        raise PreparationError(f"download_failed:{label}:{type(exc).__name__}") from exc
    actual_sha = digest.hexdigest()
    if size != expected_bytes or actual_sha != expected_sha:
        log_line(log, f"download_integrity_failed artifact={label} bytes={size} sha256={actual_sha}")
        raise PreparationError(f"download_integrity_failed:{label}")
    part.replace(target)
    log_line(log, f"download_verified artifact={label} bytes={size} sha256={actual_sha}")


def inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def normalized_link(member_name: str, linkname: str, hard: bool) -> str:
    if not linkname or "\\" in linkname:
        raise PreparationError(f"unsafe_archive_link:{member_name}")
    link = PurePosixPath(linkname)
    if link.is_absolute():
        raise PreparationError(f"absolute_archive_link:{member_name}")
    base = "" if hard else posixpath.dirname(member_name)
    normalized = posixpath.normpath(posixpath.join(base, linkname))
    if normalized in ("", ".", "..") or normalized.startswith("../"):
        raise PreparationError(f"escaping_archive_link:{member_name}")
    return normalized


def inspect_members(tf: tarfile.TarFile) -> tuple[list[tarfile.TarInfo], dict[str, tarfile.TarInfo]]:
    # This release archive contains a harmless root marker named "." as well
    # as the real `llama-b11798/` root. It has no extracted filesystem path.
    members = [m for m in tf.getmembers()
               if not (m.isdir() and m.name.rstrip("/") in ("", "."))]
    by_name: dict[str, tarfile.TarInfo] = {}
    for member in members:
        name = safe_rel(member.name.rstrip("/") if member.isdir() else member.name)
        if name in by_name:
            raise PreparationError(f"duplicate_archive_path:{name}")
        if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
            raise PreparationError(f"unsupported_archive_type:{name}")
        if member.isfile() and member.size < 0:
            raise PreparationError(f"negative_archive_size:{name}")
        if member.issym() or member.islnk():
            normalized_link(name, member.linkname, member.islnk())
        by_name[name] = member
    for name, member in by_name.items():
        if member.issym() or member.islnk():
            target = normalized_link(name, member.linkname, member.islnk())
            if target not in by_name:
                raise PreparationError(f"dangling_archive_link:{name}")
    roots = {PurePosixPath(name).parts[0] for name in by_name}
    if len(roots) != 1:
        raise PreparationError("archive_must_have_single_top_level_directory")
    return members, by_name


def safe_extract(tf: tarfile.TarFile, members: list[tarfile.TarInfo], dest: Path) -> None:
    root = dest.resolve()
    # Create directories first; no archive entry may traverse a symlink parent.
    for member in members:
        if not member.isdir():
            continue
        rel = safe_rel(member.name.rstrip("/"))
        target = root.joinpath(*PurePosixPath(rel).parts)
        if not inside(root, target.resolve(strict=False)):
            raise PreparationError(f"directory_outside_runtime:{rel}")
        target.mkdir(parents=True, exist_ok=True)
        target.chmod(member.mode & 0o777)
    regulars = [m for m in members if m.isfile()]
    for member in regulars:
        rel = safe_rel(member.name)
        target = root.joinpath(*PurePosixPath(rel).parts)
        if not inside(root, target.parent.resolve(strict=True)) or target.exists():
            raise PreparationError(f"unsafe_or_duplicate_extraction_target:{rel}")
        source = tf.extractfile(member)
        if source is None:
            raise PreparationError(f"archive_file_unreadable:{rel}")
        with source, target.open("xb") as out:
            shutil.copyfileobj(source, out, CHUNK_BYTES)
        target.chmod(member.mode & 0o777)
    for member in members:
        if not member.islnk():
            continue
        rel = safe_rel(member.name)
        link = normalized_link(rel, member.linkname, True)
        target = root.joinpath(*PurePosixPath(rel).parts)
        source = root.joinpath(*PurePosixPath(link).parts)
        if not inside(root, source.resolve(strict=True)) or not source.is_file() or target.exists():
            raise PreparationError(f"unsafe_hardlink:{rel}")
        os.link(source, target)
    for member in members:
        if not member.issym():
            continue
        rel = safe_rel(member.name)
        link = normalized_link(rel, member.linkname, False)
        target = root.joinpath(*PurePosixPath(rel).parts)
        parent = target.parent.resolve(strict=True)
        if not inside(root, parent) or target.exists() or target.is_symlink():
            raise PreparationError(f"unsafe_symlink:{rel}")
        os.symlink(member.linkname, target)
    # Every symlink chain must resolve to an existing path inside the root.
    for member in members:
        if member.issym():
            path = root.joinpath(*PurePosixPath(safe_rel(member.name)).parts)
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise PreparationError(f"unresolvable_symlink:{member.name}") from exc
            if not inside(root, resolved):
                raise PreparationError(f"symlink_escapes_runtime:{member.name}")


def verify_extracted(tf: tarfile.TarFile, by_name: dict[str, tarfile.TarInfo], dest: Path) -> list[dict[str, Any]]:
    root = dest.resolve()
    expected = set(by_name)
    actual: set[str] = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        actual.add(rel)
    if actual != expected:
        raise PreparationError("runtime_tree_inventory_mismatch")
    records: list[dict[str, Any]] = []
    for name, member in sorted(by_name.items()):
        target = root.joinpath(*PurePosixPath(name).parts)
        record: dict[str, Any] = {"path": name}
        if member.isfile():
            if target.is_symlink() or not target.is_file():
                raise PreparationError(f"extracted_type_mismatch:{name}")
            archive_file = tf.extractfile(member)
            if archive_file is None:
                raise PreparationError(f"archive_file_unreadable:{name}")
            with archive_file:
                archive_size, archive_sha = sha256_stream(archive_file)
            disk_size, disk_sha = sha256_file(target)
            if (archive_size, archive_sha) != (member.size, disk_sha) or (disk_size, disk_sha) != (member.size, archive_sha):
                raise PreparationError(f"extracted_content_mismatch:{name}")
            record.update({"type": "file", "bytes": disk_size, "sha256": disk_sha})
        elif member.isdir():
            if not target.is_dir() or target.is_symlink():
                raise PreparationError(f"extracted_type_mismatch:{name}")
            record["type"] = "directory"
        elif member.issym():
            if not target.is_symlink():
                raise PreparationError(f"extracted_type_mismatch:{name}")
            resolved = target.resolve(strict=True)
            if not inside(root, resolved):
                raise PreparationError(f"symlink_escapes_runtime:{name}")
            record.update({"type": "symlink", "target": member.linkname, "resolved_relative": resolved.relative_to(root).as_posix()})
        elif member.islnk():
            if target.is_symlink() or not target.is_file():
                raise PreparationError(f"extracted_type_mismatch:{name}")
            source = root.joinpath(*PurePosixPath(normalized_link(name, member.linkname, True)).parts).resolve(strict=True)
            if not inside(root, source):
                raise PreparationError(f"hardlink_escapes_runtime:{name}")
            a = target.stat()
            b = source.stat()
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise PreparationError(f"hardlink_not_linked:{name}")
            record.update({"type": "hardlink", "target": member.linkname, "resolved_relative": source.relative_to(root).as_posix()})
        records.append(record)
    return records


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 90) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout, check=False)
    text = result.stdout.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise PreparationError(f"command_failed:{Path(command[0]).name}:{result.returncode}")
    return text


def replace_private_paths(text: str, output_root: Path, source_repo: Path) -> str:
    for root, placeholder in ((str(output_root), "$OUTPUT_ROOT"), (str(source_repo), "$SOURCE_REPO")):
        text = text.replace(root, placeholder).replace(root.replace("\\", "/"), placeholder)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path, help="new exclusive artifact directory")
    parser.add_argument("--source-repo", required=True, type=Path, help="exact EvidenceDesk checkout at commit 120e233")
    args = parser.parse_args()
    out_arg = args.output_root.expanduser()
    source_arg = args.source_repo.expanduser()
    if out_arg.exists():
        raise PreparationError("output_root_must_be_new")
    source = source_arg.resolve(strict=True)
    if not (source / ".git").exists() or not (source / "scripts" / "download_local_model.py").is_file():
        raise PreparationError("source_repo_not_checkout_or_native_downloader_missing")
    out_arg.parent.mkdir(parents=True, exist_ok=True)
    out_arg.mkdir(mode=0o700)
    out = out_arg.resolve(strict=True)
    log = out / LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    manifest_path = out / PREPARATION_NAME
    state: dict[str, Any] = {
        "schema": "evidencedesk.generative_ci_preparation.v1",
        "status": "preparing",
        "source_repo_commit": None,
        "artifact_manifest_sha256": None,
        "model": {}, "runtime": {}, "minilm": {},
        "server_config": {}, "resource_scope": {},
        "verified_at_utc": None,
        "started_at_utc": started,
        "logs": {"preparation": LOG_NAME},
    }
    failed_stage = "startup"
    write_manifest(manifest_path, state)
    log_line(log, "preparation_started schema=evidencedesk.generative_ci_preparation.v1")
    try:
        free_bytes = shutil.disk_usage(out).free
        state["resource_scope"] = {"minimum_free_bytes_before_download": MIN_FREE_BYTES,
                                   "observed_free_bytes_before_download": free_bytes}
        if free_bytes < MIN_FREE_BYTES:
            raise PreparationError("insufficient_disk_before_download")
        failed_stage = "manifest_validation"
        raw_artifact_manifest = ARTIFACT_MANIFEST.read_bytes()
        artifact_sha = hashlib.sha256(raw_artifact_manifest).hexdigest()
        pinned = json.loads(raw_artifact_manifest.decode("utf-8"))
        expected = {
            "model": {
                "id": "Qwen3-4B-Q4_K_M.gguf",
                "revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
                "url": "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf",
                "bytes": 2497280256,
                "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
                "license": "Apache-2.0",
            },
            "runtime": {
                "version": "b11798",
                "commit": "9febe63d5a379f8568bae7116a5b2ba844a4a4a6",
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11798/llama-b11798-bin-ubuntu-x64.tar.gz",
                "archive_bytes": 22300188,
                "archive_sha256": "9342761b01a8d82e0567c9673d3dd8f6264964e0ccb27a008193853c757cf563",
                "server_binary": "llama-b11798/llama-server",
                "server_binary_sha256": "64f041152a12b82e223654c424de37861af7a059ee084c6664fa8dd623a2e1d2",
            },
        }
        for section, values in expected.items():
            if not isinstance(pinned.get(section), dict) or any(pinned[section].get(k) != v for k, v in values.items()):
                raise PreparationError(f"artifact_manifest_pin_mismatch:{section}")
        state["artifact_manifest_sha256"] = artifact_sha
        state["server_config"] = pinned.get("server", {})
        state["resource_scope"].update(pinned.get("resource_scope", {}))
        log_line(log, f"artifact_manifest_verified sha256={artifact_sha}")

        failed_stage = "source_checkout_validation"
        commit_out = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=20)
        commit = commit_out.stdout.decode("ascii", errors="ignore").strip()
        if commit_out.returncode != 0 or commit != EXPECTED_SOURCE_COMMIT:
            raise PreparationError("source_checkout_commit_mismatch")
        state["source_repo_commit"] = commit
        minilm_manifest_path = source / MINILM_MANIFEST_REL
        downloader_path = source / MINILM_DOWNLOADER_REL
        for file_path, expected_hash, label in (
            (minilm_manifest_path, MINILM_MANIFEST_SHA256, "minilm_manifest"),
            (downloader_path, MINILM_DOWNLOADER_SHA256, "native_downloader"),
        ):
            size, digest = sha256_file(file_path)
            if digest != expected_hash:
                raise PreparationError(f"source_file_hash_mismatch:{label}")
        native_manifest = load_json(minilm_manifest_path)
        if (native_manifest.get("repository") != MINILM_REPOSITORY
                or native_manifest.get("repository_revision") != MINILM_REVISION
                or native_manifest.get("license") != MINILM_LICENSE
                or not isinstance(native_manifest.get("files"), list)):
            raise PreparationError("native_minilm_manifest_content_mismatch")
        state["minilm"] = {
            "model_manifest_path": MINILM_MANIFEST_REL,
            "model_manifest_sha256": MINILM_MANIFEST_SHA256,
            "downloader_path": MINILM_DOWNLOADER_REL,
            "downloader_sha256": MINILM_DOWNLOADER_SHA256,
            "model_id": native_manifest.get("model_id"),
            "repository": MINILM_REPOSITORY,
            "revision": MINILM_REVISION,
            "license": MINILM_LICENSE,
            "model_dir": "models/minilm",
            "files": native_manifest["files"],
        }
        log_line(log, f"source_checkout_verified commit={commit} minilm_manifest_sha256={MINILM_MANIFEST_SHA256}")

        failed_stage = "download_model"
        model = pinned["model"]
        model_path = out / "models" / model["id"]
        download(model["url"], model_path, model["bytes"], model["sha256"], log, "qwen3-4b-q4-k-m")
        state["model"] = {**{k: model[k] for k in ("id", "revision", "url", "bytes", "sha256", "license")},
                           "path": "models/" + model["id"]}
        write_manifest(manifest_path, state)

        failed_stage = "download_runtime"
        runtime = pinned["runtime"]
        archive_rel = "downloads/llama-runtime.tar.gz"
        archive_path = out / archive_rel
        download(runtime["url"], archive_path, runtime["archive_bytes"], runtime["archive_sha256"], log,
                 "llama-cpp-b11798-ubuntu-x64")
        state["runtime"] = {"version": runtime["version"], "commit": runtime["commit"], "url": runtime["url"],
                            "archive_bytes": runtime["archive_bytes"], "archive_sha256": runtime["archive_sha256"],
                            "archive_path": archive_rel, "root": "runtime/llama-b11798",
                            "server_binary": "runtime/" + runtime["server_binary"],
                            "server_binary_sha256": runtime["server_binary_sha256"]}
        write_manifest(manifest_path, state)

        failed_stage = "runtime_archive_validation_and_extraction"
        runtime_dir = out / "runtime"
        runtime_dir.mkdir()
        with tarfile.open(archive_path, mode="r:gz") as tf:
            members, by_name = inspect_members(tf)
            safe_extract(tf, members, runtime_dir)
            state["runtime"]["archive_entries"] = len(members)
            state["runtime"]["archive_inventory"] = verify_extracted(tf, by_name, runtime_dir)
        binary_rel = state["runtime"]["server_binary"]
        binary_path = out / binary_rel
        if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
            raise PreparationError("pinned_server_binary_missing_or_not_executable")
        binary_size, binary_sha = sha256_file(binary_path)
        if binary_sha != runtime["server_binary_sha256"]:
            raise PreparationError("pinned_server_binary_hash_mismatch")
        state["runtime"]["server_binary_bytes"] = binary_size
        state["runtime"]["server_binary_sha256"] = binary_sha
        log_line(log, f"runtime_archive_and_tree_verified entries={len(members)} server_sha256={binary_sha}")
        write_manifest(manifest_path, state)

        failed_stage = "runtime_identity_help_and_dependencies"
        runtime_root = (out / state["runtime"]["root"]).resolve(strict=True)
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(runtime_root) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        version_text = run_checked([str(binary_path), "--version"], cwd=runtime_root, env=env)
        help_text = run_checked([str(binary_path), "--help"], cwd=runtime_root, env=env)
        required_flags = ("--parallel", "--n-gpu-layers", "--jinja", "--no-context-shift", "--n-predict",
                          "--threads", "--threads-batch", "--ctx-size", "--batch-size", "--ubatch-size",
                          "--reasoning")
        missing = [flag for flag in required_flags if flag not in help_text]
        if missing:
            raise PreparationError("runtime_required_flags_missing:" + ",".join(missing))
        ldd_text = run_checked(["ldd", str(binary_path)], cwd=runtime_root, env=env)
        if "not found" in ldd_text:
            raise PreparationError("runtime_dependency_not_found")
        safe_version = replace_private_paths(version_text, out, source)
        safe_help = replace_private_paths(help_text, out, source)
        safe_ldd = replace_private_paths(ldd_text, out, source)
        logs_dir = out / "logs"
        (logs_dir / "llama-server-version.txt").write_text(safe_version, encoding="utf-8")
        (logs_dir / "llama-server-help.txt").write_text(safe_help, encoding="utf-8")
        (logs_dir / "llama-server-ldd.txt").write_text(safe_ldd, encoding="utf-8")
        state["runtime"].update({
            "server_version": safe_version.strip(),
            "version_sha256": hashlib.sha256(safe_version.encode("utf-8")).hexdigest(),
            "help_sha256": hashlib.sha256(safe_help.encode("utf-8")).hexdigest(),
            "ldd_stdout": safe_ldd,
            "ldd_sha256": hashlib.sha256(safe_ldd.encode("utf-8")).hexdigest(),
            "required_flags_confirmed": list(required_flags),
            "runtime_env": {"LD_LIBRARY_PATH": "runtime/llama-b11798"},
        })
        state["logs"].update({"server_version": "logs/llama-server-version.txt",
                              "server_help": "logs/llama-server-help.txt",
                              "server_ldd": "logs/llama-server-ldd.txt"})
        log_line(log, f"runtime_identity_verified version_sha256={state['runtime']['version_sha256']} help_sha256={state['runtime']['help_sha256']} ldd_sha256={state['runtime']['ldd_sha256']}")
        write_manifest(manifest_path, state)

        failed_stage = "native_minilm_download"
        minilm_dir = out / "models" / "minilm"
        if minilm_dir.exists():
            raise PreparationError("refuse_existing_minilm_output")
        native_output = subprocess.run(
            [sys.executable, str(downloader_path), "--manifest", str(minilm_manifest_path), "--output", str(minilm_dir)],
            cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900, check=False,
        )
        native_output_text = replace_private_paths(native_output.stdout.decode("utf-8", errors="replace"), out, source)
        log_line(log, f"native_minilm_downloader_exit={native_output.returncode}")
        if native_output_text.strip():
            log_line(log, "native_minilm_downloader_output=" + native_output_text.strip().replace("\n", " | "))
        if native_output.returncode != 0:
            raise PreparationError("native_minilm_downloader_failed")
        minilm_expected = {safe_rel(str(item["path"])): item for item in native_manifest["files"]}
        found = {p.relative_to(minilm_dir).as_posix() for p in minilm_dir.rglob("*") if p.is_file() and not p.is_symlink()}
        if found != set(minilm_expected):
            raise PreparationError("minilm_unexpected_or_missing_files")
        for rel, item in minilm_expected.items():
            path = minilm_dir.joinpath(*PurePosixPath(rel).parts)
            size, digest = sha256_file(path)
            if size != int(item["bytes"]) or digest != item["sha256"]:
                raise PreparationError(f"minilm_post_download_hash_mismatch:{rel}")
        state["minilm"]["verified_files"] = sorted(minilm_expected)
        log_line(log, f"minilm_verified files={len(minilm_expected)} revision={MINILM_REVISION}")

        failed_stage = "finalize"
        final_free = shutil.disk_usage(out).free
        state["resource_scope"]["observed_free_bytes_after_download"] = final_free
        state["status"] = "verified"
        state["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["output_root_relative"] = "."
        write_manifest(manifest_path, state)
        log_line(log, "preparation_verified model_runtime_minilm_hashes_all_pass no_model_loaded no_server_started")
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["failed_stage"] = failed_stage
        state["error_type"] = type(exc).__name__
        state["error_code"] = str(exc).split(":", 1)[0][:120]
        write_manifest(manifest_path, state)
        log_line(log, f"preparation_failed stage={failed_stage} error_type={type(exc).__name__} error_code={state['error_code']}")
        print(f"artifact preparation failed; preserved output directory and log (stage={failed_stage}, error={state['error_code']})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
