"""Small shared, non-model utilities for the isolated generation CI pair."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATASETS = ("E4", "E5", "E6")
GENERATION_MANIFEST = HERE / "GENERATION_DATASET_MANIFEST.json"
GENERATION_MANIFEST_SHA256 = "30225ca0f193c7983949628e0a33b748b63402d8b092d16242493b9fb6f70a65"
ARTIFACT_MANIFEST_SHA256 = "1e49f0c8da294fe481607eabf66ab74d7f9182296691d2f9fe824bd650655045"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reviewed_artifact_manifest(path: Path | None = None) -> dict[str, Any]:
    source = path if path is not None else HERE / "ARTIFACT_MANIFEST.json"
    if sha256_file(source) != ARTIFACT_MANIFEST_SHA256:
        raise RuntimeError("reviewed artifact manifest digest mismatch")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("reviewed artifact manifest must be an object")
    return value


def tree_hash(root: Path) -> dict[str, str]:
    if root.is_file():
        return {root.name: sha256_file(root)}
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}


def static_server_identity(server: dict[str, Any]) -> dict[str, Any]:
    """Copy raw endpoint records and omit only /v1/models data[].created and its raw hashes."""
    stable = json.loads(json.dumps(server))
    models = stable.get("/v1/models")
    if isinstance(models, dict):
        body = models.get("body")
        if isinstance(body, dict) and isinstance(body.get("data"), list):
            for item in body["data"]:
                if isinstance(item, dict):
                    item.pop("created", None)
        # Keep raw server bytes and hashes in the receipt, but don't treat the
        # corresponding /v1/models hashes as identity: they include `created`.
        for key in ("body_sha256", "sha256", "raw_body_base64", "raw_body"):
            models.pop(key, None)
    return stable


def server_identity_sha256(server: dict[str, Any]) -> str:
    blob = json.dumps(static_server_identity(server), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def same_server_identity(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return server_identity_sha256(before) == server_identity_sha256(after)


def verify_dataset(dataset: str, root: Path) -> dict[str, str]:
    if dataset not in DATASETS:
        raise RuntimeError("only exposed development datasets E4/E5/E6 are permitted")
    if sha256_file(GENERATION_MANIFEST) != GENERATION_MANIFEST_SHA256:
        raise RuntimeError("generation dataset manifest hash mismatch")
    spec = json.loads(GENERATION_MANIFEST.read_text(encoding="utf-8"))
    if spec.get("scope") != "exposed development only; never fresh or holdout evidence":
        raise RuntimeError("generation manifest has an unexpected scope")
    entries = {item.get("id"): item for item in spec.get("datasets", [])}
    if set(entries) != set(DATASETS):
        raise RuntimeError("generation manifest must allow E4/E5/E6 only")
    entry = entries[dataset]
    if (entry.get("classification") != "exposed-development" or entry.get("license") != "CC0-1.0"
            or entry.get("synthetic_only") is not True):
        raise RuntimeError("dataset must be exposed synthetic CC0 development data")
    expected: dict[str, dict[str, Any]] = {}
    for item in entry.get("files", []):
        rel = Path(item.get("path", ""))
        if rel.is_absolute() or ".." in rel.parts or rel.parts[:2] != ("datasets", dataset):
            raise RuntimeError("unsafe or cross-dataset manifest path")
        if rel.name not in {"evaluation_cases.json", "corpus_manifest.json"}:
            raise RuntimeError("unexpected input file in generation manifest")
        expected[rel.name] = item
    if set(expected) != {"evaluation_cases.json", "corpus_manifest.json"}:
        raise RuntimeError("dataset input allowlist mismatch")
    actual: dict[str, str] = {}
    for name, item in expected.items():
        path = root / dataset / name
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"byte-exact dataset mismatch: {dataset}/{name}")
        actual[name] = item["sha256"]
    corpus = json.loads((root / dataset / "corpus_manifest.json").read_text(encoding="utf-8"))
    if corpus.get("license") != "CC0-1.0":
        raise RuntimeError("corpus does not declare CC0-1.0")
    return actual


def environment_record(expected_prefix: Path, uv_lock: Path) -> dict[str, Any]:
    prefix = Path(sys.prefix).resolve(strict=True)
    expected = expected_prefix.resolve(strict=True)
    if prefix != expected or sys.version_info[:3] != (3, 12, 14):
        raise RuntimeError("runner requires the specified Python 3.12.14 venv")
    packages = sorted((dist.metadata.get("Name", ""), dist.version)
                      for dist in importlib.metadata.distributions())
    package_blob = "\n".join(f"{name}=={version}" for name, version in packages).encode("utf-8")
    return {"python": sys.version, "prefix": str(prefix),
            "executable": str(Path(sys.executable).resolve()), "platform": platform.platform(),
            "installed_distributions_sha256": hashlib.sha256(package_blob).hexdigest(),
            "installed_distributions": packages, "uv_lock_sha256": sha256_file(uv_lock),
            "cohort": "separate 8B public Ubuntu 24.04 development cohort; distinct from 4B R4 and local E3"}
