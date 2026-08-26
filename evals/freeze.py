"""Create an immutable local record binding a Git commit to the frozen evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from evals.runner import ENGINE_FINGERPRINT_PATHS, _engine_fingerprint

DEPENDENCY_PATHS = ("pyproject.toml", "uv.lock")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_freeze_record(
    *,
    config_path: Path,
    model_manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"freeze record already exists: {output_path}")
    root = Path(__file__).resolve().parents[1]
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required to create a freeze record")
    tracked_paths = [
        *ENGINE_FINGERPRINT_PATHS,
        config_path.resolve().relative_to(root).as_posix(),
        model_manifest_path.resolve().relative_to(root).as_posix(),
    ]
    status = subprocess.run(  # noqa: S603 - executable resolved from trusted PATH
        [git, "status", "--porcelain", "--", *tracked_paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("engine, configuration, or model manifest is not committed")
    head = subprocess.run(  # noqa: S603 - executable resolved from trusted PATH
        [git, "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload: dict[str, object] = {
        "schema_version": "semantic-v2-freeze-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "engine_commit": head,
        "engine_fingerprint": _engine_fingerprint(),
        "config_sha256": _sha256(config_path),
        "model_manifest_sha256": _sha256(model_manifest_path),
        "dependency_sha256": {relative: _sha256(root / relative) for relative in DEPENDENCY_PATHS},
        "engine_paths": list(ENGINE_FINGERPRINT_PATHS),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("evals/configs/semantic-v2-frozen.json"),
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("infra/models/paraphrase-multilingual-minilm-l12-v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/configs/semantic-v2-freeze.json"),
    )
    args = parser.parse_args()
    payload = create_freeze_record(
        config_path=args.config,
        model_manifest_path=args.model_manifest,
        output_path=args.output,
    )
    print(args.output)
    print(f"engine_commit={payload['engine_commit']}")


if __name__ == "__main__":
    main()
