"""Single-use runner for the independently authored EvidenceDesk holdout v4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.runner import (
    ENGINE_FINGERPRINT_PATHS,
    EvaluationError,
)
from evals.validate_dataset import ValidationReport

DEPENDENCY_PATHS = ("pyproject.toml", "uv.lock")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def claim_one_shot(
    lock_path: Path,
    output_path: Path,
    *,
    allow_holdout: bool,
    evidence: dict[str, str],
) -> None:
    if not allow_holdout:
        raise EvaluationError("holdout execution must be explicitly authorized")
    if output_path.exists():
        raise EvaluationError("holdout output already exists")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"opened_at": datetime.now(UTC).isoformat(), **evidence}
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise EvaluationError("holdout was already opened for this parameter version") from exc


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"{path} must contain an object")
    return value


def verify_freeze_attestation(
    attestation: dict[str, Any],
    freeze: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    engine_commit = freeze.get("engine_commit")
    if not isinstance(engine_commit, str) or not engine_commit:
        raise EvaluationError("engine commit is missing from the freeze record")
    if attestation.get("engine_commit") != engine_commit:
        raise EvaluationError("engine commit differs from the freeze record")
    if freeze.get("engine_paths") != list(ENGINE_FINGERPRINT_PATHS):
        raise EvaluationError("engine path list differs from the evaluator")
    for key in ("engine_fingerprint", "config_sha256", "model_manifest_sha256"):
        label = key.replace("_", " ")
        if freeze.get(key) != evidence.get(key):
            raise EvaluationError(f"{label} differs from the freeze record")
        if attestation.get(key) != freeze.get(key):
            raise EvaluationError(f"{label} differs from the holdout attestation")
    if freeze.get("dependency_sha256") != evidence.get("dependency_sha256"):
        raise EvaluationError("dependency hashes differ from the freeze record")
    if attestation.get("dependency_sha256") != freeze.get("dependency_sha256"):
        raise EvaluationError("dependency hashes differ from the holdout attestation")
    if attestation.get("freeze_sha256") != evidence.get("freeze_sha256"):
        raise EvaluationError("freeze hash differs from the holdout attestation")
    if attestation.get("author_role") != "independent-holdout-author":
        raise EvaluationError("holdout author role is not independent")
    if attestation.get("generated_after_engine_freeze") is not True:
        raise EvaluationError("holdout was not attested after the engine freeze")
    if attestation.get("synthetic_only") is not True:
        raise EvaluationError("holdout is not attested as synthetic-only")
    overlap = attestation.get("development_overlap_check")
    if (
        not isinstance(overlap, dict)
        or not isinstance(overlap.get("method"), str)
        or not overlap["method"]
        or overlap.get("matching_documents") != 0
        or overlap.get("matching_case_ids") != 0
    ):
        raise EvaluationError("holdout development-overlap attestation failed")
    expected_hashes = attestation.get("sha256")
    if not isinstance(expected_hashes, dict):
        raise EvaluationError("holdout attestation hashes are missing")
    for name, evidence_key in (
        ("corpus_manifest.json", "corpus_sha256"),
        ("evaluation_cases.json", "manifest_sha256"),
    ):
        if expected_hashes.get(name) != evidence[evidence_key]:
            raise EvaluationError(f"holdout attestation hash mismatch: {name}")


def verify_holdout_protocol(
    config: dict[str, Any],
    manifest: dict[str, Any],
    attestation: dict[str, Any],
    report: ValidationReport,
) -> None:
    protocol = config.get("holdout_v4_protocol")
    if not isinstance(protocol, dict):
        raise EvaluationError("frozen holdout protocol is missing")
    parameters_version = config.get("parameters_version")
    seed = protocol.get("seed")
    if manifest.get("parameters_version") != parameters_version:
        raise EvaluationError("holdout parameters do not match the frozen configuration")
    if attestation.get("parameters_version") != parameters_version:
        raise EvaluationError("holdout attestation parameters do not match the freeze")
    if manifest.get("seed") != seed or attestation.get("seed") != seed:
        raise EvaluationError("holdout seed does not match the frozen protocol")
    count_fields = (
        "total_cases",
        "minimum_answerable",
        "minimum_unanswerable",
        "minimum_ambiguous_or_adversarial",
    )
    counts: dict[str, int] = {}
    for field in count_fields:
        value = protocol.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvaluationError(f"frozen holdout protocol has invalid {field}")
        counts[field] = value
    required_counts = (
        report.total_cases == counts["total_cases"]
        and report.answerable_cases >= counts["minimum_answerable"]
        and report.unanswerable_cases >= counts["minimum_unanswerable"]
        and report.adversarial_or_ambiguous_cases >= counts["minimum_ambiguous_or_adversarial"]
    )
    if (
        not required_counts
        or report.development_cases != 0
        or report.holdout_cases != report.total_cases
        or not report.document_ids_are_known
        or not report.citations_are_exact
        or not report.extraction_expectations_are_traceable
        or not report.synthetic_only
    ):
        raise EvaluationError("holdout v4 does not satisfy the frozen structural protocol")


def _verify_runtime_dependencies(config: dict[str, Any]) -> dict[str, str]:
    embedding = config.get("embedding")
    if not isinstance(embedding, dict):
        raise EvaluationError("frozen embedding configuration is missing")
    observed: dict[str, str] = {}
    for package, key in (
        ("fastembed", "fastembed_version"),
        ("onnxruntime", "onnxruntime_version"),
    ):
        expected = embedding.get(key)
        if not isinstance(expected, str) or not expected:
            raise EvaluationError(f"frozen {package} version is missing")
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise EvaluationError(f"{package} is not installed") from exc
        if installed != expected:
            raise EvaluationError(
                f"{package} version differs from frozen configuration: "
                f"expected {expected}, observed {installed}"
            )
        observed[key] = installed
    return observed


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise EvaluationError("holdout provenance paths must be inside the repository") from exc


def verify_committed_holdout_inputs(
    paths: tuple[Path, ...],
    *,
    new_holdout_paths: tuple[Path, ...],
    dataset_commit: str,
    freeze_commit: str,
    freeze_path: Path,
    expected_freeze_sha256: str,
    root: Path | None = None,
) -> None:
    """Require a clean, committed dataset strictly newer than the freeze commit."""

    repository = (root or Path(__file__).resolve().parents[1]).resolve()
    git = shutil.which("git")
    if git is None:
        raise EvaluationError("git executable is required to verify holdout provenance")
    for label, commit in (("dataset", dataset_commit), ("freeze", freeze_commit)):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise EvaluationError(f"{label} commit is not a full Git SHA")
    head = subprocess.run(  # noqa: S603 - executable resolved from trusted PATH
        [git, "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dataset_commit != head:
        raise EvaluationError("dataset commit does not match repository HEAD")
    if freeze_commit == dataset_commit:
        raise EvaluationError("freeze commit must strictly precede the dataset commit")
    ancestor = subprocess.run(  # noqa: S603 - full SHAs validated above
        [git, "merge-base", "--is-ancestor", freeze_commit, dataset_commit],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise EvaluationError("freeze commit is not an ancestor of the dataset commit")

    relatives = [_repo_relative(path, repository) for path in paths]
    new_holdout_relatives = [_repo_relative(path, repository) for path in new_holdout_paths]
    if not set(new_holdout_relatives).issubset(relatives):
        raise EvaluationError("new holdout paths must be included in committed inputs")
    status = subprocess.run(  # noqa: S603 - executable resolved from trusted PATH
        [git, "status", "--porcelain", "--untracked-files=all", "--", *relatives],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise EvaluationError("holdout inputs must be committed and clean")
    for relative in relatives:
        tracked = subprocess.run(  # noqa: S603 - commit and repository path are validated
            [git, "cat-file", "-e", f"{dataset_commit}:{relative}"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if tracked.returncode != 0:
            raise EvaluationError(f"holdout input is not tracked at dataset commit: {relative}")
    for relative in new_holdout_relatives:
        existed_at_freeze = subprocess.run(  # noqa: S603 - commit and path are validated
            [git, "cat-file", "-e", f"{freeze_commit}:{relative}"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if existed_at_freeze.returncode == 0:
            raise EvaluationError(f"holdout input already existed at freeze commit: {relative}")

    freeze_relative = _repo_relative(freeze_path, repository)
    historical = subprocess.run(  # noqa: S603 - commit and repository path are validated
        [git, "show", f"{freeze_commit}:{freeze_relative}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if historical.returncode != 0:
        raise EvaluationError("freeze record is not tracked at the freeze commit")
    historical_sha256 = hashlib.sha256(historical.stdout).hexdigest()
    if (
        historical_sha256 != expected_freeze_sha256
        or _sha256(freeze_path) != expected_freeze_sha256
    ):
        raise EvaluationError("freeze record differs from its committed freeze version")


def _verify_engine_commit(
    engine_commit: str,
    *,
    freeze_commit: str,
    config_path: Path,
    model_manifest_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    if re.fullmatch(r"[0-9a-f]{40}", engine_commit) is None:
        raise EvaluationError("frozen engine commit is not a full Git SHA")
    git = shutil.which("git")
    if git is None:
        raise EvaluationError("git executable is required to verify the engine freeze")
    auxiliary_paths: list[str] = []
    for path in (config_path, model_manifest_path):
        try:
            auxiliary_paths.append(path.resolve().relative_to(root).as_posix())
        except ValueError as exc:
            raise EvaluationError(
                "frozen configuration paths must be inside the repository"
            ) from exc
    ancestor = subprocess.run(  # noqa: S603 - full SHA and executable are validated above
        [git, "merge-base", "--is-ancestor", engine_commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise EvaluationError("frozen engine commit is not an ancestor of the dataset commit")
    freeze_ancestor = subprocess.run(  # noqa: S603 - full SHAs are validated by callers
        [git, "merge-base", "--is-ancestor", engine_commit, freeze_commit],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if freeze_ancestor.returncode != 0:
        raise EvaluationError("frozen engine commit is not an ancestor of the freeze commit")
    unchanged = subprocess.run(  # noqa: S603 - full SHA and executable are validated above
        [
            git,
            "diff",
            "--quiet",
            engine_commit,
            "--",
            *ENGINE_FINGERPRINT_PATHS,
            *auxiliary_paths,
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if unchanged.returncode != 0:
        raise EvaluationError("engine sources changed after the frozen engine commit")


def run_holdout_once(
    *,
    manifest_path: Path,
    corpus_path: Path,
    attestation_path: Path,
    freeze_path: Path,
    config_path: Path,
    model_manifest_path: Path,
    model_path: Path,
    lock_path: Path,
    output_path: Path,
    allow_holdout: bool,
) -> dict[str, Any]:
    raise EvaluationError(
        "legacy holdout runner is disabled before all input and lock access; "
        "historical v4-v6 artifacts are immutable and new protocols must use evals.holdout"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("evals/configs/semantic-v2-freeze.json"),
    )
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
        "--model-path",
        type=Path,
        default=Path("models/paraphrase-multilingual-minilm-l12-v2"),
    )
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-holdout", action="store_true")
    args = parser.parse_args()
    result = run_holdout_once(
        manifest_path=args.manifest,
        corpus_path=args.corpus,
        attestation_path=args.attestation,
        freeze_path=args.freeze,
        config_path=args.config,
        model_manifest_path=args.model_manifest,
        model_path=args.model_path,
        lock_path=args.lock,
        output_path=args.output,
        allow_holdout=args.allow_holdout,
    )
    print(args.output)
    print(f"verdict={result['verdict']}")


if __name__ == "__main__":
    main()
