"""Generic, attested, single-use holdout runner for EvidenceDesk."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidencedesk_api.extraction import SupplierExtraction
from evidencedesk_api.providers import EmbeddingProvider, LocalSemanticEmbeddingProvider
from evidencedesk_api.retrieval import EvidenceChunk, RetrievalMethod

from evals.answer_v3_runtime import build_frozen_answer_runtime
from evals.benchmark import _git_head, _hardware
from evals.freeze import _validate_v3_config_data, validate_frozen_holdout_protocol
from evals.holdout_v4 import (
    DEPENDENCY_PATHS,
    _load,
    _verify_engine_commit,
    _verify_runtime_dependencies,
    claim_one_shot,
    verify_committed_holdout_inputs,
    verify_freeze_attestation,
)
from evals.runner import (
    AnswerProviderLike,
    EvaluationError,
    _engine_fingerprint,
    _evaluate_attested_holdout_manifest,
)
from evals.validate_dataset import ValidationReport, validate_manifest

RAW_SCHEMA_VERSION = "evidencedesk-holdout-raw-v4"
LOCK_GATE = "evidencedesk-attested-holdout-v3"


@dataclass(frozen=True, slots=True)
class HoldoutPreflight:
    config: dict[str, Any]
    manifest: dict[str, Any]
    freeze: dict[str, Any]
    attestation: dict[str, Any]
    evidence: dict[str, Any]
    report: ValidationReport
    provider: EmbeddingProvider
    method: RetrievalMethod
    runtime_metadata: dict[str, Any]
    answer_provider: AnswerProviderLike
    extraction_provider: Callable[[list[EvidenceChunk]], SupplierExtraction]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_one_shot_available(
    lock_path: Path,
    output_path: Path,
    *,
    allow_holdout: bool,
) -> None:
    if not allow_holdout:
        raise EvaluationError("holdout execution must be explicitly authorized")
    if output_path.exists():
        raise EvaluationError("holdout output already exists")
    if lock_path.exists():
        raise EvaluationError("holdout was already opened for this parameter version")


def _preflight_stage(label: str, operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except EvaluationError as exc:
        if str(exc).startswith("preflight "):
            raise
        raise EvaluationError(f"preflight {label} failed: {exc}") from exc
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"preflight {label} failed: {exc}") from exc


def verify_holdout_protocol(
    config: dict[str, Any],
    manifest: dict[str, Any],
    attestation: dict[str, Any],
    report: ValidationReport,
) -> None:
    try:
        _validate_v3_config_data(config)
        protocol = validate_frozen_holdout_protocol(config.get("holdout_protocol"))
    except ValueError as exc:
        raise EvaluationError(f"frozen holdout protocol is invalid: {exc}") from exc
    parameters_version = config.get("parameters_version")
    if not isinstance(parameters_version, str) or not parameters_version:
        raise EvaluationError("frozen parameters version is missing")
    seed = protocol.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise EvaluationError("frozen holdout protocol has invalid seed")
    if manifest.get("parameters_version") != parameters_version:
        raise EvaluationError("holdout parameters do not match the frozen configuration")
    if manifest.get("mode") != config.get("mode"):
        raise EvaluationError("holdout mode does not match the frozen configuration")
    if manifest.get("metrics") != config.get("evaluation"):
        raise EvaluationError("holdout metric definitions do not match the frozen configuration")
    if attestation.get("parameters_version") != parameters_version:
        raise EvaluationError("holdout attestation parameters do not match the freeze")
    if manifest.get("seed") != seed or attestation.get("seed") != seed:
        raise EvaluationError("holdout seed does not match the frozen protocol")

    counts: dict[str, int] = {}
    for field in (
        "total_cases",
        "minimum_answerable",
        "minimum_unanswerable",
        "minimum_ambiguous_or_adversarial",
    ):
        value = protocol.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvaluationError(f"frozen holdout protocol has invalid {field}")
        counts[field] = value

    if report.total_cases != counts["total_cases"]:
        raise EvaluationError("holdout case count does not match the frozen protocol")
    if report.answerable_cases < counts["minimum_answerable"]:
        raise EvaluationError("holdout answerable count is below the frozen minimum")
    if report.unanswerable_cases < counts["minimum_unanswerable"]:
        raise EvaluationError("holdout unanswerable count is below the frozen minimum")
    if (
        report.adversarial_or_ambiguous_cases
        < counts["minimum_ambiguous_or_adversarial"]
    ):
        raise EvaluationError("holdout ambiguous/adversarial count is below the frozen minimum")
    if report.development_cases != 0 or report.holdout_cases != report.total_cases:
        raise EvaluationError("holdout manifest contains a non-holdout split")
    if not report.document_ids_are_known:
        raise EvaluationError("holdout gold references an unknown document")
    if not report.citations_are_exact:
        raise EvaluationError("holdout gold citation is not an exact corpus excerpt")
    if not report.extraction_expectations_are_traceable:
        raise EvaluationError("holdout extraction gold is not fully traceable")
    if not report.synthetic_only:
        raise EvaluationError("holdout corpus is not attested synthetic-only")


def preflight_holdout(
    *,
    manifest_path: Path,
    corpus_path: Path,
    attestation_path: Path,
    freeze_path: Path,
    config_path: Path,
    model_manifest_path: Path,
    model_path: Path,
) -> HoldoutPreflight:
    root = Path(__file__).resolve().parents[1]
    evidence: dict[str, Any] = _preflight_stage(
        "input hashing",
        lambda: {
            "dataset_commit": _git_head(),
            "engine_fingerprint": _engine_fingerprint(),
            "manifest_sha256": _sha256(manifest_path),
            "corpus_sha256": _sha256(corpus_path),
            "attestation_sha256": _sha256(attestation_path),
            "config_sha256": _sha256(config_path),
            "model_manifest_sha256": _sha256(model_manifest_path),
            "freeze_sha256": _sha256(freeze_path),
            "dependency_sha256": {
                relative: _sha256(root / relative) for relative in DEPENDENCY_PATHS
            },
        },
    )
    config = _preflight_stage(
        "configuration schema",
        lambda: _validate_v3_config_data(_load(config_path)),
    )
    attestation = _preflight_stage("attestation schema", lambda: _load(attestation_path))
    freeze = _preflight_stage("freeze schema", lambda: _load(freeze_path))
    manifest = _preflight_stage("dataset schema", lambda: _load(manifest_path))
    freeze_commit = attestation.get("freeze_commit")
    if not isinstance(freeze_commit, str) or not freeze_commit:
        raise EvaluationError("preflight attestation schema failed: freeze commit is missing")

    _preflight_stage(
        "Git provenance",
        lambda: verify_committed_holdout_inputs(
            (
                manifest_path,
                corpus_path,
                attestation_path,
                freeze_path,
                config_path,
                model_manifest_path,
            ),
            new_holdout_paths=(manifest_path, corpus_path, attestation_path),
            dataset_commit=str(evidence["dataset_commit"]),
            freeze_commit=freeze_commit,
            freeze_path=freeze_path,
            expected_freeze_sha256=str(evidence["freeze_sha256"]),
        ),
    )
    _preflight_stage(
        "attestation verification",
        lambda: verify_freeze_attestation(attestation, freeze, evidence),
    )
    engine_commit = freeze.get("engine_commit")
    if not isinstance(engine_commit, str) or not engine_commit:
        raise EvaluationError("preflight freeze schema failed: engine commit is missing")
    _preflight_stage(
        "engine freeze verification",
        lambda: _verify_engine_commit(
            engine_commit,
            freeze_commit=freeze_commit,
            config_path=config_path,
            model_manifest_path=model_manifest_path,
        ),
    )
    report = _preflight_stage(
        "dataset and gold validation",
        lambda: validate_manifest(manifest_path, corpus_path=corpus_path),
    )
    _preflight_stage(
        "frozen protocol verification",
        lambda: verify_holdout_protocol(config, manifest, attestation, report),
    )
    dependency_versions = _preflight_stage(
        "runtime dependency verification",
        lambda: _verify_runtime_dependencies(config),
    )
    provider = _preflight_stage(
        "model integrity verification",
        lambda: LocalSemanticEmbeddingProvider(
            manifest_path=model_manifest_path,
            model_path=model_path,
        ),
    )
    method = _preflight_stage(
        "retrieval configuration",
        lambda: RetrievalMethod(str(config["retrieval_method"])),
    )
    answer_runtime = _preflight_stage(
        "frozen answer runtime verification",
        lambda: build_frozen_answer_runtime(config, provider),
    )
    return HoldoutPreflight(
        config=config,
        manifest=manifest,
        freeze=freeze,
        attestation=attestation,
        evidence=evidence,
        report=report,
        provider=provider,
        method=method,
        answer_provider=answer_runtime.answer_provider,
        extraction_provider=answer_runtime.extraction_provider,
        runtime_metadata={
            **_hardware(),
            **dependency_versions,
            "model_manifest_sha256": evidence["model_manifest_sha256"],
            "external_api_cost_usd": 0.0,
        },
    )


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
    _assert_one_shot_available(lock_path, output_path, allow_holdout=allow_holdout)
    prepared = preflight_holdout(
        manifest_path=manifest_path,
        corpus_path=corpus_path,
        attestation_path=attestation_path,
        freeze_path=freeze_path,
        config_path=config_path,
        model_manifest_path=model_manifest_path,
        model_path=model_path,
    )
    claim_one_shot(
        lock_path,
        output_path,
        allow_holdout=allow_holdout,
        evidence={
            "gate": LOCK_GATE,
            "dataset_version": str(prepared.manifest.get("dataset_version", "unknown")),
            "dataset_commit": str(prepared.evidence["dataset_commit"]),
            "engine_commit": str(prepared.freeze["engine_commit"]),
        },
    )

    started = time.perf_counter()
    result: dict[str, Any] = _evaluate_attested_holdout_manifest(
        manifest_path,
        corpus_path,
        provider=prepared.provider,
        method=prepared.method,
        runtime_metadata=prepared.runtime_metadata,
        answer_provider=prepared.answer_provider,
        extraction_provider=prepared.extraction_provider,
    )
    result["schema_version"] = RAW_SCHEMA_VERSION
    result["provenance"] = {
        **prepared.evidence,
        "engine_commit": str(prepared.freeze["engine_commit"]),
        "freeze_commit": str(prepared.attestation["freeze_commit"]),
        "lock_path": str(lock_path),
        "lock_sha256": _sha256(lock_path),
    }
    result["holdout_structure"] = {
        "total_cases": prepared.report.total_cases,
        "answerable_cases": prepared.report.answerable_cases,
        "unanswerable_cases": prepared.report.unanswerable_cases,
        "adversarial_or_ambiguous_cases": prepared.report.adversarial_or_ambiguous_cases,
    }
    result["total_wall_time_ms"] = round((time.perf_counter() - started) * 1_000, 3)
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise EvaluationError("holdout output already exists") from exc
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
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
