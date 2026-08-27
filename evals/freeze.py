"""Create an immutable local record binding a Git commit to the frozen evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidencedesk_api.retrieval import (
    DEFAULT_RETRIEVAL_LIMIT,
    NAMED_ENTITY_DOCUMENT_BOOST,
    RRF_K,
)

from evals.answer_v3_runtime import AnswerRuntimeConfigError, validate_frozen_answer_config
from evals.recalculate import recalculate_metrics
from evals.runner import ENGINE_FINGERPRINT_PATHS, _engine_fingerprint
from evals.strategy_benchmark_v3 import (
    CORPUS_PATH,
    PARTITION_MANIFESTS,
    PLAN_PATH,
    QUALITY_METRICS,
    _aggregate,
    _core_decision_digest,
    _decision_digest,
    _load,
    _select_configuration,
    load_experiment_configurations,
)

DEPENDENCY_PATHS = ("pyproject.toml", "uv.lock")
DEFAULT_CONFIG_PATH = Path("evals/configs/answer-v3-frozen-v7.json")
DEFAULT_FREEZE_PATH = Path("evals/configs/answer-v3-freeze-v7.json")
EMBEDDING_MANIFEST_PATH = "infra/models/paraphrase-multilingual-minilm-l12-v2.json"
DEVELOPMENT_ARTIFACT_BINDINGS = {
    "development_manifest_sha256": "datasets/development_v3/evaluation_cases.json",
    "development_corpus_sha256": "datasets/development_v3/corpus_manifest.json",
    "development_generation_manifest_sha256": "datasets/development_v3/generation_manifest.json",
    "experiment_plan_sha256": "evals/configs/answer-v3-experiment-plan.json",
    "development_comparison_sha256": (
        "artifacts/evaluations/development_v3/strategy_v3/comparison.json"
    ),
    "selected_evaluation_artifact_sha256": (
        "artifacts/evaluations/development_v3/"
        "frozen-runtime-v4-locked-final/selection-r1.json"
    ),
    "development_recalculation_sha256": (
        "artifacts/evaluations/development_v3/frozen-runtime-v4-locked-final/"
        "selection-r1-recalculated.json"
    ),
    "development_runtime_summary_sha256": (
        "artifacts/evaluations/development_v3/frozen-runtime-v4-locked-final/summary.json"
    ),
}
FROZEN_HOLDOUT_PROTOCOL: dict[str, object] = {
    "protocol_label": "holdout-v7-independent-answer-v3",
    "seed": 2026082707,
    "total_cases": 40,
    "minimum_answerable": 25,
    "minimum_unanswerable": 10,
    "minimum_ambiguous_or_adversarial": 5,
    "families_absent_from_development_v3_required": True,
    "generation_after_engine_freeze": True,
    "independent_author_required": True,
    "independent_author_may_not_inspect_engine_or_prior_gold": True,
    "preflight_must_pass_before_lock": True,
    "preflight_checks": [
        "schema",
        "attestation",
        "filename",
        "dataset",
        "gold",
        "configuration",
    ],
    "execution_limit": 1,
    "gold_may_not_change_after_commit": True,
    "recalculation_from_raw_artifact_only": True,
    "metrics_frozen_before_generation": True,
    "models_thresholds_prompts_and_seeds_frozen_before_generation": True,
}
FROZEN_EVALUATION_PROTOCOL: dict[str, object] = {
    "citation_match": (
        "exact document and page; returned normalized excerpt must be an informative subspan "
        "of expected evidence with at least two alphanumeric tokens"
    ),
    "answer_value_match": (
        "typed date and money normalization or one-way expected-value containment"
    ),
    "extraction_match": "typed exact equivalence plus a strict citation per value",
    "status_match": (
        "ambiguous must be ambiguous; unanswerable and adversarial must be abstained"
    ),
    "retrieval_match": (
        "gold document and page must occur in the first five raw retrieved candidates"
    ),
    "targets": {
        "citation_precision": 0.9,
        "citation_recall": 0.9,
        "citation_case_accuracy": 0.9,
        "abstention_accuracy": 0.85,
        "extraction_f1": 0.9,
        "error_rate": 0.0,
        "schema_error_rate": 0.0,
    },
}
FINAL_RUNTIME_PATHS = {
    partition: [
        (
            "artifacts/evaluations/development_v3/"
            f"frozen-runtime-v4-locked-final/{partition}-r{repetition}.json"
        )
        for repetition in (1, 2, 3)
    ]
    for partition in ("calibration", "selection")
}
_V3_CONFIG_KEYS = frozenset(
    {
        "parameters_version",
        "frozen_at",
        "development_dataset_version",
        *DEVELOPMENT_ARTIFACT_BINDINGS,
        "engine_fingerprint_at_selection",
        "mode",
        "retrieval_method",
        "answer_candidate_limit",
        "retrieval_metric_cutoff",
        "rank_fusion",
        "embedding",
        "answer_engine",
        "extraction_engine",
        "evaluation",
        "holdout_protocol",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_holdout_protocol(protocol: object) -> dict[str, object]:
    if not isinstance(protocol, dict) or protocol != FROZEN_HOLDOUT_PROTOCOL:
        raise ValueError("frozen holdout protocol differs from the registered v7 protocol")
    return protocol


def _development_target_gate(selection_quality: dict[str, Any]) -> dict[str, Any]:
    targets = FROZEN_EVALUATION_PROTOCOL["targets"]
    if not isinstance(targets, dict):
        raise ValueError("frozen evaluation targets are invalid")
    failures: dict[str, dict[str, float]] = {}
    for metric, raw_target in targets.items():
        observed = selection_quality.get(metric)
        if (
            isinstance(raw_target, bool)
            or not isinstance(raw_target, (int, float))
            or isinstance(observed, bool)
            or not isinstance(observed, (int, float))
        ):
            raise ValueError(f"development target metric is invalid: {metric}")
        target = float(raw_target)
        measured = float(observed)
        failed = measured != target if target == 0.0 else measured < target
        if failed:
            failures[str(metric)] = {"observed": measured, "target": target}
    return {
        "failed_targets": failures,
        "note": (
            "Selection followed the pre-registered maximin rule; no tuning followed "
            "selection access."
        ),
        "verdict": "PASS" if not failures else "FAIL",
    }


def _artifact_paths(
    aggregate: object,
    *,
    root: Path,
    expected_paths: list[str] | None = None,
) -> list[Path]:
    if not isinstance(aggregate, dict) or not isinstance(aggregate.get("artifacts"), list):
        raise ValueError("development aggregate artifact list is invalid")
    entries = aggregate["artifacts"]
    observed_paths: list[str] = []
    resolved_paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("development aggregate artifact entry is invalid")
        relative = entry.get("path")
        expected_sha256 = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise ValueError("development aggregate artifact provenance is invalid")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("development artifact path escapes the repository") from exc
        if _sha256(candidate) != expected_sha256:
            raise ValueError(f"development aggregate artifact hash mismatch: {relative}")
        observed_paths.append(relative)
        resolved_paths.append(candidate)
    if expected_paths is not None and observed_paths != expected_paths:
        raise ValueError("development aggregate artifact paths differ from the frozen run set")
    return resolved_paths


def _validate_recalculation(raw_path: Path, recalculation_path: Path) -> None:
    raw = _load(raw_path)
    recalculation = _load(recalculation_path)
    if recalculation.get("schema_version") != "evidencedesk-evaluation-recalculation-v4":
        raise ValueError("development recalculation schema is invalid")
    if recalculation.get("raw_artifact_sha256") != _sha256(raw_path):
        raise ValueError("development recalculation does not bind its raw artifact")
    recomputed = recalculate_metrics(raw)
    for key, value in recomputed.items():
        if recalculation.get(key) != value:
            raise ValueError(f"development recalculation metric mismatch: {key}")


def _validate_comparison_raw(
    raw: dict[str, Any],
    *,
    expected_fingerprint: str,
    expected_manifest_sha256: str,
    expected_corpus_sha256: str,
    expected_configuration: dict[str, Any],
    expected_partition: str,
    expected_repetition: int,
) -> None:
    if raw.get("schema_version") != "evaluation-result-v4":
        raise ValueError("development comparison raw schema is invalid")
    provenance = {
        "engine_fingerprint": expected_fingerprint,
        "manifest_sha256": expected_manifest_sha256,
        "corpus_sha256": expected_corpus_sha256,
    }
    for key, expected_provenance in provenance.items():
        if raw.get(key) != expected_provenance:
            raise ValueError(f"development comparison raw {key} mismatch")
    benchmark = raw.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("development comparison benchmark metadata is missing")
    expected_benchmark = {
        "configuration": expected_configuration,
        "partition": expected_partition,
        "repetition": expected_repetition,
    }
    for key, expected_benchmark_value in expected_benchmark.items():
        if benchmark.get(key) != expected_benchmark_value:
            raise ValueError(f"development comparison benchmark {key} mismatch")
    if benchmark.get("decision_sha256") != _decision_digest(raw):
        raise ValueError("development comparison decision digest mismatch")
    total_wall_ms = benchmark.get("total_wall_ms")
    model_load_ms = benchmark.get("model_load_ms")
    peak_rss_bytes = benchmark.get("peak_rss_bytes")
    if (
        isinstance(total_wall_ms, bool)
        or not isinstance(total_wall_ms, (int, float))
        or not math.isfinite(float(total_wall_ms))
        or isinstance(model_load_ms, bool)
        or not isinstance(model_load_ms, (int, float))
        or not math.isfinite(float(model_load_ms))
        or float(model_load_ms) < 0
        or float(total_wall_ms) < max(float(model_load_ms), float(raw["latency_p95_ms"]))
    ):
        raise ValueError("development comparison benchmark timing is invalid")
    if (
        isinstance(peak_rss_bytes, bool)
        or not isinstance(peak_rss_bytes, int)
        or peak_rss_bytes <= 0
    ):
        raise ValueError("development comparison benchmark peak RSS is invalid")
    recomputed = recalculate_metrics(raw)
    for metric in QUALITY_METRICS:
        recorded_metric = raw.get(metric)
        recomputed_metric = recomputed.get(metric)
        if metric in {"latency_median_ms", "latency_p95_ms"} and isinstance(
            recorded_metric, (int, float)
        ) and not isinstance(recorded_metric, bool) and isinstance(
            recomputed_metric, (int, float)
        ) and not isinstance(recomputed_metric, bool):
            matches = abs(float(recorded_metric) - float(recomputed_metric)) <= 0.001001
        else:
            matches = recorded_metric == recomputed_metric
        if not matches:
            raise ValueError(f"development comparison raw metric mismatch: {metric}")
    recorded_counts = raw.get("metric_counts")
    recomputed_counts = recomputed.get("metric_counts")
    if not isinstance(recorded_counts, dict) or not isinstance(recomputed_counts, dict):
        raise ValueError("development comparison metric counts are invalid")
    if any(recomputed_counts.get(key) != value for key, value in recorded_counts.items()):
        raise ValueError("development comparison metric counts do not recalculate")


def _validate_strategy_comparison(
    comparison: dict[str, Any],
    *,
    root: Path,
    expected_fingerprint: str,
) -> tuple[dict[str, Any], list[str]]:
    if comparison.get("schema_version") != "evidencedesk-strategy-comparison-v3":
        raise ValueError("development strategy comparison schema is invalid")
    expected_partition_hashes = {
        key: _sha256(path) for key, path in PARTITION_MANIFESTS.items()
    }
    comparison_provenance = {
        "engine_fingerprint": expected_fingerprint,
        "experiment_plan_sha256": _sha256(PLAN_PATH),
        "corpus_sha256": _sha256(CORPUS_PATH),
        "partition_manifest_sha256": expected_partition_hashes,
        "repetitions": 3,
        "selection_rule": _load(PLAN_PATH).get("final_selection_rule"),
    }
    for key, expected in comparison_provenance.items():
        if comparison.get(key) != expected:
            raise ValueError(f"development strategy comparison {key} mismatch")
    comparison_items = comparison.get("comparison")
    if not isinstance(comparison_items, dict):
        raise ValueError("development strategy comparison is missing")
    registered_configurations = {
        str(item["id"]): item for item in load_experiment_configurations()
    }
    if set(comparison_items) != set(registered_configurations):
        raise ValueError("development strategy comparison configurations are incomplete")
    evidence_paths: list[str] = []
    for config_id, item in comparison_items.items():
        if not isinstance(item, dict):
            raise ValueError("development strategy comparison entry is invalid")
        configuration = item.get("configuration")
        if configuration != registered_configurations[config_id]:
            raise ValueError("development strategy comparison configuration mismatch")
        for partition in ("calibration", "selection"):
            aggregate = item.get(partition)
            if aggregate is None:
                continue
            paths = _artifact_paths(aggregate, root=root)
            if len(paths) != 3:
                raise ValueError("development strategy aggregate must contain three runs")
            for repetition, path in enumerate(paths, start=1):
                _validate_comparison_raw(
                    _load(path),
                    expected_fingerprint=expected_fingerprint,
                    expected_manifest_sha256=expected_partition_hashes[partition],
                    expected_corpus_sha256=_sha256(CORPUS_PATH),
                    expected_configuration=registered_configurations[config_id],
                    expected_partition=partition,
                    expected_repetition=repetition,
                )
            evidence_paths.extend(str(path.relative_to(root)) for path in paths)
            if _aggregate(paths) != aggregate:
                raise ValueError("development strategy aggregate does not recalculate")
        calibration = item.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError("development strategy calibration aggregate is missing")
        counts = calibration.get("metric_counts")
        calibration_rejected = (
            not calibration.get("stable_decisions")
            or not isinstance(counts, list)
            or any(
                not isinstance(count, dict)
                or int(count.get("errors", -1)) > 0
                or int(count.get("schema_errors", -1)) > 0
                for count in counts
            )
        )
        expected_gate = "REJECT" if calibration_rejected else "PASS"
        if item.get("calibration_gate") != expected_gate:
            raise ValueError("development strategy calibration gate mismatch")
        if calibration_rejected == isinstance(item.get("selection"), dict):
            raise ValueError("development strategy selection partition gating mismatch")
    selected_id = _select_configuration(comparison_items)
    if not isinstance(selected_id, str):
        raise ValueError("development strategy selection did not produce a winner")
    if comparison.get("selected_configuration") != selected_id:
        raise ValueError("development strategy selection does not follow the frozen rule")
    return comparison_items, evidence_paths


def _validate_development_evidence(loaded: dict[str, Any], root: Path) -> list[str]:
    evidence_paths = list(DEVELOPMENT_ARTIFACT_BINDINGS.values())
    for key, relative in DEVELOPMENT_ARTIFACT_BINDINGS.items():
        expected = loaded.get(key)
        if not isinstance(expected, str) or expected != _sha256(root / relative):
            raise ValueError(f"development artifact hash mismatch: {relative}")

    comparison_path = root / DEVELOPMENT_ARTIFACT_BINDINGS[
        "development_comparison_sha256"
    ]
    comparison = _load(comparison_path)
    comparison_items, comparison_evidence_paths = _validate_strategy_comparison(
        comparison,
        root=root,
        expected_fingerprint=str(loaded["engine_fingerprint_at_selection"]),
    )
    evidence_paths.extend(comparison_evidence_paths)
    selected_id = _select_configuration(comparison_items)
    if not isinstance(selected_id, str):
        raise ValueError("development strategy selection did not produce a winner")
    configured_answer = loaded.get("answer_engine")
    if not isinstance(configured_answer, dict):
        raise ValueError("frozen answer configuration is missing")
    configured_id = configured_answer.get("configuration_id")
    if selected_id != configured_id or comparison.get("selected_configuration") != selected_id:
        raise ValueError("development strategy selection does not follow the frozen rule")
    selected = comparison_items[selected_id]
    selected_configuration = selected.get("configuration")
    expected_selected_configuration = {
        "strategy_id": configured_answer.get("strategy_id"),
        "id": configured_id,
        "support_threshold": configured_answer.get("support_threshold"),
        "partial_support_threshold": configured_answer.get("partial_support_threshold"),
        "contradiction_margin": configured_answer.get("contradiction_margin"),
    }
    if not isinstance(selected_configuration, dict) or any(
        selected_configuration.get(key) != value
        for key, value in expected_selected_configuration.items()
    ):
        raise ValueError("frozen answer configuration differs from the selected strategy")

    summary_path = root / DEVELOPMENT_ARTIFACT_BINDINGS[
        "development_runtime_summary_sha256"
    ]
    summary = _load(summary_path)
    if summary.get("schema_version") != (
        "evidencedesk-development-v3-frozen-runtime-summary-v1"
    ):
        raise ValueError("development runtime summary schema is invalid")
    if summary.get("engine_fingerprint") != loaded["engine_fingerprint_at_selection"]:
        raise ValueError("development runtime summary engine fingerprint mismatch")
    if summary.get("selected_configuration") != selected.get("configuration"):
        raise ValueError("development runtime summary selected configuration mismatch")
    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, dict):
        raise ValueError("development runtime summary aggregates are missing")
    core_reproduction: dict[str, object] = {}
    for partition, expected_relatives in FINAL_RUNTIME_PATHS.items():
        aggregate = aggregates.get(partition)
        paths = _artifact_paths(
            aggregate,
            root=root,
            expected_paths=expected_relatives,
        )
        evidence_paths.extend(expected_relatives)
        if _aggregate(paths) != aggregate:
            raise ValueError("development runtime aggregate does not recalculate")
        for path in paths:
            raw = _load(path)
            if raw.get("engine_fingerprint") != loaded["engine_fingerprint_at_selection"]:
                raise ValueError("development runtime artifact engine fingerprint mismatch")
            benchmark = raw.get("benchmark")
            if not isinstance(benchmark, dict) or benchmark.get(
                "configuration"
            ) != selected.get("configuration"):
                raise ValueError("development runtime artifact configuration mismatch")
        original_paths = _artifact_paths(selected[partition], root=root)
        frozen_core = [_core_decision_digest(_load(path)) for path in paths]
        original_core = [_core_decision_digest(_load(path)) for path in original_paths]
        core_reproduction[partition] = {
            "all_match": frozen_core == original_core,
            "frozen_runtime_core_decision_sha256": frozen_core,
            "original_core_decision_sha256": original_core,
        }
    if summary.get("core_decision_reproduction") != core_reproduction or not all(
        item["all_match"] for item in core_reproduction.values() if isinstance(item, dict)
    ):
        raise ValueError("development core decisions do not reproduce the selected run")

    selection_quality = aggregates["selection"]["quality"]
    expected_development_gate = _development_target_gate(selection_quality)
    if summary.get("development_target_gate") != expected_development_gate:
        raise ValueError("development target gate differs from recalculated metrics")

    recalculation = summary.get("recalculation")
    if not isinstance(recalculation, dict):
        raise ValueError("development recalculation provenance is missing")
    for partition in ("calibration", "selection"):
        raw_path = root / FINAL_RUNTIME_PATHS[partition][0]
        path_key = f"{partition}_r1_path"
        hash_key = f"{partition}_r1_sha256"
        recalculation_relative = recalculation.get(path_key)
        if not isinstance(recalculation_relative, str) or recalculation.get(
            hash_key
        ) != _sha256(
            root / recalculation_relative
        ):
            raise ValueError("development recalculation provenance hash mismatch")
        recalculation_path = root / recalculation_relative
        _validate_recalculation(raw_path, recalculation_path)
        evidence_paths.append(recalculation_relative)
    return list(dict.fromkeys(evidence_paths))


def _validate_v3_config_data(
    loaded: dict[str, Any],
    *,
    expected_engine_fingerprint: str | None = None,
) -> dict[str, Any]:
    if frozenset(loaded) != _V3_CONFIG_KEYS:
        raise ValueError("frozen v3 configuration has unsupported or missing top-level fields")
    try:
        validate_frozen_answer_config(loaded)
    except AnswerRuntimeConfigError as exc:
        raise ValueError(f"frozen v3 answer configuration is invalid: {exc}") from exc

    if loaded.get("parameters_version") != "grounded-local-v3.0-frozen-v7":
        raise ValueError("frozen v3 parameters version is not the registered v7 version")
    if loaded.get("frozen_at") != "2026-08-27":
        raise ValueError("frozen v3 date differs from the registered freeze date")
    if loaded.get("development_dataset_version") != "development-v3-2026.08.27":
        raise ValueError("frozen v3 development dataset version is invalid")
    if loaded.get("mode") != "grounded-local-v3":
        raise ValueError("frozen v3 mode must be grounded-local-v3")
    if loaded.get("retrieval_method") != "hybrid":
        raise ValueError("frozen v3 retrieval method must be hybrid")
    if loaded.get("answer_candidate_limit") != DEFAULT_RETRIEVAL_LIMIT:
        raise ValueError("frozen v3 answer candidate limit differs from retrieval code")
    if loaded.get("retrieval_metric_cutoff") != 5:
        raise ValueError("frozen v3 retrieval metric cutoff must be five")
    if loaded.get("rank_fusion") != {
        "algorithm": "reciprocal-rank-fusion",
        "rrf_k": RRF_K,
        "named_entity_document_boost": NAMED_ENTITY_DOCUMENT_BOOST,
    }:
        raise ValueError("frozen v3 rank fusion differs from retrieval code")

    embedding = loaded.get("embedding")
    if not isinstance(embedding, dict):
        raise ValueError("frozen v3 embedding configuration is missing")
    expected_embedding = {
        "model_id": (
            "paraphrase-multilingual-minilm-l12-v2-onnx-q@"
            "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
        ),
        "model_manifest_sha256": _sha256(
            Path(__file__).resolve().parents[1] / EMBEDDING_MANIFEST_PATH
        ),
        "license": "Apache-2.0",
        "parameter_count_approx": 118000000,
        "download_bytes": 266906689,
        "fastembed_version": "0.8.0",
        "onnxruntime_version": "1.29.0",
        "dimension": 384,
        "pooling": "mean",
        "provider": "CPUExecutionProvider",
        "normalize_l2": True,
        "batch_size": 32,
    }
    if embedding != expected_embedding:
        raise ValueError("frozen v3 embedding configuration differs from its manifest/runtime")

    validate_frozen_holdout_protocol(loaded.get("holdout_protocol"))
    expected_fingerprint = expected_engine_fingerprint or _engine_fingerprint()
    if loaded.get("engine_fingerprint_at_selection") != expected_fingerprint:
        raise ValueError("frozen v3 engine fingerprint differs from current engine sources")
    if loaded.get("evaluation") != FROZEN_EVALUATION_PROTOCOL:
        raise ValueError("frozen v3 evaluation protocol differs from the registered definitions")

    root = Path(__file__).resolve().parents[1]
    _validate_development_evidence(loaded, root)
    return loaded


def _validate_v3_freeze_config(
    config_path: Path,
    *,
    expected_engine_fingerprint: str | None = None,
) -> dict[str, Any]:
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen v3 configuration is not valid JSON") from exc
    if not isinstance(loaded, dict):
        raise ValueError("frozen v3 configuration must be an object")
    return _validate_v3_config_data(
        loaded,
        expected_engine_fingerprint=expected_engine_fingerprint,
    )


def _engine_fingerprint_at_commit(commit: str, *, root: Path | None = None) -> str:
    """Recompute the frozen engine fingerprint from immutable Git objects."""

    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("historical engine commit is not a full Git SHA")
    repository = (root or Path(__file__).resolve().parents[1]).resolve()
    git = shutil.which("git")
    if git is None:
        raise ValueError("git executable is required to verify a historical engine")
    digest = hashlib.sha256()
    for relative in ENGINE_FINGERPRINT_PATHS:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_historical_file_bytes(commit, relative, repository, git))
        digest.update(b"\0")
    return digest.hexdigest()


def _historical_file_bytes(
    commit: str,
    relative: str,
    repository: Path,
    git: str,
) -> bytes:
    source = subprocess.run(  # noqa: S603 - full SHA and fixed paths are validated
        [git, "show", f"{commit}:{relative}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if source.returncode != 0:
        raise ValueError(f"historical engine source is missing: {relative}")
    return source.stdout


def _validate_historical_v3_freeze_record(
    config_path: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    """Validate a consumed freeze without pretending HEAD still equals its engine."""

    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("historical v3 freeze record is not valid JSON") from exc
    if not isinstance(freeze, dict):
        raise ValueError("historical v3 freeze record must be an object")
    if freeze.get("schema_version") != "evidencedesk-engine-freeze-v3":
        raise ValueError("historical v3 freeze schema is invalid")
    if freeze.get("engine_paths") != list(ENGINE_FINGERPRINT_PATHS):
        raise ValueError("historical v3 engine path list differs from the evaluator")
    if freeze.get("config_sha256") != _sha256(config_path):
        raise ValueError("historical v3 config hash differs from the freeze record")
    engine_commit = freeze.get("engine_commit")
    if not isinstance(engine_commit, str):
        raise ValueError("historical v3 engine commit is missing")
    root = Path(__file__).resolve().parents[1]
    historical_fingerprint = _engine_fingerprint_at_commit(engine_commit, root=root)
    if freeze.get("engine_fingerprint") != historical_fingerprint:
        raise ValueError("historical v3 engine fingerprint differs from Git objects")
    git = shutil.which("git")
    if git is None:
        raise ValueError("git executable is required to verify a historical engine")
    model_manifest_sha256 = hashlib.sha256(
        _historical_file_bytes(engine_commit, EMBEDDING_MANIFEST_PATH, root, git)
    ).hexdigest()
    if freeze.get("model_manifest_sha256") != model_manifest_sha256:
        raise ValueError("historical v3 model manifest hash differs from Git objects")
    dependency_sha256 = {
        relative: hashlib.sha256(
            _historical_file_bytes(engine_commit, relative, root, git)
        ).hexdigest()
        for relative in DEPENDENCY_PATHS
    }
    if freeze.get("dependency_sha256") != dependency_sha256:
        raise ValueError("historical v3 dependency hashes differ from Git objects")
    return _validate_v3_freeze_config(
        config_path,
        expected_engine_fingerprint=historical_fingerprint,
    )


def _assert_paths_tracked_and_clean(root: Path, relative_paths: list[str]) -> None:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required to verify frozen inputs")
    unique_paths = list(dict.fromkeys(relative_paths))
    for relative in unique_paths:
        tracked = subprocess.run(  # noqa: S603 - repository-relative paths are fixed or validated
            [git, "cat-file", "-e", f"HEAD:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"freeze input is not tracked at HEAD: {relative}")
    status = subprocess.run(  # noqa: S603 - executable resolved from trusted PATH
        [git, "status", "--porcelain", "--untracked-files=all", "--", *unique_paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError(
            "engine, configuration, models, and artifacts must be committed and clean"
        )


def create_freeze_record(
    *,
    config_path: Path,
    model_manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"freeze record already exists: {output_path}")
    config = _validate_v3_freeze_config(config_path)
    root = Path(__file__).resolve().parents[1]
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required to create a freeze record")
    config_relative = config_path.resolve().relative_to(root).as_posix()
    model_manifest_relative = model_manifest_path.resolve().relative_to(root).as_posix()
    if model_manifest_relative != EMBEDDING_MANIFEST_PATH:
        raise ValueError("frozen v3 model manifest path is not the registered embedding manifest")
    if _sha256(model_manifest_path) != config["embedding"]["model_manifest_sha256"]:
        raise ValueError("frozen v3 model manifest differs from the configuration")
    tracked_paths = [
        *ENGINE_FINGERPRINT_PATHS,
        config_relative,
        model_manifest_relative,
        *_validate_development_evidence(config, root),
    ]
    _assert_paths_tracked_and_clean(root, tracked_paths)
    head = subprocess.run(  # noqa: S603 - executable resolved from trusted PATH
        [git, "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload: dict[str, object] = {
        "schema_version": "evidencedesk-engine-freeze-v3",
        "created_at": datetime.now(UTC).isoformat(),
        "engine_commit": head,
        "engine_fingerprint": _engine_fingerprint(),
        "config_sha256": _sha256(config_path),
        "model_manifest_sha256": _sha256(model_manifest_path),
        "dependency_sha256": {relative: _sha256(root / relative) for relative in DEPENDENCY_PATHS},
        "engine_paths": list(ENGINE_FINGERPRINT_PATHS),
        "development_artifact_sha256": {
            relative: config[key] for key, relative in DEVELOPMENT_ARTIFACT_BINDINGS.items()
        },
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
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("infra/models/paraphrase-multilingual-minilm-l12-v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FREEZE_PATH,
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
