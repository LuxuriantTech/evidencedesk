"""Bounded, reproducible development-v3 answer strategy comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from evidencedesk_api.answering import DecisionConfig, DeterministicGroundedAnswerProvider
from evidencedesk_api.extraction import SupplierExtraction, extract_supplier_fields_v3
from evidencedesk_api.nli_answering import (
    LocalOnnxNliScorer,
    NliDecisionConfig,
    NliGroundedAnswerProvider,
)
from evidencedesk_api.providers import LocalSemanticEmbeddingProvider
from evidencedesk_api.qwen_answering import LlamaCppBackend, QwenAnswerProvider, QwenRunConfig
from evidencedesk_api.retrieval import EvidenceChunk

from evals.runner import AnswerProviderLike, _engine_fingerprint, evaluate_manifest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "evals/configs/answer-v3-experiment-plan.json"
CORPUS_PATH = ROOT / "datasets/development_v3/corpus_manifest.json"
PARTITION_MANIFESTS = {
    "calibration": ROOT / "datasets/development_v3/evaluation_calibration.json",
    "selection": ROOT / "datasets/development_v3/evaluation_selection.json",
}
OUTPUT_DIR = ROOT / "artifacts/evaluations/development_v3/strategy_v3"

QUALITY_METRICS = (
    "citation_precision",
    "citation_recall",
    "citation_case_accuracy",
    "abstention_accuracy",
    "extraction_precision",
    "extraction_recall",
    "extraction_f1",
    "retrieval_recall_at_5",
    "retrieval_mrr_at_5",
    "error_rate",
    "schema_error_rate",
    "latency_median_ms",
    "latency_p95_ms",
)


class StrategyBenchmarkError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StrategyBenchmarkError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_experiment_configurations(plan_path: Path = PLAN_PATH) -> list[dict[str, Any]]:
    plan = _load(plan_path)
    budget = plan.get("experiment_budget")
    strategies = plan.get("strategies")
    if not isinstance(budget, dict) or not isinstance(strategies, list):
        raise StrategyBenchmarkError("invalid experiment plan")
    maximum_strategies = int(budget.get("maximum_response_strategies", 0))
    maximum_per_strategy = int(budget.get("maximum_configurations_per_strategy", 0))
    if len(strategies) > maximum_strategies or maximum_strategies != 3:
        raise StrategyBenchmarkError("experiment plan exceeds the strategy budget")
    configurations: list[dict[str, Any]] = []
    for strategy in strategies:
        if not isinstance(strategy, dict) or not isinstance(strategy.get("configurations"), list):
            raise StrategyBenchmarkError("invalid strategy configuration")
        declared = strategy["configurations"]
        if len(declared) > maximum_per_strategy or maximum_per_strategy != 2:
            raise StrategyBenchmarkError("experiment plan exceeds the configuration budget")
        for raw in declared:
            if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
                raise StrategyBenchmarkError("invalid configuration entry")
            configurations.append(
                {
                    **raw,
                    "strategy_id": strategy["id"],
                    "strategy_license": strategy["license"],
                    "strategy_model": strategy.get("model"),
                }
            )
    if len(configurations) != 6 or len({item["id"] for item in configurations}) != 6:
        raise StrategyBenchmarkError("the frozen plan must contain exactly six unique configs")
    return configurations


def _decision_digest_for(result: dict[str, Any], *, include_candidates: bool) -> str:
    case_fields = [
        "id",
        "kind",
        "status",
        "answer",
        "answerable",
        "supporting_document",
        "supporting_page",
        "supporting_excerpt",
        "ambiguity_reason",
        "extracted_fields",
        "citations",
        "error_type",
    ]
    if include_candidates:
        case_fields.append("candidate_assessments")
    decisions = [
        {key: case.get(key) for key in case_fields if key in case}
        for case in result.get("cases", [])
        if isinstance(case, dict)
    ]
    payload = {
        "cases": decisions,
        "extractions": result.get("extractions", {}),
        "metric_counts": result.get("metric_counts", {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decision_digest(result: dict[str, Any]) -> str:
    return _decision_digest_for(result, include_candidates=True)


def _core_decision_digest(result: dict[str, Any]) -> str:
    """Compare answer/extraction decisions across schema-only instrumentation changes."""

    return _decision_digest_for(result, include_candidates=False)


def _embedding_provider() -> LocalSemanticEmbeddingProvider:
    return LocalSemanticEmbeddingProvider(
        manifest_path=ROOT / "infra/models/paraphrase-multilingual-minilm-l12-v2.json",
        model_path=ROOT / "models/paraphrase-multilingual-minilm-l12-v2",
    )


def _provider_for(
    configuration: dict[str, Any], embedding: LocalSemanticEmbeddingProvider
) -> AnswerProviderLike:
    strategy = configuration["strategy_id"]
    if strategy == "deterministic-evidence-v3":
        return cast(
            AnswerProviderLike,
            DeterministicGroundedAnswerProvider(
                embeddings=embedding,
                config=DecisionConfig(
                    support_threshold=float(configuration["support_threshold"]),
                    partial_support_threshold=float(
                        configuration["partial_support_threshold"]
                    ),
                    contradiction_margin=float(configuration["contradiction_margin"]),
                ),
            ),
        )
    if strategy == "multilingual-nli-v3":
        base = DeterministicGroundedAnswerProvider(
            embeddings=embedding,
            config=DecisionConfig(0.48, 0.38, 0.08),
        )
        scorer = LocalOnnxNliScorer(
            manifest_path=ROOT / "infra/models/mdeberta-v3-base-mnli-xnli.json",
            model_path=ROOT / "models/mdeberta-v3-base-mnli-xnli",
        )
        return cast(
            AnswerProviderLike,
            NliGroundedAnswerProvider(
                base=base,
                scorer=scorer,
                config=NliDecisionConfig(
                    entailment_threshold=float(configuration["entailment_threshold"]),
                    contradiction_threshold=float(configuration["contradiction_threshold"]),
                ),
                mode=str(configuration["id"]),
            ),
        )
    if strategy == "qwen-constrained-json-v3":
        backend = LlamaCppBackend(
            manifest_path=ROOT / "infra/models/qwen2.5-0.5b-instruct-q4-k-m.json",
            model_path=ROOT / "models/qwen2.5-0.5b-instruct-gguf",
            n_threads=max(1, min(8, os.cpu_count() or 1)),
        )
        return cast(
            AnswerProviderLike,
            QwenAnswerProvider(
                backend,
                config=QwenRunConfig(
                    seed=int(configuration["seed"]),
                    max_output_tokens=int(configuration["max_output_tokens"]),
                    temperature=float(configuration["temperature"]),
                    context_passages=int(configuration["context_passages"]),
                ),
            ),
        )
    raise StrategyBenchmarkError(f"unknown strategy: {strategy}")


def _total_ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def run_once(
    configuration: dict[str, Any], partition: str, repetition: int
) -> dict[str, Any]:
    if partition not in PARTITION_MANIFESTS or repetition not in {1, 2, 3}:
        raise StrategyBenchmarkError("invalid bounded run request")
    wall_started = time.perf_counter()
    load_started = time.perf_counter()
    embedding = _embedding_provider()
    answer_provider = _provider_for(configuration, embedding)
    model_load_ms = (time.perf_counter() - load_started) * 1_000

    def extraction_provider(chunks: list[EvidenceChunk]) -> SupplierExtraction:
        return extract_supplier_fields_v3(chunks, embeddings=embedding)

    result = evaluate_manifest(
        PARTITION_MANIFESTS[partition],
        CORPUS_PATH,
        split="development",
        provider=embedding,
        answer_provider=answer_provider,
        extraction_provider=extraction_provider,
    )
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    result["benchmark"] = {
        "configuration": configuration,
        "partition": partition,
        "repetition": repetition,
        "decision_sha256": _decision_digest(result),
        "model_load_ms": round(model_load_ms, 3),
        "total_wall_ms": round((time.perf_counter() - wall_started) * 1_000, 3),
        "peak_rss_bytes": peak_rss_bytes,
        "hardware": {
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "total_ram_bytes": _total_ram_bytes(),
            "runtime": platform.python_version(),
        },
    }
    result["runtime"] = {
        **result.get("runtime", {}),
        **result["benchmark"],
    }
    return result


def _run_path(config_id: str, partition: str, repetition: int) -> Path:
    return OUTPUT_DIR / f"{config_id}-{partition}-r{repetition}.json"


def _repo_relative_output(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def _single(config_id: str, partition: str, repetition: int, output: Path) -> None:
    configs = {item["id"]: item for item in load_experiment_configurations()}
    if config_id not in configs:
        raise StrategyBenchmarkError(f"configuration is not pre-registered: {config_id}")
    if output.exists():
        raise StrategyBenchmarkError(f"refusing to overwrite evaluation artifact: {output}")
    result = run_once(configs[config_id], partition, repetition)
    _write_json(output, result)
    print(
        json.dumps(
            {
                "artifact": str(_repo_relative_output(output)),
                "config": config_id,
                "partition": partition,
                "repetition": repetition,
                "quality": {key: result[key] for key in QUALITY_METRICS},
                "peak_rss_bytes": result["benchmark"]["peak_rss_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _aggregate(paths: list[Path]) -> dict[str, Any]:
    runs = [_load(path) for path in paths]
    digests = [str(run["benchmark"]["decision_sha256"]) for run in runs]
    return {
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for path in paths
        ],
        "stable_decisions": len(set(digests)) == 1,
        "decision_sha256": digests,
        "quality": {
            metric: round(statistics.median(float(run[metric]) for run in runs), 6)
            for metric in QUALITY_METRICS
        },
        "peak_rss_bytes": max(int(run["benchmark"]["peak_rss_bytes"]) for run in runs),
        "total_wall_ms": round(sum(float(run["benchmark"]["total_wall_ms"]) for run in runs), 3),
        "model_load_ms": round(
            statistics.median(float(run["benchmark"]["model_load_ms"]) for run in runs),
            3,
        ),
        "metric_counts": [run["metric_counts"] for run in runs],
    }


def _execute_if_needed(
    configuration: dict[str, Any], partition: str, repetition: int, *, resume: bool
) -> Path:
    config_id = str(configuration["id"])
    output = _run_path(config_id, partition, repetition)
    if output.exists():
        if resume:
            _validate_resumed_artifact(
                _load(output),
                configuration=configuration,
                partition=partition,
                repetition=repetition,
                manifest_sha256=_sha256(PARTITION_MANIFESTS[partition]),
                corpus_sha256=_sha256(CORPUS_PATH),
                engine_fingerprint=_engine_fingerprint(),
            )
            return output
        raise StrategyBenchmarkError(f"artifact already exists: {output}")
    subprocess.run(  # noqa: S603 - executable and bounded arguments are locally controlled
        [
            sys.executable,
            "-m",
            "evals.strategy_benchmark_v3",
            "--single",
            "--config",
            config_id,
            "--partition",
            partition,
            "--repetition",
            str(repetition),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    return output


def _validate_resumed_artifact(
    result: dict[str, Any],
    *,
    configuration: dict[str, Any],
    partition: str,
    repetition: int,
    manifest_sha256: str,
    corpus_sha256: str,
    engine_fingerprint: str,
) -> None:
    """Reject stale or cross-configuration artifacts before a resumed suite."""

    benchmark = result.get("benchmark")
    if result.get("schema_version") != "evaluation-result-v4" or not isinstance(
        benchmark, dict
    ):
        raise StrategyBenchmarkError("resumed artifact has an invalid schema")
    expected = {
        "configuration": configuration,
        "partition": partition,
        "repetition": repetition,
    }
    for key, value in expected.items():
        if benchmark.get(key) != value:
            raise StrategyBenchmarkError(f"resumed artifact {key} does not match the run")
    provenance = {
        "manifest_sha256": manifest_sha256,
        "corpus_sha256": corpus_sha256,
        "engine_fingerprint": engine_fingerprint,
    }
    for key, value in provenance.items():
        if result.get(key) != value:
            raise StrategyBenchmarkError(f"resumed artifact {key} does not match")
    if benchmark.get("decision_sha256") != _decision_digest(result):
        raise StrategyBenchmarkError("resumed artifact decision digest does not match")


def _select_configuration(comparison: dict[str, Any]) -> str | None:
    eligible: list[tuple[str, float, float, int, int]] = []
    for config_id, item in comparison.items():
        selection = item.get("selection")
        if not isinstance(selection, dict) or not selection["stable_decisions"]:
            continue
        quality = selection["quality"]
        if quality["error_rate"] != 0.0 or quality["schema_error_rate"] != 0.0:
            continue
        floor = min(
            quality["citation_case_accuracy"],
            quality["abstention_accuracy"],
            quality["citation_precision"],
            quality["citation_recall"],
            quality["extraction_f1"],
        )
        complexity = {
            "deterministic-evidence-v3": 0,
            "multilingual-nli-v3": 1,
            "qwen-constrained-json-v3": 2,
        }[item["strategy_id"]]
        eligible.append(
            (
                config_id,
                floor,
                float(quality["latency_p95_ms"]),
                int(selection["peak_rss_bytes"]),
                complexity,
            )
        )
    if not eligible:
        return None
    ranked = sorted(eligible, key=lambda item: (-item[1], item[2], item[3], item[4]))
    best_key = (-ranked[0][1], ranked[0][2], ranked[0][3], ranked[0][4])
    if sum(
        (-item[1], item[2], item[3], item[4]) == best_key for item in ranked
    ) != 1:
        raise StrategyBenchmarkError(
            "the pre-registered tie-break is exhausted; no configuration may be selected"
        )
    return ranked[0][0]


def run_suite(*, resume: bool) -> dict[str, Any]:
    configurations = load_experiment_configurations()
    comparison: dict[str, Any] = {}
    for configuration in configurations:
        config_id = str(configuration["id"])
        calibration_paths = [
            _execute_if_needed(configuration, "calibration", repetition, resume=resume)
            for repetition in (1, 2, 3)
        ]
        calibration = _aggregate(calibration_paths)
        comparison[config_id] = {
            "strategy_id": configuration["strategy_id"],
            "configuration": configuration,
            "calibration": calibration,
        }
        calibration_counts = calibration["metric_counts"]
        calibration_rejected = (
            not calibration["stable_decisions"]
            or any(int(item["errors"]) > 0 for item in calibration_counts)
            or any(int(item["schema_errors"]) > 0 for item in calibration_counts)
        )
        comparison[config_id]["calibration_gate"] = (
            "REJECT" if calibration_rejected else "PASS"
        )
        if calibration_rejected:
            continue
        selection_paths = [
            _execute_if_needed(configuration, "selection", repetition, resume=resume)
            for repetition in (1, 2, 3)
        ]
        comparison[config_id]["selection"] = _aggregate(selection_paths)

    selected = _select_configuration(comparison)
    summary = {
        "schema_version": "evidencedesk-strategy-comparison-v3",
        "created_at": datetime.now(UTC).isoformat(),
        "engine_fingerprint": _engine_fingerprint(),
        "experiment_plan_sha256": _sha256(PLAN_PATH),
        "corpus_sha256": _sha256(CORPUS_PATH),
        "partition_manifest_sha256": {
            key: _sha256(value) for key, value in PARTITION_MANIFESTS.items()
        },
        "repetitions": 3,
        "comparison": comparison,
        "selected_configuration": selected,
        "selection_rule": _load(PLAN_PATH)["final_selection_rule"],
    }
    _write_json(OUTPUT_DIR / "comparison.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--partition", choices=tuple(PARTITION_MANIFESTS))
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.single:
        if not args.config or not args.partition or not args.repetition or not args.output:
            parser.error("--single requires config, partition, repetition, and output")
        _single(args.config, args.partition, args.repetition, args.output)
        return
    if args.suite:
        summary = run_suite(resume=args.resume)
        print(json.dumps({"selected_configuration": summary["selected_configuration"]}))
        return
    parser.error("choose --single or --suite")


if __name__ == "__main__":
    main()
