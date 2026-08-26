"""Reproducible development-only comparison of EvidenceDesk retrieval methods."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidencedesk_api.providers import LocalSemanticEmbeddingProvider
from evidencedesk_api.retrieval import IntentEvidenceReranker, RetrievalMethod

from evals.runner import _engine_fingerprint, evaluate_manifest


def _count(result: dict[str, object], key: str) -> int:
    counts = result["metric_counts"]
    if not isinstance(counts, dict):
        raise ValueError("metric_counts must be an object")
    return int(counts[key])


def _metric(result: dict[str, object], key: str) -> float:
    value = result[key]
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _quality_key(result: dict[str, object], simplicity: int) -> tuple[int, int, int, int, int]:
    return (
        _count(result, "answerable_correct"),
        _count(result, "citation_correct"),
        _count(result, "abstention_correct"),
        _count(result, "retrieval_hits_at_5"),
        -simplicity,
    )


def select_method(results: dict[str, dict[str, object]]) -> tuple[str, dict[str, object]]:
    """Apply the pre-registered parsimony and reranker-retention rule."""

    base_order = {"lexical": 0, "dense": 1, "hybrid": 2}
    base = max(
        (results[name] for name in base_order),
        key=lambda item: _quality_key(item, base_order[str(item["retrieval_method"])]),
    )
    selected = str(base["retrieval_method"])
    hybrid = results["hybrid"]
    reranked = results["hybrid_rerank"]
    additional = _count(reranked, "answerable_correct") - _count(hybrid, "answerable_correct")
    no_regression = (
        _count(reranked, "citation_correct") >= _count(hybrid, "citation_correct")
        and _count(reranked, "abstention_correct") >= _count(hybrid, "abstention_correct")
        and _count(reranked, "retrieval_hits_at_5") >= _count(hybrid, "retrieval_hits_at_5")
        and _metric(reranked, "extraction_f1") >= _metric(hybrid, "extraction_f1")
        and _metric(reranked, "error_rate") <= _metric(hybrid, "error_rate")
    )
    reranker_retained = additional >= 1 and no_regression
    if reranker_retained and _quality_key(reranked, 3) > _quality_key(
        results[selected], base_order[selected]
    ):
        selected = "hybrid_rerank"
    return selected, {
        "rule": (
            "maximize answerable_correct, then citation_correct, abstention_correct, and "
            "retrieval_hits_at_5; "
            "prefer lexical < dense < hybrid on equality; retain reranker only for at least "
            "one additional correct answer with no citation, abstention, extraction, or "
            "error regression"
        ),
        "reranker_retained": reranker_retained,
        "reranker_additional_answerable_correct": additional,
        "reranker_no_regression": no_regression,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required for benchmark provenance")
    completed = subprocess.run(  # noqa: S603 - executable resolved from the trusted PATH
        [git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _hardware() -> dict[str, object]:
    cpu = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = next(
            (
                line.split(":", maxsplit=1)[1].strip()
                for line in cpuinfo.read_text(encoding="utf-8").splitlines()
                if line.startswith("model name")
            ),
            None,
        )
        cpu = match or cpu
    memory_kib: int | None = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        first = meminfo.read_text(encoding="utf-8").splitlines()[0].split()
        memory_kib = int(first[1])
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "logical_cpu_count": __import__("os").cpu_count(),
        "memory_total_bytes": memory_kib * 1024 if memory_kib is not None else None,
        "reference_device": "CPUExecutionProvider",
    }


def run_benchmark(
    *,
    manifest_path: Path,
    corpus_path: Path,
    model_manifest_path: Path,
    model_path: Path,
    output_path: Path,
    selected_output_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"benchmark output already exists: {output_path}")
    if selected_output_path is not None and selected_output_path.exists():
        raise FileExistsError(f"selected output already exists: {selected_output_path}")
    provider = LocalSemanticEmbeddingProvider(
        manifest_path=model_manifest_path,
        model_path=model_path,
    )
    reranker = IntentEvidenceReranker()
    hardware = _hardware()
    runtime = {
        **hardware,
        "fastembed_version": importlib.metadata.version("fastembed"),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "model_manifest_sha256": _sha256(model_manifest_path),
        "external_api_cost_usd": 0.0,
    }
    results: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for method in RetrievalMethod:
        results[method.value] = evaluate_manifest(
            manifest_path,
            corpus_path,
            split="development",
            provider=provider,
            method=method,
            reranker=reranker if method is RetrievalMethod.HYBRID_RERANK else None,
            runtime_metadata=runtime,
        )
    selected, decision = select_method(results)
    if selected_output_path is not None:
        selected_output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_output_path.write_text(
            json.dumps(results[selected], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    payload: dict[str, Any] = {
        "schema_version": "development-comparison-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "split": "development",
        "seed": json.loads(manifest_path.read_text(encoding="utf-8"))["seed"],
        "engine_commit_at_run": _git_head(),
        "engine_fingerprint": _engine_fingerprint(),
        "manifest_sha256": _sha256(manifest_path),
        "corpus_sha256": _sha256(corpus_path),
        "model_manifest_sha256": _sha256(model_manifest_path),
        "embedding_model_id": provider.model_id,
        "reranker_model_id": reranker.model_id,
        "methods": [method.value for method in RetrievalMethod],
        "selected_method": selected,
        "selection_decision": decision,
        "selected_artifact_sha256": (
            _sha256(selected_output_path) if selected_output_path is not None else None
        ),
        "hardware": hardware,
        "total_wall_time_ms": round((time.perf_counter() - started) * 1_000, 3),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "external_api_cost_usd": 0.0,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        manifest_path=args.manifest,
        corpus_path=args.corpus,
        model_manifest_path=args.model_manifest,
        model_path=args.model_path,
        output_path=args.output,
        selected_output_path=args.selected_output,
    )
    print(args.output)
    print(f"selected_method={payload['selected_method']}")


if __name__ == "__main__":
    main()
