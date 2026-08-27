from pathlib import Path
from typing import Any

import pytest

from evals.strategy_benchmark_v3 import (
    StrategyBenchmarkError,
    _core_decision_digest,
    _decision_digest,
    _repo_relative_output,
    _select_configuration,
    _validate_resumed_artifact,
    load_experiment_configurations,
)

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_uses_exactly_the_six_preregistered_configurations() -> None:
    configurations = load_experiment_configurations(
        ROOT / "evals/configs/answer-v3-experiment-plan.json"
    )

    assert [item["id"] for item in configurations] == [
        "deterministic-v3-a",
        "deterministic-v3-b",
        "multilingual-nli-v3-a",
        "multilingual-nli-v3-b",
        "qwen-json-v3-a",
        "qwen-json-v3-b",
    ]
    assert len({item["strategy_id"] for item in configurations}) == 3


def test_decision_stability_digest_ignores_timing_but_not_decisions() -> None:
    first: dict[str, Any] = {
        "cases": [{"id": "q1", "status": "answered", "answer": "A", "latency_ms": 2.0}],
        "extractions": {"doc": {"field": {"value": "A"}}},
        "latency_median_ms": 2.0,
        "runtime": {"peak_rss_bytes": 10},
    }
    second: dict[str, Any] = {
        **first,
        "cases": [{"id": "q1", "status": "answered", "answer": "A", "latency_ms": 9.0}],
        "latency_median_ms": 9.0,
        "runtime": {"peak_rss_bytes": 20},
    }

    assert _decision_digest(first) == _decision_digest(second)
    second["cases"][0]["answer"] = "B"
    assert _decision_digest(first) != _decision_digest(second)


def test_core_digest_ignores_candidate_instrumentation_only() -> None:
    first: dict[str, Any] = {
        "cases": [
            {
                "id": "q1",
                "status": "answered",
                "answer": "A",
                "candidate_assessments": [{"answerable": True}],
            }
        ],
        "extractions": {},
        "metric_counts": {},
    }
    second: dict[str, Any] = {
        **first,
        "cases": [
            {
                **first["cases"][0],
                "candidate_assessments": [{"answerable": False}],
            }
        ],
    }

    assert _core_decision_digest(first) == _core_decision_digest(second)
    assert _decision_digest(first) != _decision_digest(second)


def test_selection_tie_breaks_by_p95_then_rss_then_simplicity() -> None:
    quality: dict[str, float] = {
        "citation_case_accuracy": 0.9,
        "abstention_accuracy": 0.9,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "extraction_f1": 0.9,
        "error_rate": 0.0,
        "schema_error_rate": 0.0,
        "latency_p95_ms": 100.0,
    }
    comparison: dict[str, Any] = {
        "higher-rss": {
            "strategy_id": "deterministic-evidence-v3",
            "selection": {
                "stable_decisions": True,
                "quality": quality,
                "peak_rss_bytes": 900,
            },
        },
        "lower-rss": {
            "strategy_id": "multilingual-nli-v3",
            "selection": {
                "stable_decisions": True,
                "quality": quality,
                "peak_rss_bytes": 800,
            },
        },
    }

    assert _select_configuration(comparison) == "lower-rss"

    comparison["higher-rss"]["selection"]["peak_rss_bytes"] = 800
    assert _select_configuration(comparison) == "higher-rss"

    comparison["lower-rss"]["strategy_id"] = "deterministic-evidence-v3"
    with pytest.raises(StrategyBenchmarkError, match="tie-break is exhausted"):
        _select_configuration(comparison)


def test_resume_rejects_artifact_from_another_configuration() -> None:
    configuration = {
        "id": "deterministic-v3-a",
        "strategy_id": "deterministic-evidence-v3",
    }
    result = {
        "schema_version": "evaluation-result-v4",
        "manifest_sha256": "manifest",
        "corpus_sha256": "corpus",
        "engine_fingerprint": "engine",
        "cases": [],
        "extractions": {},
        "metric_counts": {},
        "benchmark": {
            "configuration": {**configuration, "id": "deterministic-v3-b"},
            "partition": "calibration",
            "repetition": 1,
            "decision_sha256": _decision_digest(
                {"cases": [], "extractions": {}, "metric_counts": {}}
            ),
        },
    }

    try:
        _validate_resumed_artifact(
            result,
            configuration=configuration,
            partition="calibration",
            repetition=1,
            manifest_sha256="manifest",
            corpus_sha256="corpus",
            engine_fingerprint="engine",
        )
    except Exception as exc:  # exact public error type is asserted without pytest dependency
        assert "configuration" in str(exc)
    else:
        raise AssertionError("resume accepted an artifact from another configuration")


def test_relative_output_path_is_reported_relative_to_repository() -> None:
    path = Path("artifacts/evaluations/development_v3/example.json")

    assert _repo_relative_output(path) == path
