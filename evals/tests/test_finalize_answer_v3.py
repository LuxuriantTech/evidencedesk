"""Tests for the fail-closed final v3 development runtime runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals import finalize_answer_v3


def _comparison() -> dict[str, Any]:
    artifact_root = "artifacts/evaluations/development_v3/strategy_v3"
    configuration = {
        "id": "deterministic-v3-a",
        "strategy_id": "deterministic-evidence-v3",
        "support_threshold": 0.48,
        "partial_support_threshold": 0.38,
        "contradiction_margin": 0.08,
    }
    selection = {
        "artifacts": [
            {
                "path": f"{artifact_root}/deterministic-v3-a-selection-r{repetition}.json"
            }
            for repetition in (1, 2, 3)
        ],
        "stable_decisions": True,
        "quality": {
            "citation_case_accuracy": 0.95,
            "abstention_accuracy": 0.9,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "extraction_f1": 0.93,
            "error_rate": 0.0,
            "schema_error_rate": 0.0,
            "latency_p95_ms": 10.0,
        },
        "peak_rss_bytes": 100,
    }
    return {
        "schema_version": "evidencedesk-strategy-comparison-v3",
        "engine_fingerprint": "engine-fingerprint",
        "experiment_plan_sha256": finalize_answer_v3._sha256(
            finalize_answer_v3.PLAN_PATH
        ),
        "corpus_sha256": finalize_answer_v3._sha256(finalize_answer_v3.CORPUS_PATH),
        "partition_manifest_sha256": {
            partition: finalize_answer_v3._sha256(path)
            for partition, path in finalize_answer_v3.PARTITION_MANIFESTS.items()
        },
        "repetitions": 3,
        "comparison": {
            "deterministic-v3-a": {
                "strategy_id": "deterministic-evidence-v3",
                "configuration": configuration,
                "calibration": {
                    "artifacts": [
                        {
                            "path": (
                                f"{artifact_root}/deterministic-v3-a-calibration-r{repetition}.json"
                            )
                        }
                        for repetition in (1, 2, 3)
                    ]
                },
                "selection": selection,
            }
        },
        "selected_configuration": "deterministic-v3-a",
    }


def _raw(configuration: dict[str, Any], partition: str, repetition: int) -> dict[str, Any]:
    return {
        "schema_version": "evaluation-result-v4",
        "engine_fingerprint": "engine-fingerprint",
        "benchmark": {
            "configuration": configuration,
            "partition": partition,
            "repetition": repetition,
            "decision_sha256": "same-decision",
            "peak_rss_bytes": 100,
            "total_wall_ms": 11.0,
            "model_load_ms": 1.0,
        },
        "metric_counts": {"errors": 0, "schema_errors": 0},
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "citation_case_accuracy": 1.0,
        "abstention_accuracy": 1.0,
        "extraction_precision": 1.0,
        "extraction_recall": 1.0,
        "extraction_f1": 1.0,
        "retrieval_recall_at_5": 1.0,
        "retrieval_mrr_at_5": 1.0,
        "error_rate": 0.0,
        "schema_error_rate": 0.0,
        "latency_median_ms": 1.0,
        "latency_p95_ms": 1.0,
    }


def test_finalizer_runs_six_named_runs_and_writes_recalculations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(_comparison()), encoding="utf-8")
    calls: list[tuple[str, int]] = []

    def fake_run_once(
        configuration: dict[str, Any], partition: str, repetition: int
    ) -> dict[str, Any]:
        calls.append((partition, repetition))
        return _raw(configuration, partition, repetition)

    monkeypatch.setattr(finalize_answer_v3, "run_once", fake_run_once)
    monkeypatch.setattr(finalize_answer_v3, "_engine_fingerprint", lambda: "engine-fingerprint")
    monkeypatch.setattr(
        finalize_answer_v3,
        "_validate_strategy_comparison",
        lambda *_args, **_kwargs: ({}, []),
    )
    monkeypatch.setattr(
        finalize_answer_v3,
        "_validate_comparison_raw",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(finalize_answer_v3, "_core_decision_digest", lambda raw: "core")
    monkeypatch.setattr(
        finalize_answer_v3,
        "_runtime_aggregate",
        lambda paths: {
            "artifacts": [{"path": str(path), "sha256": "x"} for path in paths],
            "quality": {
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "citation_case_accuracy": 1.0,
                "abstention_accuracy": 1.0,
                "extraction_f1": 1.0,
                "error_rate": 0.0,
                "schema_error_rate": 0.0,
            },
        },
    )
    monkeypatch.setattr(
        finalize_answer_v3,
        "recalculate_metrics",
        lambda raw: {"verdict": "PASS", "metric_counts": raw["metric_counts"]},
    )
    summary = finalize_answer_v3.finalize(
        comparison_path=comparison_path,
        output_dir=tmp_path / "runtime",
    )

    assert calls == [
        (partition, repetition)
        for partition in ("calibration", "selection")
        for repetition in (1, 2, 3)
    ]
    assert summary["selected_configuration"]["id"] == "deterministic-v3-a"
    assert (tmp_path / "runtime" / "calibration-r1.json").exists()
    assert (tmp_path / "runtime" / "calibration-r1-recalculated.json").exists()
    assert (tmp_path / "runtime" / "selection-r3.json").exists()
    assert summary["engine_fingerprint"] == "engine-fingerprint"


def test_finalizer_refuses_preexisting_output_dir(tmp_path: Path) -> None:
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(_comparison()), encoding="utf-8")
    output_dir = tmp_path / "runtime"
    output_dir.mkdir()

    with pytest.raises(finalize_answer_v3.FinalizationError, match="refusing to overwrite"):
        finalize_answer_v3.finalize(comparison_path=comparison_path, output_dir=output_dir)


def test_runtime_aggregate_resolves_repository_relative_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []

    def fake_aggregate(paths: list[Path]) -> dict[str, Any]:
        observed.extend(paths)
        return {"artifacts": []}

    monkeypatch.setattr(finalize_answer_v3, "_aggregate", fake_aggregate)
    relative = Path("artifacts/evaluations/development_v3/runtime/example.json")

    assert finalize_answer_v3._runtime_aggregate([relative]) == {"artifacts": []}
    assert observed == [(finalize_answer_v3.ROOT / relative).resolve()]


def test_finalizer_rejects_comparison_selected_outside_preregistered_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison = _comparison()
    comparison["selected_configuration"] = "not-the-maximin-choice"
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    monkeypatch.setattr(
        finalize_answer_v3,
        "_engine_fingerprint",
        lambda: "engine-fingerprint",
    )
    monkeypatch.setattr(
        finalize_answer_v3,
        "_validate_strategy_comparison",
        lambda *_args, **_kwargs: ({}, []),
    )

    with pytest.raises(finalize_answer_v3.FinalizationError, match="selection"):
        finalize_answer_v3.finalize(
            comparison_path=comparison_path,
            output_dir=tmp_path / "runtime",
        )


def test_finalizer_rejects_comparison_from_another_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison = _comparison()
    comparison["engine_fingerprint"] = "another-engine"
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    monkeypatch.setattr(
        finalize_answer_v3,
        "_engine_fingerprint",
        lambda: "engine-fingerprint",
    )

    with pytest.raises(finalize_answer_v3.FinalizationError, match="engine_fingerprint"):
        finalize_answer_v3.finalize(
            comparison_path=comparison_path,
            output_dir=tmp_path / "runtime",
        )
