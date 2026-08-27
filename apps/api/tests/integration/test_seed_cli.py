import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from evidencedesk_api.config import Settings
from evidencedesk_api.db import build_engine, build_session_factory
from evidencedesk_api.models import EvaluationRun
from evidencedesk_api.seed_cli import (
    _assert_seed_digest_allowed,
    _build_seed_embeddings,
    _load_evaluation_artifacts,
    _seed_evaluation_artifacts,
)
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine

DATABASE_URL = (
    "postgresql+asyncpg://evidencedesk:evidencedesk-local-only@127.0.0.1:55432/evidencedesk"
)


def test_demo_seed_imports_measured_evaluation_artifacts_idempotently(tmp_path: Path) -> None:
    artifact = {
        "schema_version": "evaluation-result-v1",
        "created_at": "2026-08-26T20:15:00+00:00",
        "dataset_version": "synthetic-test-v1",
        "parameters_version": "frozen-test-v1",
        "split": "holdout",
        "mode": "extractive-local",
        "case_count": 19,
        "citation_precision": 0.7,
        "extraction_f1": 0.941176,
        "abstention_accuracy": 0.8,
        "latency_p95_ms": 1.976,
        "error_rate": 0.0,
        "estimated_cost_usd": 0.0,
        "verdict": "FAIL",
        "cases": [{"id": "private-detail-not-persisted"}],
        "extractions": {"private-detail": {}},
    }
    (tmp_path / "result.json").write_text(json.dumps(artifact), encoding="utf-8")
    (tmp_path / "recalculated.json").write_text(
        json.dumps(
            {
                "schema_version": "evaluation-recalculation-v2",
                "raw_artifact_sha256": "0" * 64,
                "verdict": "PASS",
            }
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        engine = build_engine_from_url()
        factory = build_session_factory(engine)
        try:
            async with engine.begin() as connection:
                await connection.execute(text("TRUNCATE evaluation_runs"))
            await _seed_evaluation_artifacts(factory, tmp_path)
            await _seed_evaluation_artifacts(factory, tmp_path)
            async with factory() as session:
                count = await session.scalar(select(func.count(EvaluationRun.id)))
                run = await session.scalar(select(EvaluationRun))
                assert count == 1
                assert run is not None
                assert run.verdict == "FAIL"
                assert run.created_at.isoformat() == "2026-08-26T20:15:00+00:00"
                assert run.metrics["citation_precision"] == 0.7
                assert "cases" not in run.metrics
                assert "extractions" not in run.metrics
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_seed_loader_accepts_only_public_summary_schema(tmp_path: Path) -> None:
    for name, schema in (
        ("development.json", "evaluation-result-v1"),
        ("holdout-v4-raw.json", "holdout-v4-raw-result-v2"),
        ("holdout-v4-recalculated.json", "evaluation-recalculation-v2"),
    ):
        (tmp_path / name).write_text(
            json.dumps({"schema_version": schema}),
            encoding="utf-8",
        )

    loaded = _load_evaluation_artifacts(tmp_path)

    assert [path.name for path, _result in loaded] == ["development.json"]


def test_seed_loader_keeps_latest_artifact_for_the_same_run_identity(tmp_path: Path) -> None:
    identity = {
        "schema_version": "evaluation-result-v1",
        "dataset_version": "development-v2",
        "parameters_version": "semantic-v2",
        "split": "development",
    }
    (tmp_path / "development-final.json").write_text(
        json.dumps(
            {
                **identity,
                "created_at": "2026-08-26T22:15:10+00:00",
                "latency_p95_ms": 46.786,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "development-v6.json").write_text(
        json.dumps(
            {
                **identity,
                "created_at": "2026-08-26T22:41:01+00:00",
                "latency_p95_ms": 50.054,
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_evaluation_artifacts(tmp_path)

    assert len(loaded) == 1
    assert loaded[0][0].name == "development-v6.json"
    assert loaded[0][1]["latency_p95_ms"] == 50.054


def test_v7_summary_is_seedable_and_preserves_the_raw_artifact_digest() -> None:
    artifacts_dir = Path("artifacts/evaluations")
    raw_path = artifacts_dir / "holdout_v7/raw.json"
    recalculated_path = artifacts_dir / "holdout_v7/recalculated.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    recalculated = json.loads(recalculated_path.read_text(encoding="utf-8"))

    loaded = _load_evaluation_artifacts(artifacts_dir)
    summaries = [
        result
        for path, result in loaded
        if path.name == "holdout-v7-summary.json"
    ]

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["dataset_version"] == "blind-holdout-v7-2026.08.27"
    assert summary["parameters_version"] == "grounded-local-v3.0-frozen-v7"
    assert summary["verdict"] == "FAIL"
    assert summary["source_raw_path"] == str(raw_path)
    assert summary["source_raw_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert summary["source_recalculated_path"] == str(recalculated_path)
    assert summary["source_recalculated_sha256"] == hashlib.sha256(
        recalculated_path.read_bytes()
    ).hexdigest()
    assert recalculated["raw_artifact_sha256"] == summary["source_raw_sha256"]

    raw_fields = {
        "created_at",
        "dataset_version",
        "parameters_version",
        "seed",
        "split",
        "mode",
        "retrieval_method",
        "case_count",
        "citation_precision",
        "citation_recall",
        "citation_case_accuracy",
        "abstention_accuracy",
        "extraction_precision",
        "extraction_recall",
        "extraction_f1",
        "retrieval_recall_at_5",
        "retrieval_mrr_at_5",
        "latency_median_ms",
        "latency_p95_ms",
        "indexing_ms",
        "total_wall_time_ms",
        "peak_rss_bytes",
        "error_rate",
        "schema_error_rate",
        "estimated_cost_usd",
        "verdict",
    }
    assert {field: summary[field] for field in raw_fields} == {
        field: raw[field] for field in raw_fields
    }
    recalculated_metrics = {
        "citation_precision",
        "citation_recall",
        "citation_case_accuracy",
        "abstention_accuracy",
        "extraction_precision",
        "extraction_recall",
        "extraction_f1",
        "retrieval_recall_at_5",
        "retrieval_mrr_at_5",
        "latency_median_ms",
        "latency_p95_ms",
        "error_rate",
        "schema_error_rate",
        "verdict",
    }
    assert {field: summary[field] for field in recalculated_metrics} == {
        field: recalculated[field] for field in recalculated_metrics
    }
    assert summary["metric_counts"] == raw["metric_counts"]
    assert summary["recalculated_metric_counts"] == recalculated["metric_counts"]


def build_engine_from_url() -> AsyncEngine:
    settings = Settings(
        database_url=DATABASE_URL,
        jwt_secret="seed-test-secret-at-least-32-characters",
        demo_admin_password="not-used-admin",
        demo_analyst_password="not-used-analyst",
        demo_reader_password="not-used-reader",
    )
    return build_engine(settings)


def test_public_demo_seed_rejects_a_digest_outside_the_allowlist(tmp_path: Path) -> None:
    approved = hashlib.sha256(b"approved synthetic file").hexdigest()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "synthetic_only": True,
                "documents": [{"filename": "approved.txt", "sha256": approved}],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        jwt_secret="seed-test-secret-at-least-32-characters",
        public_demo_mode=True,
        public_demo_allowlist=allowlist,
        demo_admin_password="not-used-admin",
        demo_analyst_password="not-used-analyst",
        demo_reader_password="not-used-reader",
    )

    with pytest.raises(RuntimeError, match=r"^demo corpus document is not approved$"):
        _assert_seed_digest_allowed(
            settings,
            hashlib.sha256(b"unapproved content").hexdigest(),
        )

    _assert_seed_digest_allowed(settings, approved)


def test_seed_embedding_provider_uses_the_configured_model_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "onnx"
    manifest_path = tmp_path / "model.json"
    settings = Settings(
        jwt_secret="seed-test-secret-at-least-32-characters",
        answer_mode="extractive-local-onnx",
        embedding_model_path=model_path,
        embedding_manifest_path=manifest_path,
        demo_admin_password="not-used-admin",
        demo_analyst_password="not-used-analyst",
        demo_reader_password="not-used-reader",
    )
    observed: dict[str, object] = {}

    class Bundle:
        embedding = object()

    def fake_build(mode: str, **kwargs: object) -> Bundle:
        observed.update({"mode": mode, **kwargs})
        return Bundle()

    monkeypatch.setattr("evidencedesk_api.seed_cli.build_provider_bundle", fake_build)

    embedding = _build_seed_embeddings(settings)

    assert embedding is Bundle.embedding
    assert observed == {
        "mode": "extractive-local-onnx",
        "model_path": model_path,
        "manifest_path": manifest_path,
    }
