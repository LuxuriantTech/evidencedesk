from __future__ import annotations

from pathlib import Path

import pytest
from evidencedesk_api.retrieval import Citation

from evals.runner import EvaluationError, _citation_matches, claim_holdout_once, evaluate_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_development_evaluation_never_opens_holdout_cases() -> None:
    result = evaluate_manifest(
        ROOT / "datasets/evaluation_cases.json",
        ROOT / "datasets/corpus_manifest.json",
        split="development",
    )

    assert result["split"] == "development"
    assert result["case_count"] == 21
    assert all(case["id"].startswith("dev-") for case in result["cases"])
    assert result["mode"] == "extractive-local"
    assert result["estimated_cost_usd"] == 0.0
    assert result["latency_median_ms"] >= 0
    assert result["latency_p95_ms"] >= result["latency_median_ms"]
    assert result["metric_counts"]["citation_returned"] >= 0
    assert isinstance(result["cases"][0]["answer"], str)
    assert isinstance(result["cases"][0]["citations"], list)


def test_holdout_claim_is_single_use_and_requires_an_explicit_gate(tmp_path: Path) -> None:
    lock = tmp_path / "holdout.lock"

    with pytest.raises(EvaluationError, match="explicitly authorized"):
        claim_holdout_once(lock, allow_holdout=False)

    claim_holdout_once(lock, allow_holdout=True)
    assert lock.exists()

    with pytest.raises(EvaluationError, match="already opened"):
        claim_holdout_once(lock, allow_holdout=True)


def test_extraction_citation_accepts_a_verbatim_subspan_on_the_target_document() -> None:
    citation = Citation(
        chunk_id="c1",
        document_id="northstar-msa-2026",
        document_name="northstar.pdf",
        page=1,
        section=None,
        excerpt="EUR 48,000",
    )

    assert _citation_matches(
        citation,
        [{"page": 1, "excerpt": "Annual platform fee: EUR 48,000"}],
        default_document_id="northstar-msa-2026",
    )
