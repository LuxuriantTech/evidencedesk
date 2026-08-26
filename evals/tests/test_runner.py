from __future__ import annotations

from pathlib import Path

import pytest
from evidencedesk_api.extraction import EvidenceValue, SupplierExtraction
from evidencedesk_api.retrieval import Citation, EvidenceChunk

from evals.runner import (
    EvaluationError,
    _citation_matches,
    _extraction_counts,
    _scoped_chunks,
    claim_holdout_once,
    evaluate_manifest,
)

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
    assert result["mode"] == "extractive-local-hash"
    assert result["estimated_cost_usd"] == 0.0
    assert result["latency_median_ms"] >= 0
    assert result["latency_p95_ms"] >= result["latency_median_ms"]
    assert result["metric_counts"]["citation_returned"] >= 0
    assert isinstance(result["cases"][0]["answer"], str)
    assert isinstance(result["cases"][0]["citations"], list)
    answerable = next(case for case in result["cases"] if case["kind"] == "answerable")
    assert isinstance(answerable["expected_citation_matches"], list)
    assert isinstance(answerable["retrieval_matches_at_5"], list)
    derived_rank = next(
        (
            index
            for index, matched in enumerate(answerable["retrieval_matches_at_5"], start=1)
            if matched
        ),
        None,
    )
    assert answerable["retrieval_rank"] == derived_rank
    assert result["citation_recall"] >= 0
    assert result["metric_counts"]["citation_expected"] >= 1
    assert result["metric_counts"]["citation_expected_matched"] >= 0


def test_holdout_claim_is_single_use_and_requires_an_explicit_gate(tmp_path: Path) -> None:
    lock = tmp_path / "holdout.lock"

    with pytest.raises(EvaluationError, match="explicitly authorized"):
        claim_holdout_once(lock, allow_holdout=False)

    claim_holdout_once(
        lock,
        allow_holdout=True,
        evidence={"manifest_sha256": "a" * 64, "engine_fingerprint": "b" * 64},
    )
    assert lock.exists()
    assert '"engine_fingerprint"' in lock.read_text(encoding="utf-8")

    with pytest.raises(EvaluationError, match="already opened"):
        claim_holdout_once(
            lock,
            allow_holdout=True,
            evidence={"manifest_sha256": "a" * 64, "engine_fingerprint": "b" * 64},
        )


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


def test_case_scope_excludes_documents_from_another_dossier() -> None:
    chunks = [
        EvidenceChunk("one", "doc-one", "one.txt", 1, None, "Alpha", [0.0]),
        EvidenceChunk("two", "doc-two", "two.txt", 1, None, "Beta", [0.0]),
    ]

    scoped = _scoped_chunks({"document_ids": ["doc-two"]}, chunks)

    assert [chunk.document_id for chunk in scoped] == ["doc-two"]


def test_each_extracted_value_requires_its_own_expected_citation() -> None:
    first_citation = Citation("c1", "doc", "doc.txt", 1, None, "Obligation: alpha")
    extraction = SupplierExtraction(
        organization_name=EvidenceValue(None),
        document_type=EvidenceValue(None),
        effective_date=EvidenceValue(None),
        renewal_date=EvidenceValue(None),
        important_amounts=EvidenceValue(()),
        obligations=EvidenceValue(("alpha", "beta"), (first_citation,)),
        responsible_people=EvidenceValue(()),
        risks=EvidenceValue(()),
    )
    manifest = {
        "extraction_targets": [
            {
                "document_id": "doc",
                "split": "holdout",
                "fields": {"obligations": ["alpha", "beta"]},
                "field_citations": {
                    "obligations": [
                        {"page": 1, "excerpt": "Obligation: alpha"},
                        {"page": 2, "excerpt": "Obligation: beta"},
                    ]
                },
            }
        ]
    }

    true_positive, predicted, gold = _extraction_counts(
        manifest, {"doc": extraction}, split="holdout"
    )

    assert (true_positive, predicted, gold) == (1, 2, 2)
