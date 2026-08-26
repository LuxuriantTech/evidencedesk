from __future__ import annotations

from pathlib import Path

from evals.validate_dataset import validate_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_versioned_manifest_enforces_the_sealed_evaluation_protocol() -> None:
    report = validate_manifest(ROOT / "datasets" / "evaluation_cases.json")

    assert report.total_cases >= 40
    assert report.answerable_cases >= 25
    assert report.unanswerable_cases >= 10
    assert report.adversarial_or_ambiguous_cases >= 5
    assert report.development_cases > 0
    assert report.holdout_cases > 0
    assert report.document_ids_are_known is True
    assert report.citations_are_exact is True
    assert report.extraction_expectations_are_traceable is True
