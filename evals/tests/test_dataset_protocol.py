from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert report.synthetic_only is True


def test_validator_accepts_an_explicit_corpus_for_a_new_blind_holdout() -> None:
    report = validate_manifest(
        ROOT / "datasets" / "evaluation_cases.json",
        corpus_path=ROOT / "datasets" / "corpus_manifest.json",
    )

    assert report.total_cases == 40


def test_validator_rejects_a_document_without_runner_required_filename(
    tmp_path: Path,
) -> None:
    corpus = {
        "dataset_version": "schema-test-v1",
        "synthetic_only": True,
        "documents": [{"id": "doc-1", "pages": ["Synthetic evidence."]}],
    }
    manifest = {
        "dataset_version": "schema-test-v1",
        "parameters_version": "schema-test-params",
        "mode": "extractive-local-onnx",
        "seed": 1,
        "metrics": {},
        "cases": [],
        "extraction_targets": [],
    }
    corpus_path = tmp_path / "corpus.json"
    manifest_path = tmp_path / "manifest.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="document filename"):
        validate_manifest(manifest_path, corpus_path=corpus_path)
