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


def test_validator_reports_untraceable_gold_before_evaluation(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    manifest_path = tmp_path / "evaluation.json"
    corpus_path.write_text(
        json.dumps(
            {
                "dataset_version": "development-v3",
                "synthetic_only": True,
                "documents": [
                    {"id": "doc-1", "filename": "doc-1.md", "pages": ["Known evidence."]}
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "development-v3",
                "parameters_version": "answer-v3",
                "mode": "extractive-local-onnx",
                "seed": 7,
                "metrics": {"citation_precision": 0.9},
                "cases": [
                    {
                        "id": "case-1",
                        "split": "holdout",
                        "kind": "answerable",
                        "question": "What is known?",
                        "expected_answer": "Known evidence",
                        "expected_citations": [
                            {"document_id": "doc-1", "page": 1, "excerpt": "Not present"}
                        ],
                    }
                ],
                "extraction_targets": [],
            }
        ),
        encoding="utf-8",
    )

    report = validate_manifest(manifest_path, corpus_path=corpus_path)

    assert report.citations_are_exact is False


def test_validator_rejects_a_malformed_gold_citation_before_inference(
    tmp_path: Path,
) -> None:
    from evals.holdout import _preflight_stage
    from evals.runner import EvaluationError

    corpus_path = tmp_path / "corpus.json"
    manifest_path = tmp_path / "evaluation.json"
    corpus_path.write_text(
        json.dumps(
            {
                "dataset_version": "schema-test-v2",
                "synthetic_only": True,
                "documents": [
                    {"id": "doc-1", "filename": "doc-1.md", "pages": ["Evidence."]}
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "schema-test-v2",
                "parameters_version": "answer-v3",
                "mode": "grounded-local-v3",
                "seed": 19,
                "metrics": {},
                "cases": [
                        {
                            "id": "case-1",
                            "split": "holdout",
                            "kind": "answerable",
                            "question": "What is recorded?",
                            "expected_answer": "Evidence",
                            "expected_citations": ["malformed-citation"],
                        }
                ],
                "extraction_targets": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationError,
        match=r"preflight dataset and gold validation failed:.*citation.*object",
    ):
        _preflight_stage(
            "dataset and gold validation",
            lambda: validate_manifest(manifest_path, corpus_path=corpus_path),
        )


def _write_gold_fixture(
    tmp_path: Path,
    *,
    case: dict[str, object] | None = None,
    extraction_target: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    corpus_path = tmp_path / "corpus.json"
    manifest_path = tmp_path / "evaluation.json"
    corpus_path.write_text(
        json.dumps(
            {
                "dataset_version": "strict-gold-v1",
                "synthetic_only": True,
                "documents": [
                    {
                        "id": "doc-1",
                        "filename": "doc-1.md",
                        "pages": ["The named organization is North Quay Services."],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "strict-gold-v1",
                "parameters_version": "answer-v3",
                "mode": "grounded-local-v3",
                "seed": 23,
                "metrics": {},
                "cases": [case] if case is not None else [],
                "extraction_targets": (
                    [extraction_target] if extraction_target is not None else []
                ),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, corpus_path


@pytest.mark.parametrize(
    ("case_patch", "message"),
    [
        ({"question": ""}, "question"),
        ({"question": None}, "question"),
        ({"expected_answer": ""}, "expected answer"),
        ({"expected_answer": None}, "expected answer"),
    ],
)
def test_validator_rejects_incomplete_answerable_gold(
    tmp_path: Path,
    case_patch: dict[str, object],
    message: str,
) -> None:
    case: dict[str, object] = {
        "id": "case-1",
        "split": "holdout",
        "kind": "answerable",
        "question": "Which organization is named?",
        "expected_answer": "North Quay Services",
        "expected_citations": [
            {
                "document_id": "doc-1",
                "page": 1,
                "excerpt": "The named organization is North Quay Services.",
            }
        ],
    }
    case.update(case_patch)
    manifest_path, corpus_path = _write_gold_fixture(tmp_path, case=case)

    with pytest.raises(ValueError, match=message):
        validate_manifest(manifest_path, corpus_path=corpus_path)


def test_validator_rejects_a_missing_expected_answer(tmp_path: Path) -> None:
    manifest_path, corpus_path = _write_gold_fixture(
        tmp_path,
        case={
            "id": "case-1",
            "split": "holdout",
            "kind": "answerable",
            "question": "Which organization is named?",
            "expected_citations": [
                {
                    "document_id": "doc-1",
                    "page": 1,
                    "excerpt": "The named organization is North Quay Services.",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="expected answer"):
        validate_manifest(manifest_path, corpus_path=corpus_path)


@pytest.mark.parametrize(
    ("fields", "field_citations", "message"),
    [
        (
            {"not_a_supplier_field": "North Quay Services"},
            {
                "not_a_supplier_field": [
                    {
                        "page": 1,
                        "excerpt": "The named organization is North Quay Services.",
                    }
                ]
            },
            "unsupported extraction field",
        ),
        (
            {"organization_name": ["North Quay Services"]},
            {"organization_name": []},
            "scalar extraction field",
        ),
        (
            {"important_amounts": "North Quay Services"},
            {"important_amounts": []},
            "list extraction field",
        ),
        (
            {"important_amounts": ["North Quay Services", "Second amount"]},
            {
                "important_amounts": [
                    {
                        "page": 1,
                        "excerpt": "The named organization is North Quay Services.",
                    }
                ]
            },
            "one citation per expected value",
        ),
        (
            {"organization_name": "North Quay Services"},
            {"organization_name": ["not-an-object"]},
            "field citation.*object",
        ),
    ],
)
def test_validator_rejects_invalid_extraction_gold_schema(
    tmp_path: Path,
    fields: dict[str, object],
    field_citations: dict[str, object],
    message: str,
) -> None:
    manifest_path, corpus_path = _write_gold_fixture(
        tmp_path,
        extraction_target={
            "split": "holdout",
            "document_id": "doc-1",
            "fields": fields,
            "field_citations": field_citations,
        },
    )

    with pytest.raises(ValueError, match=message):
        validate_manifest(manifest_path, corpus_path=corpus_path)


def test_validator_rejects_boolean_gold_pages(tmp_path: Path) -> None:
    evidence = "The named organization is North Quay Services."
    manifest_path, corpus_path = _write_gold_fixture(
        tmp_path,
        case={
            "id": "case-1",
            "split": "holdout",
            "kind": "answerable",
            "question": "Which organization is named?",
            "expected_answer": "North Quay Services",
            "expected_citations": [
                {"document_id": "doc-1", "page": True, "excerpt": evidence}
            ],
        },
    )
    with pytest.raises(ValueError, match=r"citation page.*positive integer"):
        validate_manifest(manifest_path, corpus_path=corpus_path)

    extraction_manifest, extraction_corpus = _write_gold_fixture(
        tmp_path,
        extraction_target={
            "split": "holdout",
            "document_id": "doc-1",
            "fields": {"organization_name": "North Quay Services"},
            "field_citations": {
                "organization_name": [{"page": True, "excerpt": evidence}]
            },
        },
    )
    with pytest.raises(ValueError, match=r"field citation page.*positive integer"):
        validate_manifest(extraction_manifest, corpus_path=extraction_corpus)
