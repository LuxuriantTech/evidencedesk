import math
from collections.abc import Callable
from typing import Any

import pytest

from evals.recalculate import recalculate_metrics
from evals.schema_versions import ATTESTED_HOLDOUT_RAW_SCHEMA


def _raw() -> dict[str, Any]:
    return {
        "schema_version": "evaluation-result-v4",
        "citation_precision": 0.5,
        "citation_recall": 0.333333,
        "citation_case_accuracy": 0.5,
        "retrieval_recall_at_5": 0.5,
        "retrieval_mrr_at_5": 0.25,
        "schema_error_rate": 0.0,
        "latency_median_ms": 15.0,
        "latency_p95_ms": 20.0,
        "indexing_ms": 25.0,
        "total_wall_time_ms": 60.0,
        "peak_rss_bytes": 1024,
        "cases": [
            {
                "kind": "answerable",
                "status": "answered",
                "answer_match": True,
                "citation_matches": [True],
                "expected_citation_matches": [True, False],
                "retrieval_matches_at_5": [False, True, False, False, False],
                "expected_retrieval_locations_at_5": [
                    {"document_id": "gold-doc", "page": 2}
                ],
                "retrieval_rank": 2,
                "retrieval_completed": True,
                "retrieved": [
                    {"chunk_id": "a", "document_id": "other-doc", "page": 1},
                    {"chunk_id": "b", "document_id": "gold-doc", "page": 2},
                    {"chunk_id": "c", "document_id": "other-doc", "page": 3},
                    {"chunk_id": "d", "document_id": "other-doc", "page": 4},
                    {"chunk_id": "e", "document_id": "other-doc", "page": 5},
                ],
                "latency_ms": 10.0,
            },
            {
                "kind": "answerable",
                "status": "abstained",
                "answer_match": False,
                "citation_matches": [],
                "expected_citation_matches": [False],
                "retrieval_matches_at_5": [False, False, False, False, False],
                "expected_retrieval_locations_at_5": [
                    {"document_id": "missing-doc", "page": 9}
                ],
                "retrieval_rank": None,
                "retrieval_completed": True,
                "retrieved": [
                    {"chunk_id": "f", "document_id": "other-doc", "page": 1},
                    {"chunk_id": "g", "document_id": "other-doc", "page": 2},
                    {"chunk_id": "h", "document_id": "other-doc", "page": 3},
                    {"chunk_id": "i", "document_id": "other-doc", "page": 4},
                    {"chunk_id": "j", "document_id": "other-doc", "page": 5},
                ],
                "latency_ms": 20.0,
            },
        ],
        "extraction_evaluation": [],
    }


def test_recalculation_derives_retrieval_citation_recall_and_timings_from_raw_cases() -> None:
    result = recalculate_metrics(_raw())

    assert result["citation_recall"] == pytest.approx(1 / 3)
    assert result["retrieval_recall_at_5"] == 0.5
    assert result["retrieval_mrr_at_5"] == 0.25
    assert result["latency_median_ms"] == 15.0
    assert result["latency_p95_ms"] == 20.0
    assert result["timing_integrity"] == {
        "recorded_latency_median_matches": True,
        "recorded_latency_p95_matches": True,
        "indexing_ms_valid": True,
        "total_wall_time_ms_valid": True,
        "peak_rss_bytes_valid": True,
    }
    assert result["metric_counts"]["citation_expected"] == 3
    assert result["metric_counts"]["citation_expected_matched"] == 1
    assert result["metric_counts"]["retrieval_hits_at_5"] == 1
    assert result["retrieval_integrity"] == {
        "evidence_verified_cases": 2,
        "answerable_cases": 2,
        "evidence_recalculable": True,
        "completed_cases": 2,
        "fully_verified": True,
        "strict_schema": True,
    }
    assert result["schema_error_rate"] == 0.0


def test_strict_raw_rejects_missing_retrieval_evidence() -> None:
    raw = _raw()
    raw["cases"][0].pop("expected_retrieval_locations_at_5")

    with pytest.raises(ValueError, match="missing expected retrieval evidence"):
        recalculate_metrics(raw)


def test_current_generic_holdout_schema_is_strictly_recalculated() -> None:
    raw = _raw()
    raw["schema_version"] = ATTESTED_HOLDOUT_RAW_SCHEMA

    result = recalculate_metrics(raw)

    assert result["retrieval_integrity"]["strict_schema"] is True
    assert result["retrieval_recall_at_5"] == 0.5


def test_unknown_raw_schema_is_rejected_instead_of_treated_as_legacy() -> None:
    raw = _raw()
    raw["schema_version"] = "evidencedesk-holdout-raw-v999"

    with pytest.raises(ValueError, match="unsupported raw artifact schema"):
        recalculate_metrics(raw)


def test_recalculation_derives_schema_errors_from_raw_case_flags() -> None:
    raw = _raw()
    raw["cases"][1].update(
        status="error",
        schema_error=True,
    )
    raw["schema_error_rate"] = 0.5

    result = recalculate_metrics(raw)

    assert result["metric_counts"]["schema_errors"] == 1
    assert result["schema_error_rate"] == 0.5


def test_recalculated_verdict_fails_when_only_citation_recall_is_below_target() -> None:
    raw = _raw()
    first = raw["cases"][0]
    second = raw["cases"][1]
    first.update(
        citation_matches=[True],
        expected_citation_matches=[True, False],
        answer_match=True,
    )
    second.update(kind="unanswerable", status="abstained")
    raw.update(
        citation_precision=1.0,
        citation_recall=0.5,
        citation_case_accuracy=1.0,
        retrieval_recall_at_5=1.0,
        retrieval_mrr_at_5=0.5,
        extraction_evaluation=[{"true_positive": 1, "predicted": 1, "gold": 1}],
    )

    result = recalculate_metrics(raw)

    assert result["citation_recall"] == 0.5
    assert result["citation_precision"] == 1.0
    assert result["citation_case_accuracy"] == 1.0
    assert result["abstention_accuracy"] == 1.0
    assert result["extraction_f1"] == 1.0
    assert result["verdict"] == "FAIL"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["cases"][0].update(retrieval_rank=1), "retrieval rank"),
        (
            lambda raw: raw["cases"][0].update(
                retrieval_matches_at_5=[True, False, False, False, False]
            ),
            "retrieval matches",
        ),
        (lambda raw: raw["cases"][0].update(latency_ms=-1), "latency"),
        (lambda raw: raw.update(latency_p95_ms=19), "p95"),
        (lambda raw: raw.update(indexing_ms=math.inf), "indexing"),
        (lambda raw: raw.update(total_wall_time_ms=5), "total wall time"),
        (lambda raw: raw.update(peak_rss_bytes=0), "peak RSS"),
    ],
)
def test_recalculation_rejects_incoherent_retrieval_or_timing(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    raw = _raw()
    mutation(raw)

    with pytest.raises(ValueError, match=message):
        recalculate_metrics(raw)
