import math
from collections.abc import Callable

import pytest

from evals.recalculate import recalculate_metrics


def _raw() -> dict[str, object]:
    return {
        "citation_precision": 0.5,
        "citation_recall": 0.333333,
        "citation_case_accuracy": 0.5,
        "retrieval_recall_at_5": 0.5,
        "retrieval_mrr_at_5": 0.25,
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
                "retrieval_rank": 2,
                "retrieved": [{"chunk_id": "a"}, {"chunk_id": "b"}],
                "latency_ms": 10.0,
            },
            {
                "kind": "answerable",
                "status": "abstained",
                "answer_match": False,
                "citation_matches": [],
                "expected_citation_matches": [False],
                "retrieval_matches_at_5": [False, False, False, False, False],
                "retrieval_rank": None,
                "retrieved": [{"chunk_id": "c"}],
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["cases"][0].update(retrieval_rank=1), "retrieval rank"),
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

