"""Recalculate EvidenceDesk metrics from a raw evaluation artifact only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _non_negative_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    measured = float(value)
    if not math.isfinite(measured) or measured < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return measured


def _assert_recorded_metric(raw: dict[str, Any], key: str, recalculated: float) -> bool:
    if key not in raw:
        return False
    recorded = _non_negative_finite(raw[key], label=key.replace("_", " "))
    if abs(recorded - recalculated) > 0.000001:
        raise ValueError(f"recorded {key.replace('_', ' ')} differs from raw decisions")
    return True


def recalculate_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    cases = raw.get("cases", [])
    extraction = raw.get("extraction_evaluation", [])
    citation_correct = citation_returned = 0
    citation_expected = citation_expected_matched = 0
    answerable_correct = answerable_total = 0
    abstention_correct = abstention_total = 0
    errors = 0
    retrieval_hits = 0
    retrieval_reciprocal_rank = 0.0
    latencies: list[float] = []
    for case in cases:
        kind = case["kind"]
        status = case["status"]
        if status == "error":
            errors += 1
        if "latency_ms" in case:
            latencies.append(_non_negative_finite(case["latency_ms"], label="case latency"))
        if kind == "answerable":
            answerable_total += 1
            decisions = [bool(value) for value in case.get("citation_matches", [])]
            citation_correct += sum(decisions)
            citation_returned += len(decisions)
            expected_decisions_raw = case.get("expected_citation_matches")
            if expected_decisions_raw is not None:
                if not isinstance(expected_decisions_raw, list) or not all(
                    isinstance(value, bool) for value in expected_decisions_raw
                ):
                    raise ValueError("expected citation matches must be a boolean list")
                citation_expected += len(expected_decisions_raw)
                citation_expected_matched += sum(expected_decisions_raw)
            retrieval_matches = case.get("retrieval_matches_at_5")
            rank = case.get("retrieval_rank")
            if retrieval_matches is not None:
                if not isinstance(retrieval_matches, list) or len(retrieval_matches) > 5 or not all(
                    isinstance(value, bool) for value in retrieval_matches
                ):
                    raise ValueError("retrieval matches at 5 must be a boolean list of length <= 5")
                derived_rank = next(
                    (index for index, matched in enumerate(retrieval_matches, start=1) if matched),
                    None,
                )
                if rank != derived_rank:
                    raise ValueError("retrieval rank differs from raw retrieval matches")
            elif rank is not None and (
                isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 or rank > 5
            ):
                raise ValueError("retrieval rank must be null or an integer from 1 to 5")
            if rank is not None:
                retrieved = case.get("retrieved", [])
                if not isinstance(retrieved, list) or rank > len(retrieved):
                    raise ValueError("retrieval rank exceeds the recorded retrieved passages")
                retrieval_hits += 1
                retrieval_reciprocal_rank += 1.0 / rank
            if status == "answered" and bool(case.get("answer_match")) and any(decisions):
                answerable_correct += 1
        else:
            abstention_total += 1
            expected_status = "ambiguous" if kind == "ambiguous" else "abstained"
            if status == expected_status:
                abstention_correct += 1

    extraction_tp = sum(int(item["true_positive"]) for item in extraction)
    extraction_predicted = sum(int(item["predicted"]) for item in extraction)
    extraction_gold = sum(int(item["gold"]) for item in extraction)
    extraction_precision = _ratio(extraction_tp, extraction_predicted)
    extraction_recall = _ratio(extraction_tp, extraction_gold)
    extraction_f1 = (
        2 * extraction_precision * extraction_recall / (extraction_precision + extraction_recall)
        if extraction_precision + extraction_recall
        else 0.0
    )
    counts = {
        "citation_correct": citation_correct,
        "citation_returned": citation_returned,
        "citation_expected": citation_expected,
        "citation_expected_matched": citation_expected_matched,
        "answerable_correct": answerable_correct,
        "answerable_total": answerable_total,
        "retrieval_hits_at_5": retrieval_hits,
        "abstention_correct": abstention_correct,
        "abstention_total": abstention_total,
        "extraction_true_positive": extraction_tp,
        "extraction_predicted": extraction_predicted,
        "extraction_gold": extraction_gold,
        "errors": errors,
    }
    citation_precision = round(_ratio(citation_correct, citation_returned), 6)
    citation_recall = round(_ratio(citation_expected_matched, citation_expected), 6)
    citation_case_accuracy = round(_ratio(answerable_correct, answerable_total), 6)
    retrieval_recall = round(_ratio(retrieval_hits, answerable_total), 6)
    retrieval_mrr = round(_ratio(retrieval_reciprocal_rank, answerable_total), 6)
    abstention_accuracy = round(_ratio(abstention_correct, abstention_total), 6)
    rounded_extraction_precision = round(extraction_precision, 6)
    rounded_extraction_recall = round(extraction_recall, 6)
    rounded_extraction_f1 = round(extraction_f1, 6)
    error_rate = round(_ratio(errors, len(cases)), 6)
    latency_median = round(statistics.median(latencies), 3) if latencies else 0.0
    latency_p95 = round(_p95(latencies), 3)
    median_matches = _assert_recorded_metric(raw, "latency_median_ms", latency_median)
    p95_matches = _assert_recorded_metric(raw, "latency_p95_ms", latency_p95)
    _assert_recorded_metric(raw, "retrieval_recall_at_5", retrieval_recall)
    _assert_recorded_metric(raw, "retrieval_mrr_at_5", retrieval_mrr)
    if citation_expected:
        _assert_recorded_metric(raw, "citation_recall", citation_recall)

    indexing_valid = total_valid = rss_valid = False
    indexing_ms = 0.0
    if "indexing_ms" in raw:
        indexing_ms = _non_negative_finite(raw["indexing_ms"], label="indexing")
        indexing_valid = True
    if "total_wall_time_ms" in raw:
        total_wall_time = _non_negative_finite(raw["total_wall_time_ms"], label="total wall time")
        if total_wall_time < max(indexing_ms, latency_p95):
            raise ValueError("total wall time is smaller than a recorded component")
        total_valid = True
    if "peak_rss_bytes" in raw:
        peak_rss = raw["peak_rss_bytes"]
        if isinstance(peak_rss, bool) or not isinstance(peak_rss, int) or peak_rss <= 0:
            raise ValueError("peak RSS must be a positive integer")
        rss_valid = True
    verdict = (
        "PASS"
        if citation_precision >= 0.90
        and citation_case_accuracy >= 0.90
        and abstention_accuracy >= 0.85
        and rounded_extraction_f1 >= 0.90
        and error_rate == 0.0
        else "FAIL"
    )
    metrics: dict[str, Any] = {
        "metric_counts": counts,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_case_accuracy": citation_case_accuracy,
        "retrieval_recall_at_5": retrieval_recall,
        "retrieval_mrr_at_5": retrieval_mrr,
        "abstention_accuracy": abstention_accuracy,
        "extraction_precision": rounded_extraction_precision,
        "extraction_recall": rounded_extraction_recall,
        "extraction_f1": rounded_extraction_f1,
        "error_rate": error_rate,
        "latency_median_ms": latency_median,
        "latency_p95_ms": latency_p95,
        "timing_integrity": {
            "recorded_latency_median_matches": median_matches,
            "recorded_latency_p95_matches": p95_matches,
            "indexing_ms_valid": indexing_valid,
            "total_wall_time_ms_valid": total_valid,
            "peak_rss_bytes_valid": rss_valid,
        },
        "verdict": verdict,
    }
    return metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"recalculation output already exists: {args.output}")
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    result = {
        "schema_version": "evidencedesk-evaluation-recalculation-v3",
        "created_at": datetime.now(UTC).isoformat(),
        "raw_artifact_sha256": _sha256(args.input),
        **recalculate_metrics(raw),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
