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

from evals.runner import _meets_acceptance_targets
from evals.schema_versions import (
    ATTESTED_HOLDOUT_RAW_SCHEMA,
    RECALCULATED_EVALUATION_SCHEMA,
)

_STRICT_RETRIEVAL_SCHEMAS = {
    "evaluation-result-v4",
    ATTESTED_HOLDOUT_RAW_SCHEMA,
    # Immutable v7 and older artifacts retain this historical identifier.
    "evidencedesk-holdout-raw-v4",
}
_LEGACY_RETRIEVAL_SCHEMAS = {
    None,
    "evaluation-result-v1",
    "evaluation-result-v2",
    "evaluation-result-v3",
    "holdout-v4-raw-result-v2",
    "evidencedesk-holdout-raw-v3",
}


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
    # Metrics are persisted at millisecond precision; tolerate one final-unit
    # rounding difference while still rejecting any material timing drift.
    if abs(recorded - recalculated) > 0.001001:
        raise ValueError(f"recorded {key.replace('_', ' ')} differs from raw decisions")
    return True


def recalculate_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    schema_version = raw.get("schema_version")
    if schema_version in _STRICT_RETRIEVAL_SCHEMAS:
        strict_retrieval_evidence = True
    elif schema_version in _LEGACY_RETRIEVAL_SCHEMAS:
        strict_retrieval_evidence = False
    else:
        raise ValueError(f"unsupported raw artifact schema: {schema_version!r}")
    cases = raw.get("cases", [])
    extraction = raw.get("extraction_evaluation", [])
    citation_correct = citation_returned = 0
    citation_expected = citation_expected_matched = 0
    answerable_correct = answerable_total = 0
    abstention_correct = abstention_total = 0
    errors = 0
    schema_errors = 0
    retrieval_hits = 0
    retrieval_reciprocal_rank = 0.0
    retrieval_evidence_verified_cases = 0
    retrieval_completed_cases = 0
    latencies: list[float] = []
    for case in cases:
        kind = case["kind"]
        status = case["status"]
        if status == "error":
            errors += 1
            schema_flag = case.get("schema_error", False)
            if not isinstance(schema_flag, bool):
                raise ValueError("schema error flag must be boolean")
            schema_errors += int(schema_flag)
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
            retrieved = case.get("retrieved", [])
            if not isinstance(retrieved, list):
                raise ValueError("retrieved passages must be a list")
            retrieval_completed = case.get("retrieval_completed")
            if strict_retrieval_evidence and not isinstance(retrieval_completed, bool):
                raise ValueError("strict raw artifact is missing retrieval completion state")
            if retrieval_completed is True:
                retrieval_completed_cases += 1
            elif retrieval_completed is False and retrieved:
                raise ValueError("retrieval cannot be incomplete with recorded passages")
            expected_locations = case.get("expected_retrieval_locations_at_5")
            if strict_retrieval_evidence and expected_locations is None:
                raise ValueError("strict raw artifact is missing expected retrieval evidence")
            if expected_locations is not None:
                if not isinstance(expected_locations, list) or not expected_locations:
                    raise ValueError("expected retrieval locations must be a non-empty list")
                expected_pairs: set[tuple[str, int]] = set()
                for location in expected_locations:
                    if (
                        not isinstance(location, dict)
                        or not isinstance(location.get("document_id"), str)
                        or not isinstance(location.get("page"), int)
                        or isinstance(location.get("page"), bool)
                        or int(location["page"]) < 1
                    ):
                        raise ValueError("expected retrieval location is invalid")
                    expected_pairs.add((str(location["document_id"]), int(location["page"])))
                derived_matches: list[bool] = []
                for item in retrieved[:5]:
                    if (
                        not isinstance(item, dict)
                        or not isinstance(item.get("document_id"), str)
                        or not isinstance(item.get("page"), int)
                        or isinstance(item.get("page"), bool)
                    ):
                        raise ValueError("retrieved passage location is invalid")
                    derived_matches.append(
                        (str(item["document_id"]), int(item["page"])) in expected_pairs
                    )
                if retrieval_matches != derived_matches:
                    raise ValueError(
                        "retrieval matches differ from retrieved passages and expected evidence"
                    )
                retrieval_matches = derived_matches
                retrieval_evidence_verified_cases += 1
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
                if rank > len(retrieved):
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
        "retrieval_evidence_verified_cases": retrieval_evidence_verified_cases,
        "retrieval_completed_cases": retrieval_completed_cases,
        "abstention_correct": abstention_correct,
        "abstention_total": abstention_total,
        "extraction_true_positive": extraction_tp,
        "extraction_predicted": extraction_predicted,
        "extraction_gold": extraction_gold,
        "errors": errors,
        "schema_errors": schema_errors,
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
    schema_error_rate = round(_ratio(schema_errors, len(cases)), 6)
    latency_median = round(statistics.median(latencies), 3) if latencies else 0.0
    latency_p95 = round(_p95(latencies), 3)
    median_matches = _assert_recorded_metric(raw, "latency_median_ms", latency_median)
    p95_matches = _assert_recorded_metric(raw, "latency_p95_ms", latency_p95)
    _assert_recorded_metric(raw, "retrieval_recall_at_5", retrieval_recall)
    _assert_recorded_metric(raw, "retrieval_mrr_at_5", retrieval_mrr)
    _assert_recorded_metric(raw, "schema_error_rate", schema_error_rate)
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
    verdict_metrics = {
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_case_accuracy": citation_case_accuracy,
        "abstention_accuracy": abstention_accuracy,
        "extraction_f1": rounded_extraction_f1,
        "error_rate": error_rate,
        "schema_error_rate": schema_error_rate,
    }
    verdict = "PASS" if _meets_acceptance_targets(verdict_metrics) else "FAIL"
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
        "schema_error_rate": schema_error_rate,
        "latency_median_ms": latency_median,
        "latency_p95_ms": latency_p95,
        "timing_integrity": {
            "recorded_latency_median_matches": median_matches,
            "recorded_latency_p95_matches": p95_matches,
            "indexing_ms_valid": indexing_valid,
            "total_wall_time_ms_valid": total_valid,
            "peak_rss_bytes_valid": rss_valid,
        },
        "retrieval_integrity": {
            "evidence_verified_cases": retrieval_evidence_verified_cases,
            "answerable_cases": answerable_total,
            "evidence_recalculable": retrieval_evidence_verified_cases == answerable_total,
            "completed_cases": retrieval_completed_cases,
            "fully_verified": (
                retrieval_evidence_verified_cases == answerable_total
                and retrieval_completed_cases == answerable_total
            ),
            "strict_schema": strict_retrieval_evidence,
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
        "schema_version": RECALCULATED_EVALUATION_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "raw_artifact_sha256": _sha256(args.input),
        **recalculate_metrics(raw),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
