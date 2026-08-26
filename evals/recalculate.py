"""Recalculate EvidenceDesk metrics from a raw evaluation artifact only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def recalculate_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    cases = raw.get("cases", [])
    extraction = raw.get("extraction_evaluation", [])
    citation_correct = citation_returned = 0
    answerable_correct = answerable_total = 0
    abstention_correct = abstention_total = 0
    errors = 0
    for case in cases:
        kind = case["kind"]
        status = case["status"]
        if status == "error":
            errors += 1
        if kind == "answerable":
            answerable_total += 1
            decisions = [bool(value) for value in case.get("citation_matches", [])]
            citation_correct += sum(decisions)
            citation_returned += len(decisions)
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
        "answerable_correct": answerable_correct,
        "answerable_total": answerable_total,
        "abstention_correct": abstention_correct,
        "abstention_total": abstention_total,
        "extraction_true_positive": extraction_tp,
        "extraction_predicted": extraction_predicted,
        "extraction_gold": extraction_gold,
        "errors": errors,
    }
    citation_precision = round(_ratio(citation_correct, citation_returned), 6)
    citation_case_accuracy = round(_ratio(answerable_correct, answerable_total), 6)
    abstention_accuracy = round(_ratio(abstention_correct, abstention_total), 6)
    rounded_extraction_precision = round(extraction_precision, 6)
    rounded_extraction_recall = round(extraction_recall, 6)
    rounded_extraction_f1 = round(extraction_f1, 6)
    error_rate = round(_ratio(errors, len(cases)), 6)
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
        "citation_case_accuracy": citation_case_accuracy,
        "abstention_accuracy": abstention_accuracy,
        "extraction_precision": rounded_extraction_precision,
        "extraction_recall": rounded_extraction_recall,
        "extraction_f1": rounded_extraction_f1,
        "error_rate": error_rate,
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
        "schema_version": "evaluation-recalculation-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "raw_artifact_sha256": _sha256(args.input),
        **recalculate_metrics(raw),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
