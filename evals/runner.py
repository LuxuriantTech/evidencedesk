"""Deterministic EvidenceDesk evaluation runner.

Development and holdout execution are deliberately separate. Structural dataset
validation is not an evaluation and does not call this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from evidencedesk_api.extraction import SupplierExtraction, extract_supplier_fields
from evidencedesk_api.processing import ParsedPage, chunk_pages
from evidencedesk_api.providers import DeterministicEmbeddingProvider
from evidencedesk_api.redaction import redact_pii
from evidencedesk_api.retrieval import EvidenceChunk, ExtractiveAnswerProvider, hybrid_rank


class EvaluationError(RuntimeError):
    pass


class CitationLike(Protocol):
    @property
    def document_id(self) -> str: ...

    @property
    def page(self) -> int: ...

    @property
    def excerpt(self) -> str: ...


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"{path} must contain an object")
    return value


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _build_chunks(
    corpus: dict[str, Any],
) -> tuple[list[EvidenceChunk], dict[str, list[EvidenceChunk]]]:
    provider = DeterministicEmbeddingProvider(dimension=384)
    chunks: list[EvidenceChunk] = []
    by_document: dict[str, list[EvidenceChunk]] = {}
    for document in corpus["documents"]:
        document_chunks: list[EvidenceChunk] = []
        for page_number, raw_page in enumerate(document["pages"], start=1):
            page = ParsedPage(
                page=page_number,
                blocks=((None, redact_pii(str(raw_page)).text),),
            )
            for chunk in chunk_pages([page], max_chars=900):
                identifier = f"{document['id']}:p{page_number}:c{chunk.ordinal}"
                evidence = EvidenceChunk(
                    id=identifier,
                    document_id=str(document["id"]),
                    document_name=str(document["filename"]),
                    page=page_number,
                    section=chunk.section,
                    text=chunk.text,
                    embedding=provider.embed(chunk.text),
                )
                chunks.append(evidence)
                document_chunks.append(evidence)
        by_document[str(document["id"])] = document_chunks
    return chunks, by_document


def _scoped_chunks(case: dict[str, Any], all_chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
    document_ids = case.get("document_ids")
    if document_ids is None:
        return all_chunks
    if (
        not isinstance(document_ids, list)
        or not document_ids
        or not all(isinstance(item, str) for item in document_ids)
    ):
        raise EvaluationError(f"case {case.get('id')} has an invalid document scope")
    allowed = set(document_ids)
    scoped = [chunk for chunk in all_chunks if chunk.document_id in allowed]
    if not scoped:
        raise EvaluationError(f"case {case.get('id')} document scope is empty")
    return scoped


def _citation_matches(
    returned: CitationLike,
    expected: list[dict[str, Any]],
    *,
    default_document_id: str | None = None,
) -> bool:
    document_id = returned.document_id
    page = returned.page
    excerpt = returned.excerpt
    return any(
        document_id == item.get("document_id", default_document_id)
        and page == item.get("page")
        and (str(item.get("excerpt", "")) in excerpt or excerpt in str(item.get("excerpt", "")))
        for item in expected
    )


def _field_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return [_normalize(value)]


def _extraction_counts(
    manifest: dict[str, Any],
    extractions: dict[str, SupplierExtraction],
    *,
    split: str,
) -> tuple[int, int, int]:
    true_positive = predicted = gold = 0
    for target in manifest.get("extraction_targets", []):
        target_split = target.get("split", "development")
        if target_split != split:
            continue
        document_id = str(target["document_id"])
        extraction = extractions.get(document_id)
        if extraction is None:
            gold += sum(len(_field_values(value)) for value in target["fields"].values())
            continue
        actual = extraction.as_mapping()
        expected_citations = target.get("field_citations", {})
        for field_name, expected_value in target["fields"].items():
            expected_values = _field_values(expected_value)
            actual_field = actual[field_name]
            actual_values = _field_values(actual_field.value)
            gold += len(expected_values)
            predicted += len(actual_values)
            citation_gold = expected_citations.get(field_name, [])
            unmatched = list(actual_values)
            for expected_index, expected_item in enumerate(expected_values):
                expected_value_citations = (
                    [citation_gold[expected_index]]
                    if len(citation_gold) == len(expected_values)
                    else citation_gold
                )
                citation_ok = any(
                    _citation_matches(
                        citation,
                        expected_value_citations,
                        default_document_id=document_id,
                    )
                    for citation in actual_field.citations
                )
                match = next(
                    (
                        candidate
                        for candidate in unmatched
                        if expected_item in candidate or candidate in expected_item
                    ),
                    None,
                )
                if match is not None and citation_ok:
                    true_positive += 1
                    unmatched.remove(match)
    return true_positive, predicted, gold


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def evaluate_manifest(manifest_path: Path, corpus_path: Path, *, split: str) -> dict[str, Any]:
    if split not in {"development", "holdout"}:
        raise EvaluationError("split must be development or holdout")
    manifest = _load(manifest_path)
    corpus = _load(corpus_path)
    if manifest.get("dataset_version") != corpus.get("dataset_version"):
        raise EvaluationError("dataset versions do not match")

    provider = DeterministicEmbeddingProvider(dimension=384)
    answer_provider = ExtractiveAnswerProvider()
    all_chunks, chunks_by_document = _build_chunks(corpus)
    cases = [item for item in manifest["cases"] if item.get("split") == split]
    latencies: list[float] = []
    errors = 0
    citation_correct = 0
    citation_returned = 0
    answerable_correct = 0
    answerable_total = 0
    abstention_correct = 0
    abstention_total = 0
    case_results: list[dict[str, Any]] = []

    for case in cases:
        started = time.perf_counter()
        try:
            ranked = hybrid_rank(
                str(case["question"]), _scoped_chunks(case, all_chunks), provider=provider
            )
            answer = answer_provider.answer(str(case["question"]), ranked)
            elapsed_ms = (time.perf_counter() - started) * 1_000
            latencies.append(elapsed_ms)
            kind = case["kind"]
            expected_citations = list(case.get("expected_citations", []))
            citation_matches = [
                _citation_matches(citation, expected_citations) for citation in answer.citations
            ]
            citation_correct += sum(citation_matches)
            citation_returned += len(citation_matches)
            expected_answer = case.get("expected_answer")
            answer_match = expected_answer is not None and _normalize(
                expected_answer
            ) in _normalize(answer.answer)
            if kind == "answerable":
                answerable_total += 1
                if answer.status == "answered" and answer_match and any(citation_matches):
                    answerable_correct += 1
            else:
                abstention_total += 1
                acceptable = answer.status == "abstained" or (
                    kind == "ambiguous" and answer.status == "ambiguous"
                )
                if acceptable:
                    abstention_correct += 1
            case_results.append(
                {
                    "id": case["id"],
                    "kind": kind,
                    "status": answer.status,
                    "answer": answer.answer,
                    "answer_match": answer_match,
                    "citations": [asdict(citation) for citation in answer.citations],
                    "citation_matches": citation_matches,
                    "latency_ms": round(elapsed_ms, 3),
                }
            )
        except Exception as exc:  # evaluation must report, not hide, engine errors
            errors += 1
            elapsed_ms = (time.perf_counter() - started) * 1_000
            latencies.append(elapsed_ms)
            case_results.append(
                {
                    "id": case["id"],
                    "kind": case["kind"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "latency_ms": round(elapsed_ms, 3),
                }
            )

    extractions = {
        document_id: extract_supplier_fields(document_chunks)
        for document_id, document_chunks in chunks_by_document.items()
    }
    extraction_tp, extraction_predicted, extraction_gold = _extraction_counts(
        manifest, extractions, split=split
    )
    extraction_precision = _safe_ratio(extraction_tp, extraction_predicted)
    extraction_recall = _safe_ratio(extraction_tp, extraction_gold)
    extraction_f1 = (
        2 * extraction_precision * extraction_recall / (extraction_precision + extraction_recall)
        if extraction_precision + extraction_recall
        else 0.0
    )
    citation_precision = _safe_ratio(citation_correct, citation_returned)
    citation_case_accuracy = _safe_ratio(answerable_correct, answerable_total)
    abstention_accuracy = _safe_ratio(abstention_correct, abstention_total)
    result: dict[str, Any] = {
        "schema_version": "evaluation-result-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": manifest["dataset_version"],
        "parameters_version": manifest["parameters_version"],
        "seed": manifest["seed"],
        "split": split,
        "mode": manifest["mode"],
        "case_count": len(cases),
        "citation_precision": round(citation_precision, 6),
        "citation_case_accuracy": round(citation_case_accuracy, 6),
        "extraction_precision": round(extraction_precision, 6),
        "extraction_recall": round(extraction_recall, 6),
        "extraction_f1": round(extraction_f1, 6),
        "abstention_accuracy": round(abstention_accuracy, 6),
        "latency_median_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
        "latency_p95_ms": round(_p95(latencies), 3),
        "error_rate": round(_safe_ratio(errors, len(cases)), 6),
        "estimated_cost_usd": 0.0,
        "manifest_sha256": _sha256_file(manifest_path),
        "corpus_sha256": _sha256_file(corpus_path),
        "engine_fingerprint": _engine_fingerprint(),
        "metric_counts": {
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
        },
        "targets": {
            "citation_precision": 0.90,
            "abstention_accuracy": 0.85,
            "extraction_f1": 0.90,
        },
        "cases": case_results,
        "extractions": {key: asdict(value) for key, value in extractions.items()},
    }
    result["verdict"] = (
        "PASS"
        if result["citation_precision"] >= 0.90
        and result["abstention_accuracy"] >= 0.85
        and result["extraction_f1"] >= 0.90
        and result["error_rate"] == 0.0
        else "FAIL"
    )
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _engine_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    sources = (
        root / "evals" / "runner.py",
        root / "apps" / "api" / "evidencedesk_api" / "retrieval.py",
        root / "apps" / "api" / "evidencedesk_api" / "extraction.py",
        root / "apps" / "api" / "evidencedesk_api" / "providers.py",
        root / "apps" / "api" / "evidencedesk_api" / "processing.py",
    )
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def claim_holdout_once(
    lock_path: Path,
    *,
    allow_holdout: bool,
    evidence: dict[str, str] | None = None,
) -> None:
    if not allow_holdout:
        raise EvaluationError("holdout execution must be explicitly authorized")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"opened_at": datetime.now(UTC).isoformat(), **(evidence or {})}
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise EvaluationError("holdout was already opened for this parameter version") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("datasets/evaluation_cases.json"))
    parser.add_argument("--corpus", type=Path, default=Path("datasets/corpus_manifest.json"))
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluations"))
    args = parser.parse_args()

    manifest = _load(args.manifest)
    if args.split == "holdout":
        lock = Path("artifacts/evaluations/locks") / (
            f"holdout-{manifest['dataset_version']}-{manifest['parameters_version']}.lock"
        )
        claim_holdout_once(
            lock,
            allow_holdout=args.allow_holdout,
            evidence={
                "dataset_version": str(manifest["dataset_version"]),
                "parameters_version": str(manifest["parameters_version"]),
                "manifest_sha256": _sha256_file(args.manifest),
                "corpus_sha256": _sha256_file(args.corpus),
                "engine_fingerprint": _engine_fingerprint(),
            },
        )
    result = evaluate_manifest(args.manifest, args.corpus, split=args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        f"{args.split}-{result['dataset_version']}-{result['parameters_version']}.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
