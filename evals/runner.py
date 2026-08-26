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
import unicodedata
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from evidencedesk_api.extraction import SupplierExtraction, extract_supplier_fields
from evidencedesk_api.processing import ParsedPage, chunk_pages
from evidencedesk_api.providers import DeterministicEmbeddingProvider, EmbeddingProvider
from evidencedesk_api.redaction import redact_pii
from evidencedesk_api.retrieval import (
    EvidenceChunk,
    ExtractiveAnswerProvider,
    Reranker,
    RetrievalMethod,
    rank_chunks,
)


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
    folded = unicodedata.normalize("NFKD", str(value).casefold())
    ascii_value = folded.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


_MONTH_NUMBERS = {
    "january": 1,
    "janvier": 1,
    "february": 2,
    "fevrier": 2,
    "march": 3,
    "mars": 3,
    "april": 4,
    "avril": 4,
    "may": 5,
    "mai": 5,
    "june": 6,
    "juin": 6,
    "july": 7,
    "juillet": 7,
    "august": 8,
    "aout": 8,
    "september": 9,
    "septembre": 9,
    "october": 10,
    "octobre": 10,
    "november": 11,
    "novembre": 11,
    "december": 12,
    "decembre": 12,
}


def _normalize_typed(value: object, *, value_type: str) -> str:
    text = str(value).strip()
    folded = _normalize(text)
    if value_type == "date":
        iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
        if iso:
            return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"
        slash = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", text)
        if slash:
            return f"{slash.group(3)}-{slash.group(2)}-{slash.group(1)}"
        day_first = re.search(r"\b(\d{1,2})(?:er)?\s+([a-z]+)\s+(\d{4})\b", folded)
        month_first = re.search(r"\b([a-z]+)\s+(\d{1,2})\s+(\d{4})\b", folded)
        if day_first and day_first.group(2) in _MONTH_NUMBERS:
            return (
                f"{int(day_first.group(3)):04d}-{_MONTH_NUMBERS[day_first.group(2)]:02d}-"
                f"{int(day_first.group(1)):02d}"
            )
        if month_first and month_first.group(1) in _MONTH_NUMBERS:
            return (
                f"{int(month_first.group(3)):04d}-{_MONTH_NUMBERS[month_first.group(1)]:02d}-"
                f"{int(month_first.group(2)):02d}"
            )
        return folded
    if value_type == "money":
        currency = ""
        upper = text.upper()
        currency_symbols = {
            "EUR": ("EUR", "€"),
            "USD": ("USD", "$"),
            "GBP": ("GBP", "£"),
        }
        for code, symbols in currency_symbols.items():
            if any(symbol in upper for symbol in symbols):
                currency = code
                break
        number_match = re.search(r"\d[\d ,.]*\d|\d", text)
        digits = re.sub(r"\D", "", number_match.group(0)) if number_match else ""
        return f"{currency}:{int(digits) if digits else ''}"
    return folded


def _value_matches(actual: object, expected: object, *, value_type: str) -> bool:
    return _normalize_typed(actual, value_type=value_type) == _normalize_typed(
        expected, value_type=value_type
    )


def _answer_matches(answer: object, expected: object) -> bool:
    expected_text = str(expected)
    normalized_expected = _normalize(expected_text)
    month_pattern = "|".join(re.escape(month) for month in _MONTH_NUMBERS)
    date_like = bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", expected_text)
        or re.search(rf"\b\d{{1,2}}(?:er)?\s+(?:{month_pattern})\s+\d{{4}}\b", normalized_expected)
        or re.search(rf"\b(?:{month_pattern})\s+\d{{1,2}}\s+\d{{4}}\b", normalized_expected)
    )
    if date_like:
        expected_date = _normalize_typed(expected_text, value_type="date")
        return expected_date in {
            _normalize_typed(match, value_type="date")
            for match in re.findall(
                r"\d{4}-\d{2}-\d{2}|\d{1,2}(?:er)?\s+[A-Za-zÀ-ÿ]+\s+\d{4}|"
                r"[A-Za-zÀ-ÿ]+\s+\d{1,2},?\s+\d{4}",
                str(answer),
            )
        }
    if re.search(r"(?:EUR|USD|GBP|[$€£]).*\d|\d.*(?:EUR|USD|GBP|[$€£])", expected_text):
        return _normalize_typed(expected_text, value_type="money") == _normalize_typed(
            answer, value_type="money"
        )
    return _normalize(expected) in _normalize(answer)


def _build_chunks(
    corpus: dict[str, Any],
    provider: EmbeddingProvider,
) -> tuple[list[EvidenceChunk], dict[str, list[EvidenceChunk]]]:
    chunks: list[EvidenceChunk] = []
    by_document: dict[str, list[EvidenceChunk]] = {}
    pending: list[tuple[str, str, str, int, str | None, str]] = []
    for document in corpus["documents"]:
        for page_number, raw_page in enumerate(document["pages"], start=1):
            page = ParsedPage(
                page=page_number,
                blocks=((None, redact_pii(str(raw_page)).text),),
            )
            for chunk in chunk_pages([page], max_chars=900):
                identifier = f"{document['id']}:p{page_number}:c{chunk.ordinal}"
                pending.append(
                    (
                        identifier,
                        str(document["id"]),
                        str(document["filename"]),
                        page_number,
                        chunk.section,
                        chunk.text,
                    )
                )
    embeddings = provider.embed_many([item[5] for item in pending])
    for item, embedding in zip(pending, embeddings, strict=True):
        identifier, document_id, document_name, page_number, section, text = item
        evidence = EvidenceChunk(
            id=identifier,
            document_id=document_id,
            document_name=document_name,
            page=page_number,
            section=section,
            text=text,
            embedding=embedding,
            embedding_model_id=provider.model_id,
        )
        chunks.append(evidence)
        by_document.setdefault(document_id, []).append(evidence)
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
    returned_excerpt = _normalize(excerpt)
    informative_tokens = re.findall(r"[a-z0-9]+", returned_excerpt)
    return len(informative_tokens) >= 2 and any(
        document_id == item.get("document_id", default_document_id)
        and page == item.get("page")
        and returned_excerpt in _normalize(item.get("excerpt", ""))
        for item in expected
    )


def _field_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


_FIELD_VALUE_TYPES = {
    "effective_date": "date",
    "renewal_date": "date",
    "important_amounts": "money",
}


def _extraction_evaluation(
    manifest: dict[str, Any],
    extractions: dict[str, SupplierExtraction],
    *,
    split: str,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    true_positive = predicted = gold = 0
    details: list[dict[str, Any]] = []
    for target in manifest.get("extraction_targets", []):
        target_split = target.get("split", "development")
        if target_split != split:
            continue
        document_id = str(target["document_id"])
        extraction = extractions.get(document_id)
        if extraction is None:
            for field_name, expected_value in target["fields"].items():
                expected_values = _field_values(expected_value)
                gold += len(expected_values)
                details.append(
                    {
                        "document_id": document_id,
                        "field": field_name,
                        "expected_values": expected_values,
                        "actual_values": [],
                        "value_decisions": [False] * len(expected_values),
                        "citation_decisions": [False] * len(expected_values),
                        "true_positive": 0,
                        "predicted": 0,
                        "gold": len(expected_values),
                    }
                )
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
            value_decisions: list[bool] = []
            citation_decisions: list[bool] = []
            field_true_positive = 0
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
                        if _value_matches(
                            candidate,
                            expected_item,
                            value_type=_FIELD_VALUE_TYPES.get(field_name, "text"),
                        )
                    ),
                    None,
                )
                if match is not None and citation_ok:
                    true_positive += 1
                    field_true_positive += 1
                    unmatched.remove(match)
                value_decisions.append(match is not None)
                citation_decisions.append(citation_ok)
            details.append(
                {
                    "document_id": document_id,
                    "field": field_name,
                    "expected_values": expected_values,
                    "actual_values": actual_values,
                    "value_decisions": value_decisions,
                    "citation_decisions": citation_decisions,
                    "true_positive": field_true_positive,
                    "predicted": len(actual_values),
                    "gold": len(expected_values),
                }
            )
    return true_positive, predicted, gold, details


def _extraction_counts(
    manifest: dict[str, Any],
    extractions: dict[str, SupplierExtraction],
    *,
    split: str,
) -> tuple[int, int, int]:
    true_positive, predicted, gold, _ = _extraction_evaluation(manifest, extractions, split=split)
    return true_positive, predicted, gold


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def evaluate_manifest(
    manifest_path: Path,
    corpus_path: Path,
    *,
    split: str,
    provider: EmbeddingProvider | None = None,
    method: RetrievalMethod = RetrievalMethod.HYBRID,
    reranker: Reranker | None = None,
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if split not in {"development", "holdout"}:
        raise EvaluationError("split must be development or holdout")
    manifest = _load(manifest_path)
    corpus = _load(corpus_path)
    if manifest.get("dataset_version") != corpus.get("dataset_version"):
        raise EvaluationError("dataset versions do not match")

    active_provider = provider or DeterministicEmbeddingProvider(dimension=384)
    answer_mode = (
        "extractive-local-onnx"
        if active_provider.mode == "local-semantic-onnx-v1"
        else "extractive-local-hash"
    )
    answer_provider = ExtractiveAnswerProvider(mode=answer_mode)
    indexing_started = time.perf_counter()
    all_chunks, chunks_by_document = _build_chunks(corpus, active_provider)
    indexing_ms = (time.perf_counter() - indexing_started) * 1_000
    cases = [item for item in manifest["cases"] if item.get("split") == split]
    latencies: list[float] = []
    errors = 0
    citation_correct = 0
    citation_returned = 0
    citation_expected = 0
    citation_expected_matched = 0
    answerable_correct = 0
    answerable_total = sum(case.get("kind") == "answerable" for case in cases)
    abstention_correct = 0
    abstention_total = len(cases) - answerable_total
    retrieval_hits = 0
    retrieval_reciprocal_rank = 0.0
    case_results: list[dict[str, Any]] = []

    for case in cases:
        started = time.perf_counter()
        try:
            ranked = rank_chunks(
                str(case["question"]),
                _scoped_chunks(case, all_chunks),
                provider=active_provider,
                method=method,
                reranker=reranker,
            )
            answer = answer_provider.answer(str(case["question"]), ranked)
            elapsed_ms = (time.perf_counter() - started) * 1_000
            latencies.append(elapsed_ms)
            kind = case["kind"]
            expected_citations = list(case.get("expected_citations", []))
            citation_matches = [
                _citation_matches(citation, expected_citations) for citation in answer.citations
            ]
            expected_citation_matches = [
                any(_citation_matches(citation, [expected]) for citation in answer.citations)
                for expected in expected_citations
            ]
            if kind == "answerable":
                citation_correct += sum(citation_matches)
                citation_returned += len(citation_matches)
                citation_expected += len(expected_citation_matches)
                citation_expected_matched += sum(expected_citation_matches)
                expected_locations = {
                    (str(item.get("document_id")), int(item.get("page", 0)))
                    for item in expected_citations
                }
                retrieval_matches_at_5 = [
                    (item.chunk.document_id, item.chunk.page) in expected_locations
                    for item in ranked[:5]
                ]
                retrieved_rank = next(
                    (
                        index
                        for index, matched in enumerate(retrieval_matches_at_5, start=1)
                        if matched
                    ),
                    None,
                )
                if retrieved_rank is not None:
                    retrieval_hits += 1
                    retrieval_reciprocal_rank += 1.0 / retrieved_rank
            else:
                retrieval_matches_at_5 = []
            expected_answer = case.get("expected_answer")
            answer_match = expected_answer is not None and _answer_matches(
                answer.answer, expected_answer
            )
            if kind == "answerable":
                if answer.status == "answered" and answer_match and any(citation_matches):
                    answerable_correct += 1
            else:
                acceptable = (
                    answer.status == "ambiguous"
                    if kind == "ambiguous"
                    else answer.status == "abstained"
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
                    "expected_citation_matches": expected_citation_matches,
                    "retrieval_matches_at_5": retrieval_matches_at_5,
                    "retrieval_rank": retrieved_rank if kind == "answerable" else None,
                    "retrieved": [
                        {
                            "chunk_id": item.chunk.id,
                            "document_id": item.chunk.document_id,
                            "page": item.chunk.page,
                            "score": round(item.score, 8),
                            "lexical_score": round(item.lexical_score, 8),
                            "dense_score": round(item.dense_score, 8),
                            "rerank_score": (
                                round(item.rerank_score, 8)
                                if item.rerank_score is not None
                                else None
                            ),
                        }
                        for item in ranked
                    ],
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
    (
        extraction_tp,
        extraction_predicted,
        extraction_gold,
        extraction_evaluation,
    ) = _extraction_evaluation(manifest, extractions, split=split)
    extraction_precision = _safe_ratio(extraction_tp, extraction_predicted)
    extraction_recall = _safe_ratio(extraction_tp, extraction_gold)
    extraction_f1 = (
        2 * extraction_precision * extraction_recall / (extraction_precision + extraction_recall)
        if extraction_precision + extraction_recall
        else 0.0
    )
    citation_precision = _safe_ratio(citation_correct, citation_returned)
    citation_recall = _safe_ratio(citation_expected_matched, citation_expected)
    citation_case_accuracy = _safe_ratio(answerable_correct, answerable_total)
    abstention_accuracy = _safe_ratio(abstention_correct, abstention_total)
    result: dict[str, Any] = {
        "schema_version": "evaluation-result-v3",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": manifest["dataset_version"],
        "parameters_version": manifest["parameters_version"],
        "seed": manifest["seed"],
        "split": split,
        "mode": answer_provider.mode,
        "embedding_model_id": active_provider.model_id,
        "retrieval_method": method.value,
        "reranker_model_id": reranker.model_id if reranker is not None else None,
        "case_count": len(cases),
        "citation_precision": round(citation_precision, 6),
        "citation_recall": round(citation_recall, 6),
        "citation_case_accuracy": round(citation_case_accuracy, 6),
        "retrieval_recall_at_5": round(_safe_ratio(retrieval_hits, answerable_total), 6),
        "retrieval_mrr_at_5": round(_safe_ratio(retrieval_reciprocal_rank, answerable_total), 6),
        "extraction_precision": round(extraction_precision, 6),
        "extraction_recall": round(extraction_recall, 6),
        "extraction_f1": round(extraction_f1, 6),
        "abstention_accuracy": round(abstention_accuracy, 6),
        "latency_median_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
        "latency_p95_ms": round(_p95(latencies), 3),
        "error_rate": round(_safe_ratio(errors, len(cases)), 6),
        "indexing_ms": round(indexing_ms, 3),
        "estimated_cost_usd": active_provider.estimated_cost_usd,
        "manifest_sha256": _sha256_file(manifest_path),
        "corpus_sha256": _sha256_file(corpus_path),
        "engine_fingerprint": _engine_fingerprint(),
        "metric_counts": {
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
        },
        "targets": {
            "citation_precision": 0.90,
            "citation_case_accuracy": 0.90,
            "abstention_accuracy": 0.85,
            "extraction_f1": 0.90,
        },
        "cases": case_results,
        "extractions": {key: asdict(value) for key, value in extractions.items()},
        "extraction_evaluation": extraction_evaluation,
        "runtime": runtime_metadata or {},
    }
    result["verdict"] = (
        "PASS"
        if result["citation_precision"] >= 0.90
        and result["citation_case_accuracy"] >= 0.90
        and result["abstention_accuracy"] >= 0.85
        and result["extraction_f1"] >= 0.90
        and result["error_rate"] == 0.0
        else "FAIL"
    )
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ENGINE_FINGERPRINT_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "evals/runner.py",
    "evals/holdout_v4.py",
    "evals/holdout.py",
    "evals/validate_dataset.py",
    "evals/benchmark.py",
    "evals/recalculate.py",
    "apps/api/evidencedesk_api/retrieval.py",
    "apps/api/evidencedesk_api/extraction.py",
    "apps/api/evidencedesk_api/providers.py",
    "apps/api/evidencedesk_api/processing.py",
)


def _engine_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in ENGINE_FINGERPRINT_PATHS:
        source = root / relative
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

    if args.split == "holdout":
        raise EvaluationError(
            "holdout execution requires the attested holdout runner: python -m evals.holdout"
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
