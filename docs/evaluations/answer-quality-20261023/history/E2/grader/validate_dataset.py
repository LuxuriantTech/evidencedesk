"""Structural validator for the sealed synthetic evaluation manifest.

It deliberately never calls the retrieval or answer engine: opening a holdout is
an explicit evaluation action, not a dataset-validation side effect.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from evidencedesk_api.extraction import SupplierExtraction


@dataclass(frozen=True)
class ValidationReport:
    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    adversarial_or_ambiguous_cases: int
    development_cases: int
    holdout_cases: int
    document_ids_are_known: bool
    citations_are_exact: bool
    extraction_expectations_are_traceable: bool
    synthetic_only: bool


_SCALAR_EXTRACTION_FIELDS = frozenset(
    {"organization_name", "document_type", "effective_date", "renewal_date"}
)
_LIST_EXTRACTION_FIELDS = frozenset(
    {"important_amounts", "obligations", "responsible_people", "risks"}
)
_EXTRACTION_FIELDS = _SCALAR_EXTRACTION_FIELDS | _LIST_EXTRACTION_FIELDS
_SUPPLIER_EXTRACTION_FIELDS = frozenset(field.name for field in fields(SupplierExtraction))
if _EXTRACTION_FIELDS != _SUPPLIER_EXTRACTION_FIELDS:
    raise RuntimeError("evaluation extraction schema differs from SupplierExtraction")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_manifest(
    path: Path,
    *,
    corpus_path: Path | None = None,
) -> ValidationReport:
    manifest = _load_json(path)
    root = path.parent.parent
    corpus = _load_json(corpus_path or root / "datasets" / "corpus_manifest.json")
    raw_documents = corpus.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("corpus documents must be a non-empty list")
    documents: dict[str, dict[str, Any]] = {}
    for item in raw_documents:
        if not isinstance(item, dict):
            raise ValueError("each corpus document must be an object")
        document_id = item.get("id")
        if not isinstance(document_id, str) or not document_id or document_id in documents:
            raise ValueError("document ids must be unique non-empty strings")
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"document filename is required for {document_id}")
        pages = item.get("pages")
        if (
            not isinstance(pages, list)
            or not pages
            or not all(isinstance(page, str) and page.strip() for page in pages)
        ):
            raise ValueError(f"document pages must be non-empty strings for {document_id}")
        documents[document_id] = item
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    if manifest.get("dataset_version") != corpus.get("dataset_version"):
        raise ValueError("evaluation and corpus versions must match")
    if manifest.get("mode") not in {
        "extractive-local",
        "extractive-local-onnx",
        "grounded-local-v3",
    }:
        raise ValueError("the sealed dataset must use an approved local extractive mode")
    if not isinstance(manifest.get("seed"), int):
        raise ValueError("manifest seed is required")

    ids: set[str] = set()
    splits: dict[str, set[str]] = {"development": set(), "holdout": set()}
    known = True
    exact = True
    traceable_extractions = True
    answerable = unanswerable = adversarial = 0
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each evaluation case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError("case ids must be unique non-empty strings")
        ids.add(case_id)
        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question for {case_id} must be a non-empty string")
        split = case.get("split")
        if split not in splits:
            raise ValueError(f"invalid split for {case_id}")
        splits[split].add(case_id)
        kind = case.get("kind")
        if kind == "answerable":
            answerable += 1
        elif kind == "unanswerable":
            unanswerable += 1
        elif kind in {"ambiguous", "adversarial"}:
            adversarial += 1
        else:
            raise ValueError(f"invalid kind for {case_id}")
        expected_answer = case.get("expected_answer")
        if kind == "answerable" and (
            not isinstance(expected_answer, str) or not expected_answer.strip()
        ):
            raise ValueError(f"expected answer for {case_id} must be a non-empty string")
        if kind != "answerable" and expected_answer is not None:
            raise ValueError(f"non-answerable case {case_id} cannot have an expected answer")
        citations = case.get("expected_citations", [])
        if not isinstance(citations, list):
            raise ValueError(f"expected citations for {case_id} must be a list")
        document_scope = case.get("document_ids")
        if document_scope is not None and (
            not isinstance(document_scope, list)
            or not document_scope
            or not all(isinstance(item, str) and item in documents for item in document_scope)
        ):
            raise ValueError(f"invalid document scope for {case_id}")
        if kind == "answerable" and not citations:
            raise ValueError(f"answerable case {case_id} needs a citation")
        if kind == "unanswerable" and citations:
            raise ValueError(f"unanswerable case {case_id} cannot have citations")
        for citation in citations:
            if not isinstance(citation, dict):
                raise ValueError(f"each expected citation for {case_id} must be an object")
            citation_document_id = citation.get("document_id")
            if not isinstance(citation_document_id, str):
                raise ValueError(f"each expected citation for {case_id} needs a document id")
            document = documents.get(citation_document_id)
            if document is None:
                known = False
                continue
            page = citation.get("page")
            if isinstance(page, bool) or not isinstance(page, int):
                raise ValueError(
                    f"citation page for {case_id} must be a positive integer"
                )
            if page < 1 or page > len(document["pages"]):
                exact = False
                continue
            excerpt = citation.get("excerpt")
            if not isinstance(excerpt, str) or excerpt not in document["pages"][page - 1]:
                exact = False
    if splits["development"] & splits["holdout"]:
        raise ValueError("splits overlap")
    if not manifest.get("parameters_version"):
        raise ValueError("parameters_version is required")
    if not isinstance(manifest.get("metrics"), dict):
        raise ValueError("metric definitions are required")
    extraction_targets = manifest.get("extraction_targets", [])
    if not isinstance(extraction_targets, list):
        raise ValueError("extraction targets must be a list")
    for target in extraction_targets:
        if not isinstance(target, dict):
            raise ValueError("each extraction target must be an object")
        if target.get("split") not in splits:
            raise ValueError("each extraction target needs a valid split")
        target_document_id = target.get("document_id")
        if not isinstance(target_document_id, str):
            raise ValueError("each extraction target needs a document id")
        document = documents.get(target_document_id)
        fields = target.get("fields")
        field_citations = target.get("field_citations")
        if (
            document is None
            or not isinstance(fields, dict)
            or not isinstance(field_citations, dict)
        ):
            traceable_extractions = False
            continue
        unsupported_fields = set(fields) - _EXTRACTION_FIELDS
        if unsupported_fields:
            raise ValueError(
                f"unsupported extraction field: {sorted(unsupported_fields)[0]}"
            )
        unsupported_citation_fields = set(field_citations) - set(fields)
        if unsupported_citation_fields:
            raise ValueError(
                "field citations reference an unsupported extraction field: "
                f"{sorted(unsupported_citation_fields)[0]}"
            )
        for field, value in fields.items():
            if field in _SCALAR_EXTRACTION_FIELDS:
                if value is not None and (
                    not isinstance(value, str) or not value.strip()
                ):
                    raise ValueError(
                        f"scalar extraction field {field} must be null or a non-empty string"
                    )
                expected_value_count = 0 if value is None else 1
            else:
                if not isinstance(value, list) or not all(
                    isinstance(item, str) and item.strip() for item in value
                ):
                    raise ValueError(
                        f"list extraction field {field} must contain non-empty strings"
                    )
                expected_value_count = len(value)
            citations = field_citations.get(field, [])
            if not isinstance(citations, list):
                raise ValueError(f"field citations for {field} must be a list")
            if len(citations) != expected_value_count:
                raise ValueError(
                    f"field {field} needs one citation per expected value"
                )
            if expected_value_count == 0:
                continue
            for citation in citations:
                if not isinstance(citation, dict):
                    raise ValueError(f"each field citation for {field} must be an object")
                citation_document_id = citation.get("document_id", target_document_id)
                if citation_document_id != target_document_id:
                    traceable_extractions = False
                    continue
                page = citation.get("page")
                excerpt = citation.get("excerpt")
                if isinstance(page, bool) or not isinstance(page, int):
                    raise ValueError(
                        f"field citation page for {field} must be a positive integer"
                    )
                if (
                    page < 1
                    or page > len(document["pages"])
                    or not isinstance(excerpt, str)
                    or excerpt not in document["pages"][page - 1]
                ):
                    traceable_extractions = False
    return ValidationReport(
        total_cases=len(cases),
        answerable_cases=answerable,
        unanswerable_cases=unanswerable,
        adversarial_or_ambiguous_cases=adversarial,
        development_cases=len(splits["development"]),
        holdout_cases=len(splits["holdout"]),
        document_ids_are_known=known,
        citations_are_exact=exact,
        extraction_expectations_are_traceable=traceable_extractions,
        synthetic_only=corpus.get("synthetic_only") is True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    print(validate_manifest(args.manifest, corpus_path=args.corpus))


if __name__ == "__main__":
    main()
