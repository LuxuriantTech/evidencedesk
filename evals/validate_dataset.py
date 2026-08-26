"""Structural validator for the sealed synthetic evaluation manifest.

It deliberately never calls the retrieval or answer engine: opening a holdout is
an explicit evaluation action, not a dataset-validation side effect.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    documents = {item["id"]: item for item in corpus["documents"]}
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    if manifest.get("dataset_version") != corpus.get("dataset_version"):
        raise ValueError("evaluation and corpus versions must match")
    if manifest.get("mode") != "extractive-local":
        raise ValueError("the sealed baseline must use extractive-local")
    if not isinstance(manifest.get("seed"), int):
        raise ValueError("manifest seed is required")

    ids: set[str] = set()
    splits: dict[str, set[str]] = {"development": set(), "holdout": set()}
    known = True
    exact = True
    traceable_extractions = True
    answerable = unanswerable = adversarial = 0
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or case_id in ids:
            raise ValueError("case ids must be unique strings")
        ids.add(case_id)
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
        citations = case.get("expected_citations", [])
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
            document = documents.get(citation.get("document_id"))
            if document is None:
                known = False
                continue
            page = citation.get("page")
            if not isinstance(page, int) or page < 1 or page > len(document["pages"]):
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
    for target in manifest.get("extraction_targets", []):
        if target.get("split") not in splits:
            raise ValueError("each extraction target needs a valid split")
        document = documents.get(target.get("document_id"))
        fields = target.get("fields")
        field_citations = target.get("field_citations")
        if (
            document is None
            or not isinstance(fields, dict)
            or not isinstance(field_citations, dict)
        ):
            traceable_extractions = False
            continue
        for field, value in fields.items():
            if value in (None, []):
                continue
            citations = field_citations.get(field, [])
            if not citations:
                traceable_extractions = False
                continue
            for citation in citations:
                page = citation.get("page")
                excerpt = citation.get("excerpt")
                if (
                    not isinstance(page, int)
                    or page < 1
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    print(validate_manifest(args.manifest, corpus_path=args.corpus))


if __name__ == "__main__":
    main()
