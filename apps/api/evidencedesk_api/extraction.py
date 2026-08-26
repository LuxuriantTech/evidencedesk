import re
from dataclasses import dataclass
from typing import Any

from evidencedesk_api.retrieval import Citation, EvidenceChunk


@dataclass(frozen=True, slots=True)
class EvidenceValue[T]:
    value: T | None
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True, slots=True)
class SupplierExtraction:
    organization_name: EvidenceValue[str]
    document_type: EvidenceValue[str]
    effective_date: EvidenceValue[str]
    renewal_date: EvidenceValue[str]
    important_amounts: EvidenceValue[tuple[str, ...]]
    obligations: EvidenceValue[tuple[str, ...]]
    responsible_people: EvidenceValue[tuple[str, ...]]
    risks: EvidenceValue[tuple[str, ...]]

    def as_mapping(self) -> dict[str, EvidenceValue[Any]]:
        return {
            "organization_name": self.organization_name,
            "document_type": self.document_type,
            "effective_date": self.effective_date,
            "renewal_date": self.renewal_date,
            "important_amounts": self.important_amounts,
            "obligations": self.obligations,
            "responsible_people": self.responsible_people,
            "risks": self.risks,
        }


def _citation(chunk: EvidenceChunk, excerpt: str) -> Citation:
    return Citation(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        page=chunk.page,
        section=chunk.section,
        excerpt=excerpt,
    )


def _first_label(
    chunks: list[EvidenceChunk], label: str
) -> tuple[str | None, tuple[Citation, ...]]:
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.+)$", re.IGNORECASE)
    for chunk in chunks:
        for line in chunk.text.splitlines():
            match = pattern.match(line.strip())
            if match:
                value = match.group(1).strip()
                if label.casefold().endswith("date"):
                    value = value.removesuffix(".")
                return value, (_citation(chunk, line.strip()),)
    return None, ()


def _all_label(
    chunks: list[EvidenceChunk], label: str
) -> tuple[tuple[str, ...], tuple[Citation, ...]]:
    values: list[str] = []
    citations: list[Citation] = []
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.+?)$", re.IGNORECASE)
    for chunk in chunks:
        for line in chunk.text.splitlines():
            stripped = line.strip()
            match = pattern.match(stripped)
            if match:
                value = match.group(1).strip()
                if label.casefold() in {"renewal date", "responsible manager"}:
                    value = value.removesuffix(".")
                values.append(value)
                citations.append(_citation(chunk, stripped))
    return tuple(values), tuple(citations)


def extract_supplier_fields(chunks: list[EvidenceChunk]) -> SupplierExtraction:
    organization, organization_citations = _first_label(chunks, "Organization")
    if organization is None:
        legal_entity = re.compile(r"\b(?:Ltd\.?|LLC|GmbH|Inc\.?)$")
        for chunk in chunks:
            candidate = chunk.text.strip()
            if chunk.page == 1 and legal_entity.search(candidate):
                organization = candidate
                organization_citations = (_citation(chunk, candidate),)
                break
    document_type, document_type_citations = _first_label(chunks, "Document type")
    effective_date, effective_date_citations = _first_label(chunks, "Effective date")
    renewal_dates, renewal_citations = _all_label(chunks, "Renewal date")
    renewal_values = list(renewal_dates)
    renewal_evidence = list(renewal_citations)
    renewal_pattern = re.compile(r"\brenews?\s+on\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
    for chunk in chunks:
        match = renewal_pattern.search(chunk.text)
        if match and match.group(1) not in renewal_values:
            renewal_values.append(match.group(1))
            renewal_evidence.append(_citation(chunk, chunk.text))

    amount_values: list[str] = []
    amount_citations: list[Citation] = []
    amount_pattern = re.compile(r"\b(?:EUR|USD|GBP)\s+[0-9][0-9,.]*\b")
    for chunk in chunks:
        for match in amount_pattern.finditer(chunk.text):
            amount_values.append(match.group(0))
            amount_citations.append(_citation(chunk, match.group(0)))

    labelled_obligations, labelled_obligation_citations = _all_label(chunks, "Obligation")
    obligations = list(labelled_obligations)
    obligation_citations = list(labelled_obligation_citations)
    obligation_pattern = re.compile(r"(?:\bshall\b|\bmust\b|^No\s+.+\bmay\b)", re.IGNORECASE)
    for chunk in chunks:
        for line in chunk.text.splitlines():
            candidate = line.strip()
            if candidate.casefold().startswith("obligation:"):
                continue
            if obligation_pattern.search(candidate) and candidate not in obligations:
                obligations.append(candidate)
                obligation_citations.append(_citation(chunk, candidate))

    labelled_people, labelled_people_citations = _all_label(chunks, "Responsible manager")
    people = list(labelled_people)
    people_citations = list(labelled_people_citations)
    person_pattern = re.compile(
        r"^(?:Service owner|Responsible contact|Owner):\s*([^,\n.]+)", re.IGNORECASE
    )
    for chunk in chunks:
        for line in chunk.text.splitlines():
            match = person_pattern.match(line.strip())
            if match:
                person = match.group(1).strip()
                if person not in people:
                    people.append(person)
                    people_citations.append(_citation(chunk, line.strip()))
    explicit_risks, explicit_risk_citations = _all_label(chunks, "Risk")

    risks = list(explicit_risks)
    risk_citations = list(explicit_risk_citations)
    unique_renewal_dates = tuple(dict.fromkeys(renewal_values))
    if len(unique_renewal_dates) > 1:
        risks.insert(0, f"Conflicting renewal dates: {', '.join(sorted(unique_renewal_dates))}")
        risk_citations = renewal_evidence + risk_citations
        renewal_date: str | None = None
        selected_renewal_citations: tuple[Citation, ...] = ()
    else:
        renewal_date = unique_renewal_dates[0] if unique_renewal_dates else None
        selected_renewal_citations = tuple(renewal_evidence)

    return SupplierExtraction(
        organization_name=EvidenceValue(organization, organization_citations),
        document_type=EvidenceValue(document_type, document_type_citations),
        effective_date=EvidenceValue(effective_date, effective_date_citations),
        renewal_date=EvidenceValue(renewal_date, selected_renewal_citations),
        important_amounts=EvidenceValue(tuple(amount_values), tuple(amount_citations)),
        obligations=EvidenceValue(tuple(obligations), tuple(obligation_citations)),
        responsible_people=EvidenceValue(tuple(people), tuple(people_citations)),
        risks=EvidenceValue(tuple(risks), tuple(risk_citations)),
    )
