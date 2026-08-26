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


_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|October|November|"
    r"December|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre"
)
_DATE = re.compile(
    rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{2}}/\d{{2}}/\d{{4}}|"
    rf"\d{{1,2}}(?:er)?\s+(?:{_MONTHS})\s+\d{{4}}|"
    rf"(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}})\b",
    re.IGNORECASE,
)
_AMOUNT = re.compile(
    r"(?:\b(?:EUR|USD|GBP)\s+[$€£]?\s*[0-9][0-9 ,.]*[0-9]|"
    r"[$€£]\s*[0-9][0-9 ,.]*[0-9](?:\s+(?:EUR|USD|GBP))?|"
    r"\b[0-9][0-9 ,.]*[0-9]\s+(?:EUR|USD|GBP|€|£|\$))",
    re.IGNORECASE,
)
_LEGAL_ENTITY = re.compile(r"\b[^\n:—]+?\s+(?:Ltd\.?|LLC|GmbH|Inc\.?)\s*$", re.IGNORECASE)
_DOCUMENT_TITLE = re.compile(
    r"\b(?:agreement|contract|addendum|schedule|statement of work|order form|licence|"
    r"accord|contrat|bon de commande|déclaration de travaux)\b",
    re.IGNORECASE,
)
_OBLIGATION = re.compile(
    r"(?:\bshall\b|\bmust\b|\bis required to\b|\bundertakes? to\b|\bdoit\b|"
    r"\bs'engage à\b|\best tenu de\b|^No\s+.+\bmay\b)",
    re.IGNORECASE,
)


def _line_entries(chunks: list[EvidenceChunk]) -> list[tuple[EvidenceChunk, str]]:
    return [
        (chunk, line.strip())
        for chunk in chunks
        for line in chunk.text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _label_match(line: str, aliases: tuple[str, ...]) -> str | None:
    joined = "|".join(re.escape(alias) for alias in aliases)
    match = re.match(rf"^(?:{joined})\s*(?::|—|-)\s*(.+?)\s*$", line, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _first_label_value(
    entries: list[tuple[EvidenceChunk, str]], aliases: tuple[str, ...]
) -> tuple[str | None, tuple[Citation, ...]]:
    for chunk, line in entries:
        value = _label_match(line, aliases)
        if value is not None:
            return value, (_citation(chunk, line),)
    return None, ()


def _date_from_entries(
    entries: list[tuple[EvidenceChunk, str]],
    *,
    labels: tuple[str, ...],
    context_pattern: re.Pattern[str],
) -> tuple[list[str], list[Citation]]:
    values: list[str] = []
    citations: list[Citation] = []
    for chunk, line in entries:
        labelled = _label_match(line, labels)
        match = _DATE.search(labelled or line)
        if match is None or (labelled is None and context_pattern.search(line) is None):
            continue
        value = match.group(0).removeprefix("le ").removesuffix(".")
        if value not in values:
            values.append(value)
            citations.append(_citation(chunk, line))
    return values, citations


def _append_unique(
    values: list[str], citations: list[Citation], value: str, citation: Citation
) -> None:
    if value not in values:
        values.append(value)
        citations.append(citation)


def extract_supplier_fields(chunks: list[EvidenceChunk]) -> SupplierExtraction:
    entries = _line_entries(chunks)
    organization, organization_citations = _first_label_value(
        entries,
        (
            "Organization",
            "Organisation",
            "Supplier",
            "Fournisseur",
            "Vendor legal entity",
            "Client record supplier",
            "Société concernée",
            "Fournisseur / Organization",
        ),
    )
    if organization is None:
        for chunk, line in entries:
            match = _LEGAL_ENTITY.search(line)
            if chunk.page == 1 and match:
                organization = match.group(0).strip()
                organization_citations = (_citation(chunk, line),)
                break

    document_type, document_type_citations = _first_label_value(
        entries,
        (
            "Document type",
            "Document kind",
            "Category",
            "Catégorie",
            "Nature du document",
            "Type de pièce",
        ),
    )
    if document_type is None:
        for chunk in chunks:
            for line in chunk.text.splitlines():
                candidate = line.strip().lstrip("#").strip()
                if chunk.page == 1 and candidate.isupper() and _DOCUMENT_TITLE.search(candidate):
                    document_type = candidate
                    document_type_citations = (_citation(chunk, line.strip()),)
                    break
            if document_type is not None:
                break

    effective_values, effective_evidence = _date_from_entries(
        entries,
        labels=(
            "Effective date",
            "Effective from",
            "Commencement",
            "Date d'application",
            "Prise d'effet",
            "Entrée en vigueur",
        ),
        context_pattern=re.compile(
            r"\b(?:takes? effect|commences?|effective from|entre en vigueur|vigueur)\b",
            re.IGNORECASE,
        ),
    )
    renewal_values, renewal_evidence = _date_from_entries(
        entries,
        labels=("Renewal date", "Renouvellement", "Prochaine échéance de renouvellement"),
        context_pattern=re.compile(
            r"\b(?:renewal|renews?|reconduit|renouvellement)\b", re.IGNORECASE
        ),
    )

    amount_values: list[str] = []
    amount_citations: list[Citation] = []
    for chunk, line in entries:
        for match in _AMOUNT.finditer(line):
            amount = match.group(0).strip()
            _append_unique(amount_values, amount_citations, amount, _citation(chunk, amount))

    obligations: list[str] = []
    obligation_citations: list[Citation] = []
    for chunk, line in entries:
        labelled = _label_match(line, ("Obligation", "Engagement"))
        candidate = labelled or line
        if labelled is not None or _OBLIGATION.search(candidate):
            _append_unique(
                obligations,
                obligation_citations,
                candidate,
                _citation(chunk, line),
            )

    people: list[str] = []
    people_citations: list[Citation] = []
    person_aliases = (
        "Responsible manager",
        "Responsible contact",
        "Service owner",
        "Owner",
        "Accountable lead",
        "Contact opérationnel",
        "Responsable",
        "Responsable du compte",
        "Owner / Responsable",
    )
    for chunk, line in entries:
        value = _label_match(line, person_aliases)
        if value is not None:
            person = value.split(",", maxsplit=1)[0].strip().removesuffix(".")
            _append_unique(people, people_citations, person, _citation(chunk, line))

    risks: list[str] = []
    risk_citations: list[Citation] = []
    for chunk, line in entries:
        labelled = _label_match(line, ("Risk", "Risque", "Risk note", "Exposure note"))
        candidate = labelled or line
        implicit = bool(
            re.search(r"\bcontrol gap\b", candidate, re.IGNORECASE)
            or re.search(r"\b(?:may|might|could)\b", candidate)
            or re.search(r"\b(?:peut|peuvent)\b", candidate, re.IGNORECASE)
        )
        if (labelled is not None or implicit) and not _OBLIGATION.search(candidate):
            _append_unique(risks, risk_citations, candidate, _citation(chunk, line))

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
        effective_date=EvidenceValue(
            effective_values[0] if effective_values else None,
            (effective_evidence[0],) if effective_evidence else (),
        ),
        renewal_date=EvidenceValue(renewal_date, selected_renewal_citations),
        important_amounts=EvidenceValue(tuple(amount_values), tuple(amount_citations)),
        obligations=EvidenceValue(tuple(obligations), tuple(obligation_citations)),
        responsible_people=EvidenceValue(tuple(people), tuple(people_citations)),
        risks=EvidenceValue(tuple(risks), tuple(risk_citations)),
    )
