import re
from dataclasses import dataclass
from typing import Any

from evidencedesk_api.providers import EmbeddingProvider
from evidencedesk_api.retrieval import Citation, EvidenceChunk
from evidencedesk_api.trust_boundaries import (
    is_untrusted_document_instruction,
    untrusted_document_fragment_indexes,
)


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
    entries: list[tuple[EvidenceChunk, str]] = []
    for chunk in chunks:
        lines = [
            line.strip()
            for line in chunk.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        blocked_lines = untrusted_document_fragment_indexes(lines)
        entries.extend(
            (chunk, line) for index, line in enumerate(lines) if index not in blocked_lines
        )
    return entries


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
    entries = [
        item
        for item in _line_entries(chunks)
        if not is_untrusted_document_instruction(item[1])
    ]
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


_DATE_V3 = re.compile(
    rf"\b(?:\d{{4}}[-.]\d{{2}}[-.]\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|"
    rf"(?:\d{{1,2}}(?:er)?|premier)\s+(?:{_MONTHS})\s+\d{{4}}|"
    rf"(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}})\b",
    re.IGNORECASE,
)
_AMOUNT_V3 = re.compile(
    r"(?:\b(?:EUR|USD|GBP)\s*[$€£]?\s*\d[\d ,.]*\d(?:[.,]\d+)?k?\b|"
    r"[$€£]\s*\d[\d ,.]*\d(?:[.,]\d+)?k?(?:\s*(?:EUR|USD|GBP))?\b|"
    r"\b\d[\d ,.]*\d(?:[.,]\d+)?k?\s*(?:EUR|USD|GBP|€|£|\$)\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|"
    r"four|five|six|seven|eight|nine))?\s+thousand\s+(?:pounds?|euros?|dollars?)\b)",
    re.IGNORECASE,
)
_PERSON_V3 = re.compile(
    r"\b([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\u2019-]+"
    r"(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\u2019-]+){1,2})\b"
)
_ROLE_VERBS_V3 = re.compile(
    r"\b(?:coordinates?|owns?|leads?|led|maintains?|maintained|handles?|approves?|investigates?|"
    r"validates?|arbitre|anime|coordonne|pilote|suit|valide|tient|autorise|"
    r"responsable|stewards?)\b",
    re.IGNORECASE,
)
_OBLIGATION_VERBS_V3 = re.compile(
    r"\b(?:keep|keeps|preserve|preserves|archive|archives|distribute|distributes|"
    r"circulate|circulates|send|sends|reopen|reopens|photograph|photographs|"
    r"transmit|transmits|conserve|conservent|informe|informent|adresse|adressent)\b",
    re.IGNORECASE,
)
_RISK_V3 = re.compile(
    r"\b(?:risk|risque|single|lone|unique|absence|dependenc|dépend|could|might|may|"
    r"peut|pourrait|delay|retard|disable|isolate|interrompre|affect)\w*\b",
    re.IGNORECASE,
)
def _document_type_v3(
    entries: list[tuple[EvidenceChunk, str]],
) -> tuple[str | None, tuple[Citation, ...]]:
    kind_words = re.compile(
        r"\b(?:agreement|contract|note|file|plan|chronicle|catalogue|playbook|letter|"
        r"schedule|report|accord|contrat|dossier|lettre|compte rendu)\b",
        re.IGNORECASE,
    )
    for chunk, line in entries:
        if chunk.page != 1 or line != line.upper() or not kind_words.search(line):
            continue
        candidates = [part.strip() for part in re.split(r"\s+[—/]\s+", line)]
        typed = [part for part in candidates if kind_words.search(part)]
        if typed:
            value = min(typed, key=lambda item: len(item.split()))
            return value, (_citation(chunk, line),)
    return _first_label_value(
        entries,
        ("Document type", "Document kind", "Category", "Catégorie", "Nature du document"),
    )


def _organization_v3(
    entries: list[tuple[EvidenceChunk, str]],
) -> tuple[str | None, tuple[Citation, ...]]:
    labelled = _first_label_value(
        entries,
        (
            "Organization",
            "Organisation",
            "Supplier",
            "Fournisseur",
            "Vendor legal entity",
            "Account holder",
            "Plan owner organization",
            "Operational beneficiary",
        ),
    )
    if labelled[0] is not None:
        return labelled
    patterns = (
        r"(?:operator named for .+?|custodian covered .+?|beneficiary operationnel)"
        r"\s+(?:is|est)\s+([^.;]+)",
        r"\b(?:dossier concerne|issued by)\s+([^.;]+?)(?=\s+(?:et non|for)\b|[.;])",
        r"\bconfirme que\s+([^,.;]+?)\s+(?:porte|gère|gere)\b",
        r"\b(?:account holder|plan owner organization)\s*(?:—|:)\s*([^.;]+)",
    )
    for chunk, line in entries:
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip(), (_citation(chunk, line),)
        legal_entity = _LEGAL_ENTITY.search(line)
        if chunk.page == 1 and legal_entity:
            return legal_entity.group(0).strip(), (_citation(chunk, line),)
    return None, ()


def _context_date_v3(
    entries: list[tuple[EvidenceChunk, str]], pattern: re.Pattern[str], *, last: bool = False
) -> tuple[str | None, tuple[Citation, ...]]:
    for chunk, line in entries:
        if pattern.search(line):
            matches = list(_DATE_V3.finditer(line))
            if matches:
                return matches[-1 if last else 0].group(0), (_citation(chunk, line),)
    return None, ()


def _labelled_or_context_date_v3(
    entries: list[tuple[EvidenceChunk, str]],
    *,
    labels: tuple[str, ...],
    context_pattern: re.Pattern[str],
    last: bool = False,
) -> tuple[str | None, tuple[Citation, ...]]:
    labelled_value, labelled_citations = _first_label_value(entries, labels)
    if labelled_value is not None:
        matches = list(_DATE_V3.finditer(labelled_value))
        if matches:
            return matches[-1 if last else 0].group(0), labelled_citations
    return _context_date_v3(entries, context_pattern, last=last)


def _obligation_clauses_v3(line: str) -> list[str]:
    candidate = re.sub(
        r"^(?:during|after|before|at|à)\b[^,]*,\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"^.+?\bqui\s+", "", candidate, flags=re.IGNORECASE)
    parts = [part.strip() for part in re.split(r"\s+(?:and|et)\s+", candidate)]
    return [part for part in parts if _OBLIGATION_VERBS_V3.search(part)]


def extract_supplier_fields_v3(
    chunks: list[EvidenceChunk], *, embeddings: EmbeddingProvider
) -> SupplierExtraction:
    """Extract source-linked fields from relational prose and explicit field labels.

    Embeddings are part of the stable provider contract; this deterministic strategy
    uses auditable lexical relations and leaves semantic inference to the bounded
    NLI/JSON candidates.
    """

    del embeddings
    entries = [
        item
        for item in _line_entries(chunks)
        if not is_untrusted_document_instruction(item[1])
    ]
    organization, organization_citations = _organization_v3(entries)
    document_type, document_type_citations = _document_type_v3(entries)
    effective, effective_citations = _labelled_or_context_date_v3(
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
            r"\b(?:begin|begins|start|starts|takes? effect|went live|activation|applicable|"
            r"custody passes|prendre effet|prendra effet|devient applicable|ouvre la période|"
            r"ouvre la periode|changent? de garde)\b",
            re.IGNORECASE,
        ),
    )
    renewal, renewal_citations = _labelled_or_context_date_v3(
        entries,
        labels=("Renewal date", "Renouvellement", "Prochaine échéance de renouvellement"),
        context_pattern=re.compile(
            r"\b(?:next .+ review|annual review|reviewed again|anniversary|annual reset|"
            r"renew\w*|revue annuelle|examen annuel|prochain état|prochain etat)\b",
            re.IGNORECASE,
        ),
        last=True,
    )

    amounts: list[str] = []
    amount_citations: list[Citation] = []
    people: list[str] = []
    people_citations: list[Citation] = []
    obligations: list[str] = []
    obligation_citations: list[Citation] = []
    risks: list[str] = []
    risk_citations: list[Citation] = []
    person_aliases = (
        "Responsible manager",
        "Responsible contact",
        "Service owner",
        "Owner",
        "Accountable lead",
        "Contact opérationnel",
        "Responsable",
        "Responsable du compte",
    )
    for chunk, line in entries:
        citation = _citation(chunk, line)
        for match in _AMOUNT_V3.finditer(line):
            _append_unique(amounts, amount_citations, match.group(0).strip(), citation)
        labelled_person = _label_match(line, person_aliases)
        if labelled_person is not None:
            person = labelled_person.split(",", maxsplit=1)[0].strip().removesuffix(".")
            _append_unique(people, people_citations, person, citation)
        elif _ROLE_VERBS_V3.search(line):
            for match in _PERSON_V3.finditer(line):
                tail = line[match.end() :]
                if _ROLE_VERBS_V3.search(tail[:40]) or re.search(
                    r"\b(?:stewards?|responsables?)\b", line[: match.start()], re.IGNORECASE
                ):
                    _append_unique(people, people_citations, match.group(1), citation)
        modal_obligation = _OBLIGATION.search(line)
        lexical_obligation = _OBLIGATION_VERBS_V3.search(line)
        if modal_obligation or (lexical_obligation and not _RISK_V3.search(line)):
            clauses = [line] if modal_obligation else _obligation_clauses_v3(line)
            for clause in clauses:
                _append_unique(obligations, obligation_citations, clause, citation)
        if _RISK_V3.search(line) and not modal_obligation and not lexical_obligation:
            _append_unique(risks, risk_citations, line, citation)

    return SupplierExtraction(
        organization_name=EvidenceValue(organization, organization_citations),
        document_type=EvidenceValue(document_type, document_type_citations),
        effective_date=EvidenceValue(effective, effective_citations),
        renewal_date=EvidenceValue(renewal, renewal_citations),
        important_amounts=EvidenceValue(tuple(amounts), tuple(amount_citations)),
        obligations=EvidenceValue(tuple(obligations), tuple(obligation_citations)),
        responsible_people=EvidenceValue(tuple(people), tuple(people_citations)),
        risks=EvidenceValue(tuple(risks), tuple(risk_citations)),
    )
