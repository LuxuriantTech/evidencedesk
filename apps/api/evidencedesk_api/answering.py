"""Grounded, deterministic answer decisions for the no-key EvidenceDesk mode.

The module deliberately separates four outcomes: supported, partially supported,
ambiguous, and absent evidence.  Document text is always data, never an instruction.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from evidencedesk_api.providers import EmbeddingProvider
from evidencedesk_api.retrieval import Citation, RankedChunk, lexical_tokens
from evidencedesk_api.trust_boundaries import (
    TrustSeparatedInput,
    UntrustedDocumentContent,
    admit_evidence_excerpt,
    is_untrusted_document_instruction,
    untrusted_document_fragment_indexes,
)

AnswerStatus = Literal["answered", "partially_supported", "ambiguous", "abstained"]


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    support_threshold: float
    partial_support_threshold: float
    contradiction_margin: float

    def __post_init__(self) -> None:
        values = (
            self.support_threshold,
            self.partial_support_threshold,
            self.contradiction_margin,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("decision thresholds must be finite values between zero and one")


@dataclass(frozen=True, slots=True)
class PassageCandidateAssessment:
    answerable: bool
    answer: str | None
    confidence: float
    supporting_document: str | None
    supporting_page: int | None
    supporting_excerpt: str | None
    ambiguity_reason: str | None
    extracted_fields: dict[str, str | tuple[str, ...]]
    supporting_chunk_id: str


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    status: AnswerStatus
    answerable: bool
    answer: str
    confidence: float
    supporting_document: str | None
    supporting_page: int | None
    supporting_excerpt: str | None
    ambiguity_reason: str | None
    extracted_fields: dict[str, str | tuple[str, ...]]
    supporting_chunk_ids: tuple[str, ...]
    citations: tuple[Citation, ...] = ()
    candidate_assessments: tuple[PassageCandidateAssessment, ...] = ()


class GroundingValidationError(ValueError):
    """Raised when a structured answer is not grounded in the supplied chunks."""


_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|October|November|"
    r"December|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre"
)
_DATE = re.compile(
    rf"\b(?:\d{{4}}[-.]\d{{2}}[-.]\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|"
    rf"(?:\d{{1,2}}(?:er)?|premier)\s+(?:{_MONTHS})\s+\d{{4}}|"
    rf"(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}})\b",
    re.IGNORECASE,
)
_AMOUNT = re.compile(
    r"(?:\b(?:EUR|USD|GBP)\s*[$€£]?\s*\d[\d ,.]*\d(?:[.,]\d+)?k?(?=$|[^\w])|"
    r"[$€£]\s*\d[\d ,.]*\d(?:[.,]\d+)?k?(?:\s*(?:EUR|USD|GBP))?(?=$|[^\w])|"
    r"\b\d[\d ,.]*\d(?:[.,]\d+)?k?\s*(?:EUR|USD|GBP|€|£|\$)(?=$|[^\w])|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|"
    r"four|five|six|seven|eight|nine))?\s+thousand\s+(?:pounds?|euros?|dollars?)\b)",
    re.IGNORECASE,
)
_DURATION = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|un|une|deux|"
    r"trois|quatre|cinq|six|sept|huit|neuf|dix)\s+"
    r"(?:minutes?|hours?|days?|heures?|jours?)\b",
    re.IGNORECASE,
)
_SEVERITY = re.compile(
    r"\b(?:SEV[- ]?\d+|severity\s+[a-z]+|sévérité\s+[a-z0-9-]+)\b",
    re.IGNORECASE,
)
_PERSON = re.compile(
    r"\b([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\u2019-]+"
    r"(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\u2019-]+){1,2})\b"
)

_QUESTION_NOISE = {
    "a",
    "an",
    "are",
    "account",
    "arrangement",
    "catalogue",
    "chronicle",
    "covered",
    "dossier",
    "does",
    "do",
    "did",
    "document",
    "for",
    "has",
    "have",
    "how",
    "is",
    "it",
    "file",
    "of",
    "on",
    "plan",
    "quelle",
    "quelles",
    "this",
    "under",
    "was",
    "what",
    "when",
    "which",
    "whose",
    "with",
}

_CAPITALIZED_QUERY_STOPWORDS = {
    "a", "an", "can", "de", "does", "how", "if", "is", "le", "la",
    "on", "the", "was", "what", "when", "which", "whose", "who", "where",
    "why", "may", "might", "will", "would", "qui", "quand", "comment", "pourquoi",
    "ou", "les", "un",
    "une", "des", "du", "au", "aux", "quel", "quelle", "quels", "quelles",
}

_DOCUMENT_TYPE_TERMS = {
    "agreement": {"agreement", "contract", "contrat"},
    "addendum": {"addendum", "annex", "amendment"},
    "report": {"report"},
    "register": {"register"},
}

_RELATION_EQUIVALENTS = {
    "reserve": {"reserve", "reserves", "fund", "funds", "fond", "fonds", "pool"},
    "maintenance": {"maintenance", "entretien"},
    "coverage": {"coverage", "cover", "couverture"},
    "precedence": {
        "control", "controls", "govern", "governs", "prevail", "prevails",
        "precedence", "priority", "replace", "replaces", "supersede", "supersedes",
    },
}

_ABSENCE = re.compile(
    r"\b(?:not\s+(?:provided|recorded|listed|available|specified|stated|supplied)|"
    r"(?:unavailable|absent|missing|indisponible|absent|manquant)\b|"
    r"no\s+(?:date|amount|value|number|record|entry)\s+(?:is\s+)?(?:provided|recorded|listed)|"
    r"unknown|unrecorded|n/?a|none|non\s+(?:renseigne|enregistre|indique)|"
    r"aucun\s+(?:enregistrement|montant|numero|date))\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|cannot|can't|mustn't|non|pas|jamais|aucun)\b",
    re.IGNORECASE,
)
_SCALAR_ALTERNATIVE = re.compile(
    r"\b(?:or|ou)\b.*\b(?:\d|minutes?|hours?|days?|heures?|jours?|EUR|USD|GBP)\b",
    re.IGNORECASE,
)


def _relation_tokens(text: str) -> set[str]:
    tokens = set(lexical_tokens(text))
    for equivalents in _RELATION_EQUIVALENTS.values():
        if tokens & equivalents:
            tokens.add(next(iter(sorted(equivalents))))
    return tokens


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()


def _requested_document_type(question: str) -> str | None:
    folded = _fold(question)
    if re.search(r"\b(?:agreements|contracts|reports|addenda|annexes)\b", folded):
        return None
    occurrences = [
        (match.start(), kind)
        for kind, terms in _DOCUMENT_TYPE_TERMS.items()
        for term in terms
        if (match := re.search(rf"\b{re.escape(term)}\b", folded)) is not None
    ]
    return min(occurrences)[1] if occurrences else None


def _document_type(item: RankedChunk, contents: str) -> str | None:
    filename = _fold(item.chunk.document_name)
    if re.search(r"(?:^|[_\W])(?:addendum|annex|amendment|dpa)(?:$|[_\W])", filename):
        return "addendum"
    if re.search(r"(?:^|[_\W])(?:agreement|contract|contrat|msa)(?:$|[_\W])", filename):
        return "agreement"
    if re.search(r"(?:^|[_\W])register(?:$|[_\W])", filename):
        return "register"
    if re.search(r"(?:^|[_\W])report(?:$|[_\W])", filename):
        return "report"
    # The first line of the per-document context is its filename; titles begin
    # in the following source line.
    lines = contents.splitlines()
    title = _fold(lines[1] if len(lines) > 1 else "")
    if "addendum" in title or "annex" in title:
        return "addendum"
    if "agreement" in title or "contract" in title or "contrat" in title:
        return "agreement"
    if "register" in title:
        return "register"
    if "report" in title:
        return "report"
    return None


def _named_subject_tokens(question: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.finditer(r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'-]+\b", question):
        raw = match.group(0)
        token = re.sub(r"(?:'s|s')$", "", _fold(raw))
        if (raw.isupper() and len(raw) < 2) or token in _CAPITALIZED_QUERY_STOPWORDS:
            continue
        if token in set.union(*_DOCUMENT_TYPE_TERMS.values()):
            continue
        tokens.add(token)
    return tokens


def _scope_ranked(question: str, ranked: list[RankedChunk]) -> list[RankedChunk]:
    requested_type = _requested_document_type(question)
    subject_tokens = _named_subject_tokens(question)
    if requested_type is None and not subject_tokens:
        return ranked

    grouped: dict[str, list[RankedChunk]] = {}
    for item in ranked:
        grouped.setdefault(item.chunk.document_id, []).append(item)
    context_by_document = {
        document_id: "\n".join(
            [items[0].chunk.document_name]
            + [item.chunk.text for item in sorted(items, key=lambda item: item.chunk.id)]
        )
        for document_id, items in grouped.items()
    }
    allowed: set[str] = set()
    for document_id, items in grouped.items():
        contents = context_by_document[document_id]
        if requested_type is not None and _document_type(items[0], contents) != requested_type:
            continue
        context_tokens = _relation_tokens(contents)
        if subject_tokens:
            overlap = subject_tokens & context_tokens
            if len(overlap) < max(1, (len(subject_tokens) * 2 + 2) // 3):
                continue
        allowed.add(document_id)
    if not allowed:
        return []
    return [item for item in ranked if item.chunk.document_id in allowed]


def _has_explicit_absence(segment: str) -> bool:
    return _ABSENCE.search(_fold(segment)) is not None


def _has_negated_relation(question: str, segment: str, kind: str) -> bool:
    """Reject a value when its nearby relation is explicitly negated."""
    folded_segment = _fold(segment)
    relation_tokens = _question_relation_tokens(question, kind)
    if kind == "date":
        role = _date_role(question)
        relation_tokens |= {
            "effective" if role == "effective" else "renewal" if role == "renewal" else ""
        }
    relation_tokens.discard("")
    if not relation_tokens:
        return False
    clauses = re.split(
        r"[;,]|\b(?:but|however|whereas|while|mais|cependant|tandis que)\b",
        folded_segment,
    )
    negation = r"\b(?:not|never|no|without|cannot|can't|mustn't|pas|jamais|aucun)\b"
    for clause in clauses:
        clause_tokens = set(re.findall(r"[a-z0-9']+", clause))
        if relation_tokens & clause_tokens and re.search(negation, clause):
            return True
    return False


def _asks_past_event_date(question: str) -> bool:
    folded = _fold(question)
    return bool(
        re.search(r"\b(?:on\s+what\s+date|when|a\s+quelle\s+date)\b", folded)
        and re.search(r"\b(?:did|has|was|a-t-il|a-t-elle)\b", folded)
        and re.search(r"\b(?:notify|notifie|send|sent|occur|happen|report|declare)\w*\b", folded)
    )


def _question_relation_tokens(question: str, kind: str) -> set[str]:
    noise_by_kind = {
        "date": {"date", "effective", "deadline"},
        "amount": {"amount", "cost", "charge", "fee", "price", "tariff", "budget", "apply"},
        "duration": {"duration", "deadline", "period"},
        "severity": {"severity"},
        "responsible_people": {"owner"},
        "organization_name": set(),
        "boolean": {
            "is", "are", "was", "were", "does", "do", "did", "can", "could",
            "may", "might", "will", "would", "has", "have",
        },
        "text": set(),
    }
    tokens = _relation_tokens(question) - _QUESTION_NOISE - noise_by_kind.get(kind, set())
    tokens -= _named_subject_tokens(question)
    requested_type = _requested_document_type(question)
    if requested_type is not None:
        tokens -= _DOCUMENT_TYPE_TERMS[requested_type]
    return tokens


def _unsafe_question(question: str) -> bool:
    folded = _fold(question)
    return bool(
        re.search(
            r"\b(?:ignore|ignorez|disregard|invent|revele|reveal|contourne|recupere|override)\b",
            folded,
        )
        and re.search(
            r"\b(?:source\w*|citation\w*|controle\w*|coordonne\w*|masked|masqu\w*|"
            r"system|prompt|code)\b",
            folded,
        )
    )


def _segments(text: str) -> list[str]:
    result: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("•").strip()
        if not stripped:
            continue
        result.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])", stripped)
            if part.strip()
        )
    return result


def _question_kind(question: str) -> str:
    folded = _fold(question)
    yes_no_prefix = bool(
        re.match(
            r"\s*(?:is|are|was|were|does|do|did|can|could|may|might|will|would|has|have|"
            r"est-ce\s+que)\b",
            folded,
        )
    )
    if yes_no_prefix and not _SCALAR_ALTERNATIVE.search(folded):
        return "boolean"
    if re.search(
        r"\b(?:how\s+much\s+time|how\s+long|how\s+quickly|how\s+many\s+"
        r"(?:minutes?|hours?|days?)|duration|duree|"
        r"combien\s+de\s+temps|delai|notification\s+period|deadline)\b",
        folded,
    ):
        return "duration"
    if re.search(
        r"\b(?:when|quand|date|milestone|effect|applicable|commence|start|ouvre)\b",
        folded,
    ):
        return "date"
    if re.search(
        r"\b(?:how much|amount|budget|cost|charge|fee|tariff|price|prix|montant|"
        r"fonds|reserve|enveloppe)\b",
        folded,
    ):
        return "amount"
    if re.search(r"\b(?:severity|severite|sev)\b", folded):
        return "severity"
    if re.search(
        r"\b(?:organization|organisation|entity|entite|supplier|custodian|mutuelle|"
        r"beneficiary|beneficiaire|account holder|operator|issued)\b",
        folded,
    ) or "whose account" in folded:
        return "organization_name"
    if re.search(r"\b(?:who|qui|whose)\b", folded):
        return "responsible_people"
    return "text"


def _date_role(text: str) -> str | None:
    """Return a requested/evidenced date role when it is stated explicitly."""
    folded = _fold(text)
    renewal = bool(re.search(r"\b(?:renew|renewal|renouvel|recondu|echeance)\w*\b", folded))
    effective = bool(
        re.search(
            r"\b(?:effective|takes? effect|effective from|became effective|"
            r"commence(?:ment)?|starts?|application|vigueur|prise d[' ]?effet|"
        r"prend\w*(?:-t-(?:il|elle)|-il|-elle)?\s+effet|prendre\s+effet|"
            r"date d[' ]?effet|date effet)\w*\b",
            folded,
        )
    )
    if renewal == effective:
        return None
    return "renewal" if renewal else "effective"


def _date_match_for_question(segment: str, question: str) -> re.Match[str] | None:
    matches = list(_DATE.finditer(segment))
    if not matches:
        return None
    requested_role = _date_role(question)
    if requested_role is None or len(matches) == 1:
        return matches[0]
    for match in matches:
        if _date_role(_date_context(segment, match)) == requested_role:
            return match
    return matches[0]


def _date_context_bounds(segment: str, match: re.Match[str]) -> tuple[int, int]:
    """Return the local clause bounds for one date among possible date matches."""
    matches = list(_DATE.finditer(segment))
    index = next((i for i, item in enumerate(matches) if item.start() == match.start()), 0)
    start_candidates = [
        segment.rfind(delimiter, 0, match.start()) + 1
        for delimiter in (";", "\n", ",")
    ]
    end_candidates = [
        position
        for delimiter in (";", "\n", ",")
        if (position := segment.find(delimiter, match.end())) >= 0
    ]
    if index > 0:
        start_candidates.append(matches[index - 1].end())
    if index + 1 < len(matches):
        end_candidates.append(matches[index + 1].start())
    return max(start_candidates), min(end_candidates) if end_candidates else len(segment)


def _date_context(segment: str, match: re.Match[str]) -> str:
    start, end = _date_context_bounds(segment, match)
    return segment[start:end]


def _date_evidence_segments(segment: str, question: str) -> list[str]:
    matches = list(_DATE.finditer(segment))
    if len(matches) < 2:
        return [segment]
    requested_role = _date_role(question)
    clauses = [
        context
        for match in matches
        if (context := _date_context(segment, match).strip())
        and (requested_role is None or _date_role(context) == requested_role)
    ]
    return clauses or [segment]


def _value_for(kind: str, segment: str, question: str) -> str | None:
    if kind == "boolean":
        if not _relation_supported(question, segment, kind):
            return None
        if _has_explicit_absence(segment) or _has_negated_relation(question, segment, kind):
            return f"No. {segment}"
        question_amounts = list(_AMOUNT.finditer(question))
        question_dates = list(_DATE.finditer(question))
        question_durations = list(_DURATION.finditer(question))
        for requested, observed in (
            (question_amounts, list(_AMOUNT.finditer(segment))),
            (question_dates, list(_DATE.finditer(segment))),
            (question_durations, list(_DURATION.finditer(segment))),
        ):
            if requested:
                requested_values = {_fold(match.group(0)) for match in requested}
                observed_values = {_fold(match.group(0)) for match in observed}
                if observed_values and not requested_values & observed_values:
                    return f"No. {segment}"
        return f"Yes. {segment}"
    if kind == "date":
        match = _date_match_for_question(segment, question)
        if match is None:
            return None
        value = match.group(0)
        if re.search(r"\b(?:time|heure)\b", _fold(question)):
            context_start, context_end = _date_context_bounds(segment, match)
            context = segment[context_start:context_end]
            clock_matches = list(
                re.finditer(
                    r"\b(?:at|a|à)\s*(\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?)",
                    context,
                    re.IGNORECASE,
                )
            )
            if len(clock_matches) == 1:
                clock = clock_matches[0]
                start = min(match.start() - context_start, clock.start())
                end = max(match.end() - context_start, clock.end())
                value = context[start:end].strip()
        return value
    if kind == "amount":
        matches = list(_AMOUNT.finditer(segment))
        if not matches:
            return None
        question_relation_tokens = _relation_tokens(question)
        if len(matches) == 1:
            return matches[0].group(0).strip()
        clauses = re.split(r";|,(?=\s*[A-Za-zÀ-ÖØ-öø-ÿ])", segment)
        best = max(
            clauses,
            key=lambda clause: len(question_relation_tokens & _relation_tokens(clause)),
        )
        match = _AMOUNT.search(best)
        return (match or matches[0]).group(0).strip()
    if kind == "duration":
        match = _DURATION.search(segment)
        return match.group(0) if match else None
    if kind == "severity":
        match = _SEVERITY.search(segment)
        return match.group(0) if match else None
    if kind == "responsible_people":
        question_tokens = lexical_tokens(question) - _QUESTION_NOISE
        clauses = [part.strip() for part in re.split(r";|\b(?:and|et|while)\b", segment)]
        clauses.sort(
            key=lambda clause: len(question_tokens & lexical_tokens(clause)),
            reverse=True,
        )
        for clause in clauses:
            person = _PERSON.search(clause)
            if person:
                return person.group(1)
        return None
    if kind == "organization_name":
        name = (
            r"([A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'\u2019-]+"
            r"(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'\u2019-]+){1,4})"
        )
        patterns = (
            rf"(?i:\b(?:issued by|émis par|emis par))\s+{name}",
            rf"(?i:\b(?:dossier concerne))\s+{name}",
            rf"(?i:\bconfirme que)\s+{name}(?i:\s+(?:porte|gère|gere)\b)",
            rf"(?i:\b(?:holder|custodian|operator|organisation|organization|entité|"
            rf"entity|beneficiary|bénéficiaire)[^.;]{{0,60}}?(?:—|:|\bis\b|\best\b))\s+{name}",
        )
        for pattern in patterns:
            match = re.search(pattern, segment)
            if match:
                return match.group(1).strip().removesuffix(".")
        return None
    return segment


def _semantic_similarity(left: str, right: str, provider: EmbeddingProvider) -> float:
    left_vector, right_vector = provider.embed_many([left, right])
    value = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _relation_categories(text: str) -> frozenset[str]:
    folded = _fold(text)
    categories: dict[str, str] = {
        "operate": r"\boperat\w*\b",
        "issue": r"\b(?:issu\w*|emis\w*)\b",
        "concern": r"\bconcern\w*\b",
        "account": r"\b(?:account|holder)\b",
        "custody": r"\bcustod\w*\b",
        "owner": r"\b(?:own\w*|porte)\b",
        "beneficiary": r"\bbenefici\w*\b",
        "mutual": r"\bmutu\w*\b",
    }
    return frozenset(key for key, pattern in categories.items() if re.search(pattern, folded))


def _relation_supported(question: str, segment: str, kind: str) -> bool:
    question_tokens = _question_relation_tokens(question, kind)
    segment_tokens = _relation_tokens(segment)
    if kind == "date":
        requested_role = _date_role(question)
        if requested_role is not None:
            match = _date_match_for_question(segment, question)
            if match is None or _date_role(_date_context(segment, match)) != requested_role:
                return False
            return True
    if kind == "organization_name":
        question_categories = _relation_categories(question)
        if question_categories:
            return bool(question_categories & _relation_categories(segment))
        return bool(
            re.search(r"\b(?:organization|organisation|entity|entite|name|nom)\b", _fold(question))
        )
    noise_by_kind = {
        "date": {"date", "effective", "deadline"},
        "amount": {"amount", "cost", "charge", "fee", "price", "tariff", "budget", "apply"},
        "duration": {"duration", "deadline"},
        "severity": {"severity"},
        "responsible_people": {"owner"},
        "boolean": set(),
        "text": set(),
    }
    relation_tokens = question_tokens - noise_by_kind.get(kind, set())
    overlap = relation_tokens & segment_tokens
    if kind == "amount":
        return bool(overlap)
    if kind == "date" and requested_role is None:
        return bool(relation_tokens) and len(overlap) >= min(2, len(relation_tokens))
    if kind == "boolean":
        return len(overlap) >= 2 or _has_explicit_absence(segment) or _has_negated_relation(
            question, segment, kind
        )
    if kind == "text":
        ordered: list[str] = []
        for raw in re.findall(r"[a-z0-9]+", _fold(question)):
            for token in lexical_tokens(raw):
                if token not in _QUESTION_NOISE and token not in ordered:
                    ordered.append(token)
        return len(overlap) >= 2 and bool(set(ordered[:2]) & segment_tokens)
    if kind == "duration" and re.search(r"\b(?:duration|duree)\b", _fold(segment)):
        return True
    if kind == "severity" and _SEVERITY.search(segment):
        return True
    return bool(overlap)


def _support_score(question: str, segment: str, kind: str, provider: EmbeddingProvider) -> float:
    query_tokens = lexical_tokens(question) - _QUESTION_NOISE
    segment_tokens = lexical_tokens(segment)
    lexical = len(query_tokens & segment_tokens) / max(1, len(query_tokens))
    semantic = _semantic_similarity(question, segment, provider)
    has_value = _value_for(kind, segment, question) is not None
    relation = float(_relation_supported(question, segment, kind))
    return min(1.0, 0.30 * lexical + 0.20 * semantic + 0.35 * float(has_value) + 0.15 * relation)


def _has_explicit_relation(question: str, segment: str) -> bool:
    return _relation_supported(question, segment, _question_kind(question))


def _citation(item: RankedChunk, excerpt: str) -> Citation:
    chunk = item.chunk
    return Citation(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        page=chunk.page,
        section=chunk.section,
        excerpt=excerpt,
    )


def _field_name(kind: str) -> str:
    return {
        "boolean": "answer",
        "date": "date",
        "amount": "important_amounts",
        "duration": "duration",
        "severity": "severity",
        "responsible_people": "responsible_people",
        "organization_name": "organization_name",
        "text": "answer",
    }[kind]


def _abstained(
    message: str = "Insufficient evidence in the selected dossier.",
    *,
    candidate_assessments: tuple[PassageCandidateAssessment, ...] = (),
) -> GroundedAnswer:
    return GroundedAnswer(
        status="abstained",
        answerable=False,
        answer=message,
        confidence=0.0,
        supporting_document=None,
        supporting_page=None,
        supporting_excerpt=None,
        ambiguity_reason=None,
        extracted_fields={},
        supporting_chunk_ids=(),
        candidate_assessments=candidate_assessments,
    )


def validate_grounded_answer(answer: GroundedAnswer, ranked: list[RankedChunk]) -> None:
    if not math.isfinite(answer.confidence) or not 0.0 <= answer.confidence <= 1.0:
        raise GroundingValidationError("confidence must be finite and between zero and one")
    if answer.status == "answered":
        if not answer.answerable:
            raise GroundingValidationError("answered result must be answerable")
        if answer.ambiguity_reason is not None:
            raise GroundingValidationError("answered result cannot have an ambiguity reason")
    elif answer.status in {"partially_supported", "ambiguous", "abstained"}:
        if answer.answerable:
            raise GroundingValidationError("non-answered result cannot be answerable")
        if answer.status in {"partially_supported", "ambiguous"} and not answer.ambiguity_reason:
            raise GroundingValidationError("supported uncertainty requires an ambiguity reason")
    else:
        raise GroundingValidationError("answer status is invalid")
    by_id = {item.chunk.id: item.chunk for item in ranked}
    if any(identifier not in by_id for identifier in answer.supporting_chunk_ids):
        raise GroundingValidationError("supporting chunk does not exist")
    unsafe_channels = [answer.answer]
    if answer.supporting_excerpt is not None:
        unsafe_channels.append(answer.supporting_excerpt)
    unsafe_channels.extend(citation.excerpt for citation in answer.citations)
    for value in answer.extracted_fields.values():
        unsafe_channels.extend((value,) if isinstance(value, str) else value)
    for assessment in answer.candidate_assessments:
        if assessment.answer is not None:
            unsafe_channels.append(assessment.answer)
        if assessment.supporting_excerpt is not None:
            unsafe_channels.append(assessment.supporting_excerpt)
        for value in assessment.extracted_fields.values():
            unsafe_channels.extend((value,) if isinstance(value, str) else value)
    if any(is_untrusted_document_instruction(value) for value in unsafe_channels):
        raise GroundingValidationError(
            "untrusted document instruction cannot cross evidence boundary"
        )
    if answer.supporting_excerpt is None:
        if answer.supporting_document is not None or answer.supporting_page is not None:
            raise GroundingValidationError("support location requires an excerpt")
        if answer.status != "abstained":
            raise GroundingValidationError(f"{answer.status} result requires complete grounding")
        if (
            answer.ambiguity_reason is not None
            or answer.extracted_fields
            or answer.supporting_chunk_ids
            or answer.citations
        ):
            raise GroundingValidationError("abstained result cannot contain supporting evidence")
        return
    if answer.status == "abstained":
        raise GroundingValidationError("abstained result cannot contain supporting evidence")
    if answer.supporting_document is None or answer.supporting_page is None:
        raise GroundingValidationError("supporting excerpt requires document and page")
    matching = [
        chunk
        for chunk in by_id.values()
        if chunk.document_id == answer.supporting_document
        and chunk.page == answer.supporting_page
    ]
    if not matching:
        if not any(chunk.document_id == answer.supporting_document for chunk in by_id.values()):
            raise GroundingValidationError("supporting document does not exist")
        raise GroundingValidationError("supporting page does not exist")
    if not any(answer.supporting_excerpt in chunk.text for chunk in matching):
        raise GroundingValidationError("supporting excerpt is not present on the cited page")
    if any(
        answer.supporting_excerpt in fragment
        for chunk in matching
        for index, fragment in enumerate(_segments(chunk.text))
        if index in untrusted_document_fragment_indexes(_segments(chunk.text))
    ):
        raise GroundingValidationError(
            "untrusted document instruction cannot cross evidence boundary"
        )
    if answer.status == "answered" and _fold(answer.answer) not in _fold(
        answer.supporting_excerpt
    ):
        boolean_supported = (
            answer.answer.startswith("Yes. ")
            and answer.answer[5:] == answer.supporting_excerpt
        ) or (
            answer.answer.startswith("No. ")
            and answer.answer[4:] == answer.supporting_excerpt
        )
        if not boolean_supported:
            raise GroundingValidationError("answer is not supported by the cited excerpt")
    folded_excerpt = _fold(answer.supporting_excerpt)
    for field, value in answer.extracted_fields.items():
        if not field or not isinstance(field, str):
            raise GroundingValidationError("extracted field name is invalid")
        values = (value,) if isinstance(value, str) else value
        if not isinstance(values, tuple) or not values:
            raise GroundingValidationError(f"extracted field {field} has an invalid value")
        if any(
            not isinstance(item, str) or not item or _fold(item) not in folded_excerpt
            for item in values
        ):
            raise GroundingValidationError(
                f"extracted field {field} is not supported by the cited excerpt"
            )
    if not any(
        chunk.id in answer.supporting_chunk_ids and answer.supporting_excerpt in chunk.text
        for chunk in matching
    ):
        raise GroundingValidationError("supporting chunk must identify the cited passage")
    for citation in answer.citations:
        chunk = by_id.get(citation.chunk_id)
        if (
            chunk is None
            or citation.document_id != chunk.document_id
            or citation.page != chunk.page
            or citation.document_name != chunk.document_name
            or citation.excerpt not in chunk.text
        ):
            raise GroundingValidationError("citation excerpt is not grounded")


class DeterministicGroundedAnswerProvider:
    """A local, deterministic evidence decision engine with explicit outcomes."""

    mode = "deterministic-evidence-v3"
    estimated_cost_usd = 0.0

    def __init__(self, *, embeddings: EmbeddingProvider, config: DecisionConfig) -> None:
        self.embeddings = embeddings
        self.config = config

    def _candidate_assessments(
        self,
        kind: str,
        ranked: list[RankedChunk],
        evaluated: dict[str, list[tuple[float, str, str | None, bool]]],
    ) -> tuple[PassageCandidateAssessment, ...]:
        assessments: list[PassageCandidateAssessment] = []
        for item in ranked:
            rows = evaluated.get(item.chunk.id, [])
            direct = [row for row in rows if row[2] is not None and row[3]]
            if direct:
                score, excerpt, value, _relation = max(direct, key=lambda row: row[0])
                supported = score >= self.config.support_threshold
                assessments.append(
                    PassageCandidateAssessment(
                        answerable=supported,
                        answer=value if supported else None,
                        confidence=score,
                        supporting_document=item.chunk.document_id,
                        supporting_page=item.chunk.page,
                        supporting_excerpt=excerpt,
                        ambiguity_reason=(
                            None
                            if supported
                            else "Candidate evidence is below the frozen support threshold."
                        ),
                        extracted_fields=(
                            {_field_name(kind): value}
                            if supported and value and kind != "boolean"
                            else {}
                        ),
                        supporting_chunk_id=item.chunk.id,
                    )
                )
                continue
            partial = [
                row
                for row in rows
                if row[0] >= self.config.partial_support_threshold
                and row[3]
            ]
            if partial:
                score, excerpt, _value, _relation = max(partial, key=lambda row: row[0])
                assessments.append(
                    PassageCandidateAssessment(
                        answerable=False,
                        answer=None,
                        confidence=score,
                        supporting_document=item.chunk.document_id,
                        supporting_page=item.chunk.page,
                        supporting_excerpt=excerpt,
                        ambiguity_reason=(
                            "The passage identifies the relation but not the requested value."
                        ),
                        extracted_fields={},
                        supporting_chunk_id=item.chunk.id,
                    )
                )
                continue
            assessments.append(
                PassageCandidateAssessment(
                    answerable=False,
                    answer=None,
                    confidence=max((row[0] for row in rows), default=0.0),
                    supporting_document=None,
                    supporting_page=None,
                    supporting_excerpt=None,
                    ambiguity_reason="No sufficient evidence in this passage.",
                    extracted_fields={},
                    supporting_chunk_id=item.chunk.id,
                )
            )
        return tuple(assessments)

    def answer(self, question: str, ranked: list[RankedChunk]) -> GroundedAnswer:
        if _unsafe_question(question) or not ranked:
            assessments = tuple(
                PassageCandidateAssessment(
                    answerable=False,
                    answer=None,
                    confidence=0.0,
                    supporting_document=None,
                    supporting_page=None,
                    supporting_excerpt=None,
                    ambiguity_reason="The question was refused before evidence evaluation.",
                    extracted_fields={},
                    supporting_chunk_id=item.chunk.id,
                )
                for item in ranked
            )
            return _abstained(
                "Request refused: document instructions are untrusted."
                if _unsafe_question(question)
                else "Insufficient evidence in the selected dossier.",
                candidate_assessments=assessments,
            )
        kind = _question_kind(question)
        ranked = _scope_ranked(question, ranked)
        if not ranked:
            return _abstained()
        trust_input = TrustSeparatedInput.build(
            user_question=question,
            documents=tuple(
                UntrustedDocumentContent(
                    document_id=item.chunk.document_id,
                    page=item.chunk.page,
                    chunk_id=item.chunk.id,
                    text=item.chunk.text,
                )
                for item in ranked
            ),
        )
        candidates: list[tuple[float, str, str | None, RankedChunk]] = []
        partials: list[tuple[float, str, RankedChunk]] = []
        evaluated: dict[str, list[tuple[float, str, str | None, bool]]] = {}
        documents_by_chunk = {
            document.chunk_id: document for document in trust_input.untrusted_document_content
        }
        for item in ranked:
            parent_content = documents_by_chunk[item.chunk.id]
            segments = _segments(item.chunk.text)
            blocked_segments = untrusted_document_fragment_indexes(segments)
            for segment_index, segment in enumerate(segments):
                if segment_index in blocked_segments:
                    continue
                if (
                    segment.isupper()
                    and not any(
                        pattern.search(segment)
                        for pattern in (_DATE, _AMOUNT, _DURATION, _SEVERITY)
                    )
                    and not _has_explicit_absence(segment)
                    and not _NEGATION.search(segment)
                ):
                    # All-caps headings often repeat query terms without making
                    # an assertion that can support a yes/no answer.
                    continue
                evidence_segments = (
                    _date_evidence_segments(segment, question) if kind == "date" else [segment]
                )
                for evidence_segment in evidence_segments:
                    if kind != "boolean" and (
                        _has_explicit_absence(evidence_segment)
                        or _has_negated_relation(question, evidence_segment, kind)
                    ):
                        evaluated.setdefault(item.chunk.id, []).append(
                            (0.0, evidence_segment, None, False)
                        )
                        continue
                    admitted = admit_evidence_excerpt(
                        UntrustedDocumentContent(
                            document_id=parent_content.document_id,
                            page=parent_content.page,
                            chunk_id=item.chunk.id,
                            text=evidence_segment,
                        )
                    )
                    if admitted is None:
                        continue
                    evidence_text = admitted.text
                    score = _support_score(question, evidence_text, kind, self.embeddings)
                    value = _value_for(kind, evidence_text, question)
                    relation = _relation_supported(question, evidence_text, kind)
                    evaluated.setdefault(item.chunk.id, []).append(
                        (score, evidence_text, value, relation)
                    )
                    if value is not None and relation:
                        candidates.append((score, evidence_text, value, item))
                    elif (
                        score >= self.config.partial_support_threshold
                        and _has_explicit_relation(question, evidence_text)
                        and not (kind == "date" and _asks_past_event_date(question))
                    ):
                        partials.append((score, evidence_text, item))
        assessments = self._candidate_assessments(kind, ranked, evaluated)
        candidates.sort(key=lambda value: (-value[0], -value[3].score, value[3].chunk.id))
        supported = [item for item in candidates if item[0] >= self.config.support_threshold]
        if not supported:
            if partials:
                score, excerpt, item = max(partials, key=lambda value: value[0])
                result = GroundedAnswer(
                    status="partially_supported",
                    answerable=False,
                    answer="The retrieved evidence does not state the requested value.",
                    confidence=score,
                    supporting_document=item.chunk.document_id,
                    supporting_page=item.chunk.page,
                    supporting_excerpt=excerpt,
                    ambiguity_reason=(
                        "The evidence identifies the event but not the requested value."
                    ),
                    extracted_fields={},
                    supporting_chunk_ids=(item.chunk.id,),
                    citations=(_citation(item, excerpt),),
                    candidate_assessments=assessments,
                )
                validate_grounded_answer(result, ranked)
                return result
            return _abstained(candidate_assessments=assessments)

        best = supported[0]
        distinct = [
            other
            for other in supported[1:]
            if _fold(other[2] or "") != _fold(best[2] or "")
        ]
        if distinct and kind in {"boolean", "date", "amount", "duration", "severity"}:
            other = distinct[0]
            citations = (_citation(best[3], best[1]), _citation(other[3], other[1]))
            result = GroundedAnswer(
                status="ambiguous",
                answerable=False,
                answer="Conflicting values are supported; review the cited passages.",
                confidence=min(best[0], other[0]),
                supporting_document=best[3].chunk.document_id,
                supporting_page=best[3].chunk.page,
                supporting_excerpt=best[1],
                ambiguity_reason="Conflicting values are supported by comparable evidence.",
                extracted_fields={},
                supporting_chunk_ids=(best[3].chunk.id, other[3].chunk.id),
                citations=citations,
                candidate_assessments=assessments,
            )
            validate_grounded_answer(result, ranked)
            return result

        field_name = _field_name(kind)
        result = GroundedAnswer(
            status="answered",
            answerable=True,
            answer=best[2] or best[1],
            confidence=best[0],
            supporting_document=best[3].chunk.document_id,
            supporting_page=best[3].chunk.page,
            supporting_excerpt=best[1],
            ambiguity_reason=None,
            extracted_fields=(
                {field_name: best[2] or best[1]} if kind != "boolean" else {}
            ),
            supporting_chunk_ids=(best[3].chunk.id,),
            citations=(_citation(best[3], best[1]),),
            candidate_assessments=assessments,
        )
        validate_grounded_answer(result, ranked)
        return result
