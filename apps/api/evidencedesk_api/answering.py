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
_CLOCK = re.compile(
    r"\b(?:at|a|à)?\s*(\d{1,2}:\d{2})(?:\s*(CET|CEST|UTC|GMT|[AP]M))?\b",
    re.IGNORECASE,
)
_EXPLICIT_ID = re.compile(
    r"(?<![A-Z0-9])(?=[A-Z0-9/-]*\d)([A-Z]{1,8}(?:[-/][A-Z0-9]+)+)(?![A-Z0-9])",
    re.IGNORECASE,
)
_UNIT = (
    r"°\s*[CF]|(?:k|M)?(?:Pa|bar)|psi|NTU|(?:mg|µg|μg|g|kg)/(?:mL|L|dL)|"
    r"(?:km|m|cm|mm)/h|m/s|"
    r"(?:m|cm|mm|km|in|ft|mi)|(?:mL|L|dL|kg|g|mg|µg|μg)|"
    r"(?:ms|s|min|h|Hz|kHz|MHz|GHz|rpm|W|kW|MW|V|mV|A|mA|kWh|Mbps|Gbps)|%"
)
_MEASUREMENT = re.compile(
    rf"(?<![\w/.-])(\d+(?:[.,]\d+)?)\s*({_UNIT})(?![\w])",
    re.IGNORECASE,
)
_CONDITION = re.compile(
    r"\b(?:under|during|if|sous|pendant|si)\s+(?:the\s+|la\s+|le\s+)?"
    r"([a-z0-9][a-z0-9 -]{0,35})",
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
    "du",
    "chez",
    "par",
    "hors",
    "sauf",
    "and",
    "or",
    "also",
    "selon",
    "according",
    "accordingto",
    "dapres",
}

_CAPITALIZED_QUERY_STOPWORDS = {
    "a", "an", "can", "de", "does", "how", "if", "is", "le", "la",
    "on", "the", "was", "what", "when", "which", "whose", "who", "where", "combien",
    "why", "may", "might", "will", "would", "qui", "quand", "comment", "pourquoi",
    "ou", "les", "un", "master", "services", "service", "incident",
    "data", "processing", "personal", "security", "review", "platform",
    "une", "des", "du", "au", "aux", "quel", "quelle", "quels", "quelles",
}

_DOCUMENT_TYPE_TERMS = {
    "agreement": {"agreement", "contract", "contrat"},
    "addendum": {"addendum", "annex", "amendment"},
    "report": {"report"},
    "register": {"register"},
    "permit": {"permit", "permis"},
    "plan": {"plan"},
}

_RELATION_EQUIVALENTS = {
    "entry": {"entry", "entree", "entrance"},
    "start": {
        "start", "starts", "starting", "begin", "begins", "beginning",
        "commence", "commences", "commencer", "debut", "debute", "demarre",
    },
    "finish": {
        "finish", "finishes", "finished", "complete", "completes", "completed",
        "end", "ends", "ending", "termine", "terminer", "acheve",
    },
    "reserve": {"reserve", "reserves", "fund", "funds", "fond", "fonds", "pool"},
    "maintenance": {"maintenance", "entretien"},
    "coverage": {"coverage", "cover", "couverture"},
    "precedence": {
        "control", "controls", "govern", "governs", "prevail", "prevails",
        "precedence", "priority", "replace", "replaces", "supersede", "supersedes",
    },
    "amend": {
        "amend", "amends", "amended", "amendment", "change", "changes", "changed",
        "modify", "modifies", "modified", "alter", "alters", "altered",
    },
    "access": {"access", "accesses", "accessed", "accessing", "acces"},
    "storage": {"storage", "stockage", "archivage", "archiving"},
    "restoration": {
        "restore", "restores", "restored", "restoration", "recovery", "recover",
        "retablissement",
    },
    "monthly": {
        "monthly", "month", "months", "mois", "mensuel", "mensuelle", "mensuels",
        "mensuelles",
    },
}

_ABSENCE = re.compile(
    r"\b(?:not\s+(?:provided|recorded|listed|available|specified|stated|supplied)|"
    r"(?:unavailable|absent|missing|indisponible|absent|manquant)\b|"
    r"no\s+(?:date|amount|value|number|record|entry)\s+(?:is\s+)?(?:provided|recorded|listed)|"
    r"unknown|unrecorded|n/?a|none|non\s+(?:renseigne|enregistre|indique|consigne|consign(?:ée|é|ee))|"
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
    folded = _fold(text)
    for term in ("fee", "fees", "budget", "budgets", "tariff", "tariffs"):
        if re.search(rf"\b{term}\b", folded):
            tokens.add(term.rstrip("s"))
    known_terms = set().union(*_RELATION_EQUIVALENTS.values())
    for token in tuple(tokens):
        if token[:1] in {"l", "d"} and token[1:] in known_terms:
            tokens.discard(token)
            tokens.add(token[1:])
    for canonical, equivalents in _RELATION_EQUIVALENTS.items():
        matched = tokens & equivalents
        if matched:
            tokens.difference_update(matched)
            tokens.add(canonical)
    return tokens


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()


def _requested_document_type(question: str) -> str | None:
    folded = _fold(question)
    occurrences = sorted([
        (match.start(), match.end(), kind, term)
        for kind, terms in _DOCUMENT_TYPE_TERMS.items()
        for term in terms
        for match in re.finditer(rf"\b{re.escape(term)}\b", folded)
    ])
    if not occurrences:
        return None
    eligible_occurrences = []
    source_cues = []
    for start, end, kind, term in occurrences:
        prefix = folded[max(0, start - 32):start]
        suffix = folded[end:end + 32]
        if re.search(
            r"\b(?:date|status|statut)\b.{0,32}\b(?:renewal|renouvellement)\b.{0,12}$",
            prefix,
        ):
            continue
        eligible_occurrences.append((start, end, kind, term))
        # A type word modifying a field is not a source-document request.
        if re.match(r"\s+(?:renewal|effective|start|end|expiry)\s+(?:date|day|term)\b", suffix):
            continue
        if re.search(
            r"\b(?:in|under|from|within|of|according\s+to|per|selon|d['\u2019]apres)"
            r"\s+(?:the\s+|le\s+|la\s+|l['\u2019])?$",
            prefix,
        ):
            source_cues.append((start, kind))
    if source_cues:
        return min(source_cues)[1]
    source_agent = re.search(
        r"\b(?:the\s+|le\s+|la\s+|l['\u2019])?"
        r"(plan|permit|permis|report|rapport|register|registre)\b"
        r"(?:\s+(?:itself|lui[- ]m[eê]me))?\s+"
        r"(?:associates?|links?|lists?|indicates?|identifies?|records?|"
        r"shows?|states?|relie|associe|indique|identifie|consigne|mentionne)\b",
        folded,
    )
    if source_agent is not None:
        term = source_agent.group(1)
        return next(
            kind for kind, terms in _DOCUMENT_TYPE_TERMS.items() if term in terms
        )
    if not eligible_occurrences:
        return None
    if len({kind for _, _, kind, _ in eligible_occurrences}) == 1:
        start, end, kind, _ = eligible_occurrences[0]
        if kind in {"permit", "plan"}:
            return None
        suffix = folded[end:end + 32]
        if re.match(r"\s+(?:renewal|effective|start|end|expiry)\s+(?:date|day|term)\b", suffix):
            return None
        return kind
    # When a document is the grammatical subject of a change question, use it
    # as the evidence source and treat the later document as its target.
    source = re.search(
        r"\b(?:report|register|agreement|contract|addendum|annex)\b.{0,50}"
        r"\b(?:amend|amends|amended|change|changes|changed|modify|modifies|alter|alters)\b",
        folded,
    )
    if source:
        first = next(
            (kind for start, _, kind, _ in eligible_occurrences if start >= source.start()),
            None,
        )
        return first
    return None


def _requested_source_terms(question: str) -> set[str]:
    """Resolve a direct source phrase such as 'according to the plan'."""
    folded = _fold(question)
    match = re.search(
        r"\b(?:according\s+to|per|selon|d['\u2019]apres|in\s+the|from\s+the|under\s+the)\s+"
        r"(?:the\s+|le\s+|la\s+|l['\u2019])?([a-z][a-z0-9 _-]{1,38}?)"
        r"(?=\s*(?:,|\.|;|:|itself\b|lui[- ]m[eê]me\b|\bwhat\b|\bwhich\b|"
        r"\bwhen\b|\bwhere\b|\bcomment\b|\bquelle?\b|\ba\s+quelle\b|$))",
        folded,
    )
    if match is None:
        # A document used as the grammatical subject is also an explicit source
        # request (for example, “which code does the plan associate?”).
        if _requested_document_type(question) is None:
            return set()
        source_agent = re.search(
            r"\b(?:the\s+|le\s+|la\s+|l['\u2019])?"
            r"(plan|permit|permis|report|rapport|register|registre)\b"
            r"(?:\s+(?:itself|lui[- ]m[eê]me))?\s+"
            r"(?:associates?|links?|lists?|indicates?|identifies?|records?|"
            r"shows?|states?|relie|associe|indique|identifie|consigne|mentionne)\b",
            folded,
        )
        if source_agent is None:
            return set()
        source_term = source_agent.group(1)
        kind = next(
            kind for kind, terms in _DOCUMENT_TYPE_TERMS.items() if source_term in terms
        )
        return {kind}
    phrase = match.group(1).strip()
    terms = set(re.findall(r"[a-z0-9]+", phrase))
    for identifier in _EXPLICIT_ID.finditer(phrase):
        terms.difference_update(re.findall(r"[a-z0-9]+", _fold(identifier.group(1))))
    normalized_types = {
        kind
        for kind, aliases in _DOCUMENT_TYPE_TERMS.items()
        if terms & aliases
    }
    if normalized_types:
        terms = {
            term
            for term in terms
            if not any(term in aliases for aliases in _DOCUMENT_TYPE_TERMS.values())
        }
        terms.update(normalized_types)
    return terms - {
        "document", "source", "itself", "lui", "meme", "directly",
        "explicitly", "the", "le", "la", "les", "un", "une",
    }


def _requested_source_ids(question: str) -> set[str]:
    folded = _fold(question)
    match = re.search(
        r"\b(?:according\s+to|per|selon|d['\u2019]apres|in\s+the|from\s+the|under\s+the)\s+"
        r"(?:the\s+|le\s+|la\s+|l['\u2019])?([a-z0-9 _/-]{1,48})",
        folded,
    )
    if match is None:
        return set()
    phrase = re.split(
        r"\b(?:itself|directly|explicitly|states?|says?|lists?|indique|mentionne|"
        r"what|which|who|whose|is|are|was|were|does|do|did|has|have|that|the)\b",
        match.group(1),
        maxsplit=1,
    )[0]
    return {_fold(identifier.group(1)) for identifier in _EXPLICIT_ID.finditer(phrase)}


def _document_type(item: RankedChunk, contents: str) -> str | None:
    filename = _fold(item.chunk.document_name)
    if re.search(r"(?:^|[_\W])(?:permit|permis)(?:$|[_\W])", filename):
        return "permit"
    if re.search(r"(?:^|[_\W])plan(?:$|[_\W])", filename):
        return "plan"
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
    if re.search(r"\b(?:permit|permis)\b", title):
        return "permit"
    if re.search(r"\bplan\b", title):
        return "plan"
    return None


def _named_subject_tokens(question: str) -> set[str]:
    tokens: set[str] = set()
    requested_field = _requested_field_words(question)
    for match in re.finditer(
        r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ']*(?:-[A-Za-z0-9]+)*\b", question
    ):
        raw = match.group(0)
        token = re.sub(r"(?:'s|s')$", "", _fold(raw))
        expanded = _camel_words(raw)
        typed_label = bool(re.search(r"(?:ID|Code|Tag|Ref|Number)$", raw))
        field_word = bool(
            typed_label
            and requested_field
            and set(_relation_tokens(expanded)) & requested_field
        )
        if (
            (raw.isupper() and len(raw) < 2)
            or token in _CAPITALIZED_QUERY_STOPWORDS
            or token in _QUESTION_NOISE
            or typed_label
            or field_word
        ):
            continue
        if token in set.union(*_DOCUMENT_TYPE_TERMS.values()):
            continue
        tokens.add(token)
    return tokens


def _camel_words(value: str) -> str:
    separated = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    separated = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", separated)
    return separated


def _requested_field_words(question: str) -> set[str]:
    """Identify the requested field label, separate from its record constraints."""
    return set(_relation_tokens(" ".join(_requested_field_parts(question))))


def _requested_field_parts(question: str) -> list[str]:
    camel_label = re.search(
        r"^\s*(?:what|which|quel|quelle|quels|quelles)\s+"
        r"(?:(?:is|are|was|were|est|etait|etaient)\s+)?"
        r"(?:(?:the|le|la|les|l)\s+)?"
        r"([A-Z][A-Za-z0-9]*(?:ID|Code|Tag|Ref|Number))\b",
        question,
        re.IGNORECASE,
    )
    if camel_label:
        words = _camel_words(camel_label.group(1)).split()
        if words and words[-1].casefold() == "id":
            words.pop()
        return [_fold(word) for word in words if word.casefold() not in {"id"}]

    folded = _fold(question)
    match = re.search(
        r"\b(?:what|which|quel|quelle|quels|quelles)\s+"
        r"(?:(?:is|are|was|were|est|etait|etaient)\s+)?"
        r"(?:(?:the|le|la|les|l)\s+)?(.+)",
        folded,
    )
    if match:
        stops = {
            "a", "an", "the", "le", "la", "les", "de", "du", "des", "d",
            "for", "in", "on", "at", "to", "under", "from", "by", "with",
            "is", "are", "was", "were", "does", "do", "did", "has", "have",
            "est", "sont", "ont", "il", "elle", "ils", "elles", "lequel",
            "laquelle", "printed", "listed", "recorded", "indicated", "begins",
            "commence", "reported", "donne", "indique", "indiquee",
            "indiquees", "note", "notee", "explicitly", "directly", "is-il", "est-il",
        }
        words = re.findall(r"[a-z0-9]+", match.group(1))
        selected: list[str] = []
        for word in words:
            if word in stops or re.fullmatch(r"[a-z]{1,8}-\d+[a-z0-9-]*", word):
                break
            selected.append(word)
            if len(selected) == 5:
                break
        return [word for word in selected if word not in _QUESTION_NOISE]
    return []


def _requested_identifier_values(question: str) -> set[str]:
    return {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(question)}


def _requested_clock_values(question: str) -> set[str]:
    values: set[str] = set()
    for match in _CLOCK.finditer(question):
        value = match.group(1)
        zone = match.group(2)
        values.add(f"{value} {zone}".casefold() if zone else value.casefold())
    return values


def _requested_conditions(question: str) -> set[str]:
    conditions = {
        re.sub(r"\s+", " ", _fold(match.group(1))).strip()
        for match in _CONDITION.finditer(question)
    }
    return {
        condition
        for condition in conditions
        if not re.search(
            r"\b(?:agreement|contract|addendum|policy|accord|contrat|avenant)\b",
            condition,
        )
        and not re.search(
            r"\b(?:one|must|selected|controls|which|deadline|delai)\b", condition
        )
    }


def _identifier_prefix(value: str) -> str:
    return re.split(r"[-/]", value, maxsplit=1)[0]


def _constraints_present(
    question: str, evidence: str, constraint_context: str | None = None
) -> bool:
    requested_ids = _requested_identifier_values(question)
    evidence_ids = {
        _fold(match.group(1)) for match in _EXPLICIT_ID.finditer(evidence)
    }
    context = constraint_context or evidence
    context_ids = {
        _fold(match.group(1)) for match in _EXPLICIT_ID.finditer(context)
    }
    for requested_id in requested_ids:
        local_matches = {
            value
            for value in evidence_ids
            if _identifier_prefix(value) == _identifier_prefix(requested_id)
        }
        context_matches = {
            value
            for value in context_ids
            if _identifier_prefix(value) == _identifier_prefix(requested_id)
        }
        if requested_id in local_matches:
            if local_matches != {requested_id}:
                return False
        elif context_matches != {requested_id}:
            return False
    evidence_clocks = {
        f"{match.group(1)} {match.group(2)}".casefold()
        if match.group(2)
        else match.group(1).casefold()
        for match in _CLOCK.finditer(evidence)
    }
    requested_clocks = _requested_clock_values(question)
    if requested_clocks and not requested_clocks & evidence_clocks:
        return False
    requested_dates = {_fold(match.group(0)) for match in _DATE.finditer(question)}
    evidence_dates = {_fold(match.group(0)) for match in _DATE.finditer(evidence)}
    if requested_dates and not requested_dates & evidence_dates:
        return False
    return all(condition in _fold(evidence) for condition in _requested_conditions(question))


def _field_value(
    question: str, segment: str, constraint_context: str | None = None
) -> tuple[str, str] | None:
    """Extract one explicitly labeled value with all query constraints present."""
    field_parts = _requested_field_parts(question)
    if not field_parts or not _constraints_present(question, segment, constraint_context):
        return None
    aliases = {
        "modele": "model",
        "turbidite": "turbidity",
        "maximum": "peak",
        "maximale": "peak",
        "maximal": "peak",
        "laboratoire": "laboratory",
        "concentration": "",
        "level": "",
        "reading": "",
        "readings": "",
        "value": "",
    }
    target_options = [field_parts]
    canonical_parts = [aliases.get(part, part) for part in field_parts]
    canonical_parts = [part for part in canonical_parts if part]
    camel_id_label = bool(
        re.match(
            r"^\s*(?:what|which|quel|quelle|quels|quelles)\s+"
            r"(?:(?:is|are|est)\s+)?(?:(?:the|le|la|les|l)\s+)?"
            r"[A-Z][A-Za-z0-9]*ID\b",
            question,
            re.IGNORECASE,
        )
    )

    def label_pattern(parts: list[str]) -> str:
        suffix = r"(?:[\s_-]*id)?" if camel_id_label or parts else ""
        return (
            r"(?<![A-Za-z0-9])"
            + r"[\s_-]*".join(re.escape(part) for part in parts)
            + suffix
            + r"(?![A-Za-z0-9])"
        )

    field_matches: list[re.Match[str]] = []
    for parts in target_options:
        field_matches.extend(re.finditer(label_pattern(parts), segment, re.IGNORECASE))
    if not field_matches and canonical_parts != field_parts:
        target_options = [canonical_parts]
        field_matches.extend(
            re.finditer(label_pattern(canonical_parts), segment, re.IGNORECASE)
        )
    # A measurement may be stated in a different word order, such as
    # "Peak turbidity" for a question about maximum turbidity.
    if not field_matches and "peak" in canonical_parts and "turbidity" in canonical_parts:
        field_matches.extend(
            re.finditer(
                r"(?<![A-Za-z0-9])peak[\s_-]*turbidity(?![A-Za-z0-9])",
                segment,
                re.IGNORECASE,
            )
        )
    field_matches = list({(match.start(), match.end()): match for match in field_matches}.values())
    if len(field_matches) != 1:
        return None
    match = field_matches[0]
    tail = segment[match.end() :]
    tail = re.split(r"[|;\n!?]|(?<!\d)\.(?!\d)", tail, maxsplit=1)[0]
    requested_ids = _requested_identifier_values(question)
    for constrained_identifier in _EXPLICIT_ID.finditer(tail):
        if _fold(constrained_identifier.group(1)) not in requested_ids:
            continue
        after_identifier = tail[constrained_identifier.end() :]
        value_after_constraint = re.match(
            r"\s*(?:,\s*)?(?:is|are|was|were|est|sont|:|=)\s*(.+)$",
            after_identifier,
            re.IGNORECASE,
        )
        if value_after_constraint:
            # A requested record ID is a constraint, not the answer to the
            # field preceding “for/of/at <record> is ...”.
            tail = value_after_constraint.group(1)
            break
    tail = re.sub(
        r"^\s*(?::|=|-)?\s*(?:(?:is|are|was|were|listed|recorded|entered|printed|"
        r"indicated|begins|names|explicitly\s+names|explicitly\s+references|"
        r"references|associates|assigned|assigns|noted|reported|maximum|peak|"
        r"reading|value|concentration)\b\s*)?[:=]?\s*",
        "",
        tail,
        flags=re.IGNORECASE,
    ).strip(" ,:")
    if not tail:
        return None
    measurement, unknown_measurement_precedes = _measurement_after_label(tail)
    if unknown_measurement_precedes:
        return None
    identifier = _EXPLICIT_ID.search(tail)
    if measurement:
        value = measurement.group(0).strip()
    elif identifier:
        value = identifier.group(1)
    else:
        number = re.match(r"\s*(\d+(?:[.,]\d+)?)\b", tail)
        if number and set(field_parts) & {"ph", "score", "rating"}:
            value = number.group(1)
        else:
            text_match = re.match(
                r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9 .'\u2019/-]*", tail
            )
            if text_match is None:
                return None
            value = text_match.group(0).strip(" .,:;!?\t")
    original = segment
    original_labels = [label_pattern(parts) for parts in target_options]
    if "peak" in canonical_parts and "turbidity" in canonical_parts:
        original_labels.append(r"\bpeak\s+turbidity\b")
    if not original_labels:
        return None
    original_match = next(
        (
            found
            for label in original_labels
            if (found := re.search(label, original, re.IGNORECASE))
        ),
        None,
    )
    if original_match is None:
        return None
    original_tail = original[original_match.end() :]
    if _requested_conditions(question) and len(list(_MEASUREMENT.finditer(original_tail))) > 1:
        # Without a parsed condition-to-measurement link, multiple readings in
        # one sentence cannot be safely assigned to the requested condition.
        return None
    source_measurement, unknown_measurement_precedes = _measurement_after_label(
        original_tail
    )
    if unknown_measurement_precedes:
        return None
    if source_measurement:
        value = source_measurement.group(0).strip()
    original_value = re.search(
        re.escape(value), original[original_match.end() :], re.IGNORECASE
    )
    if original_value is None:
        return None
    assertion_start, assertion_end = _assertion_bounds(
        segment,
        original_match.start(),
        original_match.end() + original_value.end(),
    )
    assertion = segment[assertion_start:assertion_end]
    assertion_folded = _fold(assertion)
    assertion_ids = _requested_identifier_values(assertion)
    requested_ids = _requested_identifier_values(question)
    context_header = _unique_context_header(
        question, segment, constraint_context
    )
    row_ids = _requested_identifier_values(original)
    row_link = "|" in original and requested_ids <= row_ids and all(
        len(
            {
                identifier
                for identifier in row_ids
                if _identifier_prefix(identifier) == _identifier_prefix(requested_id)
            }
        ) == 1
        for requested_id in requested_ids
    )
    if not requested_ids <= assertion_ids and not context_header and not row_link:
        return None
    if not _requested_conditions(question) <= {
        condition
        for condition in _requested_conditions(question)
        if condition in assertion_folded
    }:
        return None
    assertion_clocks = {
        f"{match.group(1)} {match.group(2)}".casefold()
        if match.group(2)
        else match.group(1).casefold()
        for match in _CLOCK.finditer(assertion)
    }
    if _requested_clock_values(question) and not (
        _requested_clock_values(question) & assertion_clocks
    ):
        return None
    assertion_dates = {_fold(match.group(0)) for match in _DATE.finditer(assertion)}
    if _DATE.search(question) and not (
        {_fold(match.group(0)) for match in _DATE.finditer(question)} & assertion_dates
    ):
        return None
    source = original
    offset = 0
    start = offset + original_match.start()
    end = offset + original_match.end() + original_value.end()
    assertion_source_start = offset + assertion_start
    assertion_source_end = offset + assertion_end
    for requested_id in _requested_identifier_values(question):
        id_match = re.search(
            rf"(?<![A-Z0-9]){re.escape(requested_id)}(?![A-Z0-9])",
            source,
            re.IGNORECASE,
        )
        if id_match and (
            assertion_source_start <= id_match.start() < assertion_source_end
            or row_link
        ):
            clause = _local_value_clause(source, id_match.start(), id_match.end())
            clause_start = source.find(clause)
            start = min(
                start, clause_start if clause_start >= 0 else id_match.start()
            )
            end = max(
                end,
                clause_start + len(clause)
                if clause_start >= 0
                else id_match.end(),
            )
    for clock in _CLOCK.finditer(source):
        if not assertion_source_start <= clock.start() < assertion_source_end:
            continue
        clock_value = (
            f"{clock.group(1)} {clock.group(2)}".casefold()
            if clock.group(2)
            else clock.group(1).casefold()
        )
        if clock_value in _requested_clock_values(question):
            start = min(start, clock.start())
            end = max(end, clock.end())
    for date_match in _DATE.finditer(source):
        if not assertion_source_start <= date_match.start() < assertion_source_end:
            continue
        if _fold(date_match.group(0)) in {
            _fold(match.group(0)) for match in _DATE.finditer(question)
        }:
            start = min(start, date_match.start())
            end = max(end, date_match.end())
    for condition_match in _CONDITION.finditer(source):
        if not assertion_source_start <= condition_match.start() < assertion_source_end:
            continue
        condition = re.sub(r"\s+", " ", _fold(condition_match.group(1))).strip()
        if condition in _requested_conditions(question):
            start = min(start, condition_match.start())
            end = max(end, condition_match.end())
    if end < len(source) and source[end] in ".!?":
        end += 1
    source_excerpt = source[start:end].strip()
    return value, source_excerpt


def _subject_matches(subject: str, contents: str) -> bool:
    folded_subject = _fold(subject)
    folded_contents = _fold(contents)
    if re.fullmatch(r"[a-z]{1,8}-\d+[a-z0-9-]*", folded_subject):
        return re.search(
            rf"(?<![a-z0-9]){re.escape(folded_subject)}(?![a-z0-9])",
            folded_contents,
        ) is not None
    return bool(_relation_tokens(subject) & _relation_tokens(contents))


_TYPED_LINK_ID = re.compile(
    r"\b(shipment|dispatch|consignment|purchase\s+order|order|contract|"
    r"agreement|invoice|case|claim|batch|lot|container)\s+"
    r"(?:(?:reference|ref|id|number|no)\s*[:#]?\s*)?"
    r"([a-z]{1,8}[-/]\d+[a-z0-9/-]*)\b",
    re.IGNORECASE,
)


def _typed_link_keys(contents: str) -> set[tuple[str, str]]:
    """Return explicit, typed record identifiers suitable for joining sources."""
    return {
        (re.sub(r"\s+", " ", _fold(match.group(1))), _fold(match.group(2)))
        for match in _TYPED_LINK_ID.finditer(_fold(contents))
    }


def _requested_link_type(question: str) -> str | None:
    folded = _fold(question)
    type_terms = (
        ("purchase order", ("purchase order", "bon de commande", "commande")),
        ("shipment", ("shipment", "shipping", "expedition", "livraison")),
        ("dispatch", ("dispatch", "envoi")),
        ("consignment", ("consignment",)),
        ("contract", ("contract", "contrat")),
        ("agreement", ("agreement", "accord")),
        ("invoice", ("invoice", "facture")),
        ("case", ("case", "dossier")),
        ("claim", ("claim", "reclamation")),
        ("batch", ("batch", "lot")),
        ("container", ("container", "conteneur")),
        ("order", ("order", "commande")),
    )
    for link_type, terms in type_terms:
        if any(re.search(rf"\b{re.escape(term)}\b", folded) for term in terms):
            return link_type
    return None


def _scope_ranked(question: str, ranked: list[RankedChunk]) -> list[RankedChunk]:
    requested_type = _requested_document_type(question)
    source_terms = _requested_source_terms(question)
    subject_tokens = _named_subject_tokens(question)
    requested_source_ids = _requested_source_ids(question)
    if (
        requested_type is None
        and not source_terms
        and not requested_source_ids
        and not subject_tokens
    ):
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
    subjects_found_globally: set[str] = set()
    fully_scoped_documents: set[str] = set()
    eligible_documents: set[str] = set()
    link_keys_by_document: dict[str, set[tuple[str, str]]] = {}
    for document_id, items in grouped.items():
        contents = context_by_document[document_id]
        identity_lines = [items[0].chunk.document_name]
        title = contents.splitlines()[1] if len(contents.splitlines()) > 1 else ""
        if title and not re.search(
            r"\b(?:references?|cites?|quotes?|states?|says?|provides?|authorizes?|"
            r"mentions?|lists?|reports?|indique|mentionne|cite|renvoie)\b",
            _fold(title),
        ):
            identity_lines.append(title)
        identity_text = _fold(" ".join(identity_lines))
        if source_terms and not source_terms <= set(re.findall(r"[a-z0-9]+", identity_text)):
            continue
        identity_ids = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(identity_text)}
        if requested_source_ids and not requested_source_ids <= identity_ids:
            continue
        if requested_type is not None and _document_type(items[0], contents) != requested_type:
            continue
        if subject_tokens:
            eligible_documents.add(document_id)
            link_keys_by_document[document_id] = _typed_link_keys(contents)
            overlap = {
                subject for subject in subject_tokens if _subject_matches(subject, contents)
            }
            subjects_found_globally.update(overlap)
            if overlap == subject_tokens:
                fully_scoped_documents.add(document_id)
            if len(overlap) < max(1, (len(subject_tokens) * 2 + 2) // 3):
                continue
        allowed.add(document_id)
    if (
        requested_type is None
        and subject_tokens
        and subjects_found_globally == subject_tokens
        and not fully_scoped_documents
    ):
        # Cross-document evidence is joined only by an explicit identifier with
        # a record type (for example, shipment SH-2841) shared by the records.
        # Coincidental co-occurrence of named entities across documents is not a link.
        documents_by_key: dict[tuple[str, str], set[str]] = {}
        for document_id in eligible_documents:
            for key in link_keys_by_document.get(document_id, set()):
                documents_by_key.setdefault(key, set()).add(document_id)
        linked_documents: set[str] = set()
        requested_link_type = _requested_link_type(question)
        for key, document_ids in documents_by_key.items():
            if len(document_ids) < 2:
                continue
            if requested_link_type is None or key[0] != requested_link_type:
                continue
            linked_values = {
                value
                for document_id in document_ids
                for link_type, value in link_keys_by_document.get(document_id, set())
                if link_type == requested_link_type
            }
            if len(linked_values) != 1:
                continue
            linked_subjects = {
                subject
                for document_id in document_ids
                for subject in subject_tokens
                if _subject_matches(subject, context_by_document[document_id])
            }
            if linked_subjects == subject_tokens:
                linked_documents.update(document_ids)
        if not linked_documents:
            return []
        return [item for item in ranked if item.chunk.document_id in linked_documents]
    if not allowed:
        return []
    return [item for item in ranked if item.chunk.document_id in allowed]


def _has_explicit_absence(segment: str) -> bool:
    return _ABSENCE.search(_fold(segment)) is not None


def _has_requested_field_absence(question: str, segment: str) -> bool:
    """Recognize a stated absence only when it names the whole requested field."""
    parts = _requested_field_parts(question)
    if not parts:
        return False
    alternatives = [parts]
    aliases = {
        "modele": "model",
        "turbidite": "turbidity",
        "maximum": "peak",
        "maximale": "peak",
        "maximal": "peak",
    }
    canonical = [aliases.get(part, part) for part in parts]
    if canonical != parts:
        alternatives.append(canonical)
    action = r"(?:name|give|state|provide|list|record|mention|specify|identify)"
    field_absence = (
        r"(?:not\s+(?:provided|recorded|listed|available|specified|stated|supplied)|"
        r"unavailable|absent|missing|unrecorded)"
    )
    for candidate in _assertion_clauses(segment):
        for label in alternatives:
            pattern = (
                r"(?<![A-Za-z0-9])"
                + r"[\s_-]*".join(re.escape(part) for part in label)
                + r"(?:[\s_-]*id)?"
                + r"(?![A-Za-z0-9])"
            )
            label_match = re.search(pattern, candidate, re.IGNORECASE)
            if label_match is None:
                continue
            field = label_match.group(0)
            folded = _fold(candidate)
            # The field itself must be the omitted object or the subject whose
            # value is missing. A document noun that is the subject of a
            # different omission (“the permit does not name an operator”) does
            # not establish that the permit ID is absent.
            object_omitted = re.search(
                rf"\b(?:does\s+not|do\s+not|did\s+not|never|cannot|can't|"
                rf"ne\s+.*?pas)\s+{action}\s+(?:the\s+|a\s+|an\s+|le\s+|la\s+|l[' ]|un\s+|une\s+)?"
                rf"{re.escape(field)}\b",
                folded,
                re.IGNORECASE,
            )
            subject_omitted = re.search(
                rf"{re.escape(field)}\s+(?:(?:is|are|was|were|est|sont|n[' ]est)\s+)?"
                rf"(?:{field_absence}|non\s+(?:renseigne|enregistre|indique|fourni))\b",
                folded,
                re.IGNORECASE,
            )
            bare_omitted = re.search(
                rf"\b(?:no|without|aucun|aucune|sans)\s+(?:known\s+|known\s+value\s+|"
                rf"recorded\s+|documented\s+)?{re.escape(field)}\b",
                folded,
                re.IGNORECASE,
            )
            if object_omitted or subject_omitted or bare_omitted:
                return True
    return False


_LOCAL_PREDICATE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being|has|have|had|do|does|did|"
    r"can|could|may|might|must|shall|should|will|would|"
    r"est|sont|etait|etaient|était|étaient|sera|seront|a|ont|peut|peuvent|doit|doivent|"
    r"included|provided|recorded|listed|available|specified|stated|supplied|"
    r"covered|eligible|approved|authorized|allowed|required|applies|applied|"
    r"provides|gives|names|mentions|states|lists|records|identifies|"
    r"inclus|incluse|incluses|fourni|fournie|enregistre|enregistree|"
    r"consigne|consignee|disponible|couvert|couverte|autorise|autorisee|"
    r"requis|requise|logged|measured|observed|collected|mesure|note|releve)\b",
    re.IGNORECASE,
)


def _assertion_clauses(text: str) -> list[str]:
    """Split coordination only when each side expresses its own predicate."""
    for match in re.finditer(r"\b(?:and|et|but|mais|however|cependant)\b", text):
        left, right = text[: match.start()], text[match.end() :]
        if _LOCAL_PREDICATE.search(left) and _LOCAL_PREDICATE.search(right):
            return _assertion_clauses(left) + _assertion_clauses(right)
    return [text]


def _has_related_absence(question: str, segment: str, kind: str) -> bool:
    relation_tokens = _question_relation_tokens(question, kind)
    clauses = [
        clause
        for part in re.split(r"[;,]", _fold(segment))
        for clause in _assertion_clauses(part)
    ]
    return any(
        relation_tokens & _relation_tokens(clause)
        and _has_explicit_absence(clause)
        for clause in clauses
    )


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
    clauses = [
        clause
        for part in re.split(r"[;,]", folded_segment)
        for clause in _assertion_clauses(part)
    ]
    negation = r"\b(?:not|never|no|without|cannot|can't|mustn't|pas|jamais|aucun)\b"
    for clause in clauses:
        clause_tokens = _relation_tokens(clause)
        if relation_tokens & clause_tokens and re.search(negation, clause):
            return True
    return False


def _local_value_clause(segment: str, start: int, end: int) -> str:
    """Return the assertion clause containing a scalar, excluding adjacent facts."""
    left, right = _assertion_bounds(segment, start, end)
    return segment[left:right].strip(" ,;:")


def _assertion_bounds(segment: str, start: int, end: int) -> tuple[int, int]:
    boundary_text = re.sub(
        r"(?<!\w)([A-ZÀ-ÖØ-Þ])\.(?=\s+[A-ZÀ-ÖØ-Þ])",
        lambda match: match.group(1) + "\ue000",
        segment,
    )
    left = max(
        (
            boundary_text.rfind(mark, 0, start) + 1
            for mark in (";", "|", ".", "!", "?", "\n")
        ),
        default=0,
    )
    right_candidates = [
        position
        for mark in (";", "|", ".", "!", "?", "\n")
        if (position := boundary_text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(segment)
    base_left, base_right = left, right
    for match in re.finditer(
        r"\b(?:and|et|but|mais)\b", segment[base_left:base_right], re.IGNORECASE
    ):
        boundary = base_left + match.start()
        connector_end = base_left + match.end()
        left_clause = segment[base_left:boundary]
        right_clause = segment[connector_end:base_right]
        if _LOCAL_PREDICATE.search(left_clause) and _LOCAL_PREDICATE.search(right_clause):
            if boundary < start:
                left = connector_end
            elif boundary >= end:
                right = boundary
                break
    return left, right


def _is_identifier_header(text: str) -> bool:
    if not _EXPLICIT_ID.search(text):
        return False
    remainder = _EXPLICIT_ID.sub("", text)
    remainder = re.sub(
        r"\b[A-Za-z][A-Za-z0-9]*(?:ID|Code|Tag|Ref|Number)\b", "", remainder
    )
    return re.sub(r"[\s|:;,./-]+", "", remainder) == ""


def _has_unique_context_header(
    question: str, segment: str, constraint_context: str | None
) -> bool:
    return _unique_context_header(question, segment, constraint_context) is not None


def _unique_context_header(
    question: str, segment: str, constraint_context: str | None
) -> str | None:
    if (
        not constraint_context
        or constraint_context.count(segment) != 1
    ):
        return None
    requested = _requested_identifier_values(question)
    if not requested:
        return None
    segment_start = constraint_context.find(segment)
    headers = [
        (match.start(), match.group(0).strip())
        for match in re.finditer(r"[^.!?;\n]+", constraint_context)
        if _is_identifier_header(match.group(0).strip())
    ]
    eligible = [
        header
        for position, header in headers
        if position < segment_start
        and requested <= {
            _fold(match.group(1)) for match in _EXPLICIT_ID.finditer(header)
        }
    ]
    if len(eligible) != 1:
        return None
    header = eligible[0]
    all_context_ids = {
        _fold(match.group(1)) for match in _EXPLICIT_ID.finditer(constraint_context)
    }
    for requested_id in requested:
        same_type = {
            value
            for value in all_context_ids
            if _identifier_prefix(value) == _identifier_prefix(requested_id)
        }
        if same_type != {requested_id}:
            return None
    return header


def _question_excluded_terms(question: str) -> set[str]:
    folded = _fold(question)
    match = re.search(r"\b(?:except|excluding|other than|hors|sauf)\b(.*)$", folded)
    if match is None:
        return set()
    return _relation_tokens(match.group(1)) - _QUESTION_NOISE


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
            "amount": {
                "amount", "cost", "price", "prix", "how", "much", "apply", "charg",
                "constitue", "combien", "coute", "cout", "par", "hors", "sauf",
            },
        "duration": {"duration", "deadline", "period", "long", "soon", "quickly", "many"},
        "status": {"status", "statut", "recorded", "record", "consigne"},
        "severity": {"severity"},
        "responsible_people": {"owner"},
        "organization_name": set(),
        "boolean": {
            "is", "are", "was", "were", "does", "do", "did", "can", "could",
            "may", "might", "will", "would", "has", "have",
        },
        "text": set(),
    }
    relation_question = _EXPLICIT_ID.sub(" ", question)
    relation_question = re.sub(
        r"\b[A-Z][A-Za-z0-9]*(?:ID|Code|Tag|Ref|Number)\b",
        " ",
        relation_question,
    )
    tokens = (
        _relation_tokens(relation_question)
        - _QUESTION_NOISE
        - noise_by_kind.get(kind, set())
        - _question_excluded_terms(question)
    )
    tokens -= set(lexical_tokens(" ".join(_named_subject_tokens(question))))
    document_terms = set.union(*_DOCUMENT_TYPE_TERMS.values()) - {"amendment"}
    tokens -= set(lexical_tokens(" ".join(document_terms)))
    tokens -= {"master", "servic", "service", "incident", "signed", "sign"}
    # Numbers and units constrain a value request; they do not name its relation.
    for pattern in (_DATE, _AMOUNT, _DURATION, _SEVERITY):
        for match in pattern.finditer(question):
            tokens.difference_update(lexical_tokens(match.group(0)))
    requested_type = _requested_document_type(question)
    if requested_type is not None:
        tokens -= _DOCUMENT_TYPE_TERMS[requested_type]
    return tokens


def _event_scope_terms(text: str) -> set[str]:
    folded = _fold(text)
    terms: set[str] = set()
    if re.search(r"\b(?:cybersecurity|cyber)\s+(?:security\s+)?incident\b", folded):
        terms.update({"cybersecurity_incident", "incident"})
    if re.search(r"\bpersonal(?:[-\s]+data)?[-\s]+breach\b", folded):
        terms.add("personal_data_breach")
    if re.search(r"\b(?:service\s+)?(?:interruption|outage|disruption)\b", folded):
        terms.add("service_interruption")
    if re.search(r"\b(?:restoration|recovery)\b", folded):
        terms.add("restoration")
    if (
        re.search(r"\b(?:confirmed\s+)?incident\b", folded)
        and "cybersecurity_incident" not in terms
    ):
        terms.add("incident")
    return terms


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
        # Protect a person's initial from being interpreted as a sentence
        # boundary (for example, “technician R. Vale replaced the valve”).
        protected = re.sub(
            r"\b([A-ZÀ-ÖØ-Þ])\.(?=\s+[A-ZÀ-ÖØ-Þ])",
            lambda match: match.group(1) + "\ue000",
            stripped,
        )
        result.extend(
            part.replace("\ue000", ".").strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])", protected)
            if part.strip()
        )
    return result


_UNRECOGNIZED_MEASUREMENT = re.compile(
    r"(?<![\w/.-])\d+(?:[.,]\d+)?\s+[A-Za-zµμ°%][A-Za-z0-9µμ°/%]*"
)


def _measurement_after_label(text: str) -> tuple[re.Match[str] | None, bool]:
    """Find a typed measure without jumping over an earlier unknown unit."""
    recognized = _MEASUREMENT.search(text)
    unrecognized = _UNRECOGNIZED_MEASUREMENT.search(text)
    if unrecognized is not None and (
        recognized is None or unrecognized.start() < recognized.start()
    ):
        return None, True
    return recognized, False


def _question_kind(question: str) -> str:
    folded = _fold(question)
    interrogative_slots = re.findall(
        r"\b(?:who|which\s+(?:person|employee|time|date|amount)|whose|when|where|"
        r"what\s+(?:time|date|amount|status)|how\s+(?:much|many|long|soon))\b",
        folded,
    )
    if len(interrogative_slots) > 1:
        return "multi_slot"
    if re.search(
        r"\b(?:what\s+time|at\s+what\s+time|quelle\s+heure|a\s+quelle\s+heure)\b",
        folded,
    ):
        return "time"
    if re.search(r"\b(?:status|statut)\b", folded) and re.search(
        r"\b(?:date|renewal|renouvellement|value|champ|field)\b", folded
    ):
        return "status"
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
        r"\b(?:how\s+much\s+time|how\s+long|how\s+quickly|how\s+soon|how\s+many\s+"
        r"(?:minutes?|hours?|days?)|duration|duree|notice\s+period|notification\s+period|"
        r"combien\s+de\s+temps|delai(?:\s+de\s+notification)?|deadline|periode\s+de\s+preavis)\b",
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
        r"fonds|reserve|enveloppe|combien\s+(?:coute|cout|est|facture))\b",
        folded,
    ):
        return "amount"
    if re.search(r"\b(?:severity|severite|sev)\b", folded):
        return "severity"
    if re.match(
        r"^\s*(?:what|which|quel|quelle|quels|quelles)\s+"
        r"(?:(?:is|are|est)\s+)?(?:(?:the|le|la|les|l)\s+)?"
        r"[A-Z][A-Za-z0-9]*(?:ID|Code|Tag|Ref|Number)\b",
        question,
        re.IGNORECASE,
    ):
        return "field_value"
    if re.search(
        r"\b(?:organization|organisation|entity|entite|supplier|custodian|mutuelle|"
        r"beneficiary|beneficiaire|account holder|operator|issued)\b",
        folded,
    ) or "whose account" in folded:
        return "organization_name"
    if re.search(r"\b(?:who|qui|whose)\b", folded):
        return "responsible_people"
    if _requested_field_parts(question):
        return "field_value"
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


def _value_for(
    kind: str,
    segment: str,
    question: str,
    *,
    constraint_context: str | None = None,
) -> str | None:
    if kind == "field_value":
        extracted = _field_value(question, segment, constraint_context)
        return extracted[0] if extracted else None
    if kind == "time":
        if not _constraints_present(question, segment, constraint_context):
            return None
        selected = _time_value_context(question, segment)
        return selected[0] if selected else None
    if kind == "boolean":
        if not _relation_supported(question, segment, kind):
            return None
        if _has_related_absence(question, segment, kind) or _has_negated_relation(
            question, segment, kind
        ):
            return f"No. {segment}"
        question_amounts = list(_AMOUNT.finditer(question))
        question_dates = list(_DATE.finditer(question))
        question_durations = list(_DURATION.finditer(question))
        for requested_matches, observed_matches in (
            (question_amounts, list(_AMOUNT.finditer(segment))),
            (question_dates, list(_DATE.finditer(segment))),
            (question_durations, list(_DURATION.finditer(segment))),
        ):
            if requested_matches:
                requested_values = {
                    _fold(match.group(0)) for match in requested_matches
                }
                observed_values = {
                    _fold(match.group(0)) for match in observed_matches
                }
                if observed_values and not requested_values & observed_values:
                    return f"No. {segment}"
        return f"Yes. {segment}"
    if kind == "date":
        if not _constraints_present(question, segment, constraint_context):
            return None
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
        if not _constraints_present(question, segment, constraint_context):
            return None
        matches = list(_AMOUNT.finditer(segment))
        if not matches:
            return None
        query_terms = _question_relation_tokens(question, kind)
        excluded = _question_excluded_terms(question)
        scored = []
        for match in matches:
            context = _local_value_clause(segment, match.start(), match.end())
            context_terms = _relation_tokens(context)
            if excluded & context_terms:
                continue
            overlap = query_terms & context_terms
            scored.append((len(overlap), match.start(), match.group(0).strip()))
        if not scored:
            return None
        return max(scored, key=lambda row: (row[0], -row[1]))[2]
    if kind == "duration":
        if not _constraints_present(question, segment, constraint_context):
            return None
        match = _DURATION.search(segment)
        return match.group(0) if match else None
    if kind == "status":
        match = _ABSENCE.search(segment)
        return match.group(0) if match else None
    if kind == "multi_slot":
        return None
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
    if kind == "text":
        if not _constraints_present(question, segment, constraint_context):
            return None
        if _unknown_measurement_for_relation(question, segment):
            return None
    return segment


def _unknown_measurement_for_relation(question: str, segment: str) -> bool:
    unknown_measurement = _UNRECOGNIZED_MEASUREMENT.search(segment)
    if unknown_measurement is None:
        return False
    measurement_context = segment[: unknown_measurement.start()]
    return bool(
        _question_relation_tokens(question, "text")
        & _relation_tokens(measurement_context)
    )


def _time_value_context(question: str, segment: str) -> tuple[str, str] | None:
    requested = _requested_clock_values(question)
    candidates: list[tuple[int, int, str, str]] = []
    query_terms = _question_relation_tokens(question, "time")
    event_terms = query_terms - {
        "time", "record", "recorded", "list", "listed", "enter", "entered",
        "indicate", "indicated", "show", "shown", "date", "start", "end",
    }

    def event_is_supported(clause: str) -> bool:
        return not event_terms or bool(event_terms & _relation_tokens(clause))

    for match in _CLOCK.finditer(segment):
        value = f"{match.group(1)} {match.group(2)}".strip()
        normalized = (
            f"{match.group(1)} {match.group(2)}".casefold()
            if match.group(2)
            else match.group(1).casefold()
        )
        if requested and normalized not in requested:
            continue
        clause = _local_value_clause(segment, match.start(), match.end())
        overlap = len(query_terms & _relation_tokens(clause))
        candidates.append((overlap, match.start(), value, clause))
    if not candidates:
        return None
    if requested:
        if len(candidates) != 1 or not event_is_supported(candidates[0][3]):
            return None
        return candidates[0][2], candidates[0][3]
    if len(candidates) == 1:
        candidate = candidates[0]
        if not event_is_supported(candidate[3]):
            return None
        return candidate[2], candidate[3]
    best_overlap = max(candidate[0] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate[0] == best_overlap]
    if best_overlap == 0 or len(best) != 1:
        return None
    if not event_is_supported(best[0][3]):
        return None
    return best[0][2], best[0][3]


def _boolean_polarity(value: str | None) -> str | None:
    """Return assertion polarity independent of supporting wording."""
    if value is None:
        return None
    folded = _fold(value)
    if folded.startswith("no. "):
        return "no"
    if folded.startswith("yes. "):
        return "yes"
    return None


def _semantic_similarity(left: str, right: str, provider: EmbeddingProvider) -> float:
    left_vector, right_vector = provider.embed_many([left, right])
    value = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _semantic_relation_fallback(
    question: str,
    evidence: str,
    *,
    provider: EmbeddingProvider | None,
    threshold: float,
) -> bool:
    """Allow a local evidence clause to support a paraphrase missed lexically."""
    kind = _question_kind(question)
    lexical_anchor = _question_relation_tokens(question, kind) & _relation_tokens(evidence)
    return (
        provider is not None
        and provider.mode == "local-semantic-onnx-v1"
        and bool(lexical_anchor)
        and _semantic_similarity(question, evidence, provider) >= threshold
    )


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


def _relation_supported(
    question: str,
    segment: str,
    kind: str,
    *,
    provider: EmbeddingProvider | None = None,
    semantic_threshold: float = 0.48,
    constraint_context: str | None = None,
) -> bool:
    question_tokens = _question_relation_tokens(question, kind)
    segment_tokens = _relation_tokens(segment)
    if kind == "duration":
        requested_events = _event_scope_terms(question)
        evidence_events = _event_scope_terms(segment)
        if requested_events and not requested_events & evidence_events:
            return False
        if requested_events:
            return (
                _value_for(kind, segment, question, constraint_context=constraint_context)
                is not None
            )
    if kind == "boolean":
        action_anchors = {"access", "amend"} & question_tokens
        if action_anchors and not action_anchors & segment_tokens:
            return False
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
    if kind in {"field_value", "time"}:
        return _value_for(
            kind, segment, question, constraint_context=constraint_context
        ) is not None
    noise_by_kind = {
        "date": {"date", "effective", "deadline"},
        "amount": {
            "amount", "cost", "price", "prix", "how", "much", "apply", "charg",
            "constitue", "combien", "coute", "cout", "par", "hors", "sauf",
        },
        "duration": {"duration", "deadline", "period", "long", "soon", "quickly", "many"},
        "status": {"status", "statut", "recorded", "record", "consigne"},
        "severity": {"severity"},
        "responsible_people": {"owner"},
        "boolean": set(),
        "text": set(),
    }
    relation_tokens = question_tokens - noise_by_kind.get(kind, set())
    overlap = relation_tokens & segment_tokens
    if kind in {"amount", "duration", "status"}:
        value = _value_for(
            kind, segment, question, constraint_context=constraint_context
        )
        if value is None:
            return False
        pattern = _AMOUNT if kind == "amount" else _DURATION if kind == "duration" else _ABSENCE
        match = next(
            (item for item in pattern.finditer(segment) if _fold(item.group(0)) == _fold(value)),
            None,
        )
        context = _local_value_clause(segment, match.start(), match.end()) if match else segment
        query_terms = _question_relation_tokens(question, kind)
        if not query_terms:
            return True
        relation_overlap = query_terms & _relation_tokens(context)
        if bool(relation_overlap) and len(relation_overlap) / len(query_terms) >= 0.4:
            return True
        if kind == "amount" and len(list(_AMOUNT.finditer(segment))) != 1:
            return False
        if kind == "duration" and len(list(_DURATION.finditer(segment))) != 1:
            return False
        return _semantic_relation_fallback(
            question,
            context,
            provider=provider,
            threshold=semantic_threshold,
        )
    if kind == "date" and requested_role is None:
        if bool(relation_tokens) and len(overlap) >= min(2, len(relation_tokens)):
            return True
        if len(list(_DATE.finditer(segment))) != 1:
            return False
        return _semantic_relation_fallback(
            question,
            segment,
            provider=provider,
            threshold=semantic_threshold,
        )
    if kind == "boolean":
        return (
            len(overlap) >= 2
            or bool(overlap & {"amend"})
            or _has_related_absence(question, segment, kind)
            or _has_negated_relation(question, segment, kind)
        )
    if kind == "status":
        return bool(overlap) and _value_for(
            kind, segment, question, constraint_context=constraint_context
        ) is not None
    if kind == "multi_slot":
        return False
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


def _support_score(
    question: str,
    segment: str,
    kind: str,
    provider: EmbeddingProvider,
    *,
    constraint_context: str | None = None,
) -> float:
    query_tokens = lexical_tokens(question) - _QUESTION_NOISE
    segment_tokens = lexical_tokens(segment)
    lexical = len(query_tokens & segment_tokens) / max(1, len(query_tokens))
    semantic = _semantic_similarity(question, segment, provider)
    has_value = (
        _value_for(kind, segment, question, constraint_context=constraint_context) is not None
    )
    relation = float(
        _relation_supported(
            question,
            segment,
            kind,
            provider=provider,
            constraint_context=constraint_context,
        )
    )
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


def _page_contexts(ranked: list[RankedChunk]) -> dict[str, str]:
    grouped: dict[tuple[str, int], list[tuple[int, RankedChunk]]] = {}
    for index, item in enumerate(ranked):
        match = re.search(r"(?:^|:)c(\d+)$", item.chunk.id)
        order = int(match.group(1)) if match else index
        grouped.setdefault((item.chunk.document_id, item.chunk.page), []).append(
            (order, item)
        )
    contexts: dict[str, str] = {}
    for items in grouped.values():
        context = "\n".join(item.chunk.text for _, item in sorted(items, key=lambda row: row[0]))
        for _, item in items:
            contexts[item.chunk.id] = context
    return contexts


def _evidence_citations(
    item: RankedChunk,
    excerpt: str,
    context_headers: dict[str, tuple[RankedChunk, str]],
) -> tuple[Citation, ...]:
    header = context_headers.get(item.chunk.id)
    if header is None:
        return (_citation(item, excerpt),)
    header_item, header_excerpt = header
    header_citation = _citation(header_item, header_excerpt)
    field_citation = _citation(item, excerpt)
    if header_citation.chunk_id == field_citation.chunk_id and (
        header_citation.excerpt == field_citation.excerpt
    ):
        return (field_citation,)
    return (header_citation, field_citation)


def _field_name(kind: str) -> str:
    return {
        "boolean": "answer",
        "field_value": "answer",
        "time": "time",
        "date": "date",
        "amount": "important_amounts",
        "duration": "duration",
        "severity": "severity",
        "responsible_people": "responsible_people",
        "organization_name": "organization_name",
        "status": "status",
        "multi_slot": "answer",
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
        if kind == "multi_slot":
            return _abstained(
                "The question asks for multiple facts that this answer mode cannot verify together."
            )
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
        context_headers: dict[str, tuple[RankedChunk, str]] = {}
        field_absences: list[tuple[str, RankedChunk, float]] = []
        page_contexts = _page_contexts(ranked)
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
                    constraint_context = (
                        page_contexts.get(item.chunk.id, parent_content.text)
                        if kind in {"field_value", "date", "time"}
                        else parent_content.text
                    )
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
                    if kind == "field_value" and _has_requested_field_absence(
                        question, evidence_text
                    ):
                        if _constraints_present(
                            question, evidence_text, constraint_context
                        ):
                            field_absences.append((evidence_text, item, 0.5))
                            header = _unique_context_header(
                                question, evidence_text, constraint_context
                            )
                            if header:
                                header_item = next(
                                    (
                                        candidate
                                        for candidate in ranked
                                        if candidate.chunk.document_id == item.chunk.document_id
                                        and candidate.chunk.page == item.chunk.page
                                        and header in candidate.chunk.text
                                    ),
                                    None,
                                )
                                if header_item is not None:
                                    context_headers[item.chunk.id] = (header_item, header)
                        evaluated.setdefault(item.chunk.id, []).append(
                            (0.0, evidence_text, None, False)
                        )
                        continue
                    if kind not in {"boolean", "status"} and _has_negated_relation(
                        question, evidence_text, kind
                    ):
                        evaluated.setdefault(item.chunk.id, []).append(
                            (0.0, evidence_text, None, False)
                        )
                        continue
                    score = _support_score(
                        question,
                        evidence_text,
                        kind,
                        self.embeddings,
                        constraint_context=constraint_context,
                    )
                    value = _value_for(
                        kind,
                        evidence_text,
                        question,
                        constraint_context=constraint_context,
                    )
                    relation = _relation_supported(
                        question,
                        evidence_text,
                        kind,
                        provider=self.embeddings,
                        semantic_threshold=self.config.support_threshold,
                        constraint_context=constraint_context,
                    )
                    citation_text = evidence_text
                    if kind == "field_value":
                        extracted = _field_value(
                            question, evidence_text, constraint_context
                        )
                        if extracted is not None:
                            citation_text = extracted[1]
                            header = _unique_context_header(
                                question, evidence_text, constraint_context
                            )
                            if header:
                                header_item = next(
                                    (
                                        candidate
                                        for candidate in ranked
                                        if candidate.chunk.document_id == item.chunk.document_id
                                        and candidate.chunk.page == item.chunk.page
                                        and header in candidate.chunk.text
                                    ),
                                    None,
                                )
                                if header_item is not None:
                                    context_headers[item.chunk.id] = (header_item, header)
                    elif kind in {"time", "date"}:
                        if kind == "time":
                            time_context = _time_value_context(question, evidence_text)
                            if time_context is not None:
                                citation_text = time_context[1]
                        header = _unique_context_header(
                            question, evidence_text, constraint_context
                        )
                        if header:
                            header_item = next(
                                (
                                    candidate
                                    for candidate in ranked
                                    if candidate.chunk.document_id == item.chunk.document_id
                                    and candidate.chunk.page == item.chunk.page
                                    and header in candidate.chunk.text
                                ),
                                None,
                            )
                            if header_item is not None:
                                context_headers[item.chunk.id] = (header_item, header)
                    evaluated.setdefault(item.chunk.id, []).append(
                        (score, citation_text, value, relation)
                    )
                    if value is not None and relation:
                        candidates.append((score, citation_text, value, item))
                    elif (
                        score >= self.config.partial_support_threshold
                        and _has_explicit_relation(question, evidence_text)
                        and not _unknown_measurement_for_relation(question, evidence_text)
                        and not (kind == "date" and _asks_past_event_date(question))
                    ):
                        partials.append((score, evidence_text, item))
        assessments = self._candidate_assessments(kind, ranked, evaluated)
        candidates.sort(key=lambda value: (-value[0], -value[3].score, value[3].chunk.id))
        supported = [item for item in candidates if item[0] >= self.config.support_threshold]
        if field_absences:
            if supported:
                positive = supported[0]
                absence_excerpt, absence_item, absence_score = field_absences[0]
                citations = tuple(
                    dict.fromkeys(
                        (
                            *_evidence_citations(positive[3], positive[1], context_headers),
                            *_evidence_citations(absence_item, absence_excerpt, context_headers),
                        )
                    )
                )
                result = GroundedAnswer(
                    status="ambiguous",
                    answerable=False,
                    answer=(
                        "Conflicting presence and absence evidence is supported; "
                        "review the cited passages."
                    ),
                    confidence=min(positive[0], absence_score),
                    supporting_document=positive[3].chunk.document_id,
                    supporting_page=positive[3].chunk.page,
                    supporting_excerpt=positive[1],
                    ambiguity_reason=(
                        "The same requested field is both stated and explicitly "
                        "marked absent."
                    ),
                    extracted_fields={},
                    supporting_chunk_ids=tuple(
                        dict.fromkeys(citation.chunk_id for citation in citations)
                    ),
                    citations=citations,
                    candidate_assessments=assessments,
                )
                validate_grounded_answer(result, ranked)
                return result
            return _abstained(candidate_assessments=assessments)
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
            if (
                _boolean_polarity(other[2]) != _boolean_polarity(best[2])
                if kind == "boolean"
                else _fold(other[2] or "") != _fold(best[2] or "")
            )
        ]
        if distinct and kind in {
            "boolean",
            "date",
            "amount",
            "duration",
            "severity",
            "field_value",
            "time",
        }:
            other = distinct[0]
            citations = tuple(
                dict.fromkeys(
                    (
                        *_evidence_citations(best[3], best[1], context_headers),
                        *_evidence_citations(other[3], other[1], context_headers),
                    )
                )
            )
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
                supporting_chunk_ids=tuple(
                    dict.fromkeys(citation.chunk_id for citation in citations)
                ),
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
            supporting_chunk_ids=tuple(
                dict.fromkeys(
                    citation.chunk_id
                    for citation in _evidence_citations(
                        best[3], best[1], context_headers
                    )
                )
            ),
            citations=_evidence_citations(best[3], best[1], context_headers),
            candidate_assessments=assessments,
        )
        validate_grounded_answer(result, ranked)
        return result
