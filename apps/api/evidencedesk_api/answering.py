"""Grounded, deterministic answer decisions for the no-key EvidenceDesk mode.

The module deliberately separates four outcomes: supported, partially supported,
ambiguous, and absent evidence.  Document text is always data, never an instruction.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from itertools import pairwise
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
_UNVERIFIED_SOURCE_GAP = "[unverified source gap]"


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
    r"(?:ms|s|min|h|Hz|kHz|MHz|GHz|rpm|W|kW|MW|V|mV|A|mA|kWh|Mbps|Gbps)|"
    r"percent|per\s+cent|%"
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
    r"(?:(?:calendar|business|working)\s+)?(?:minutes?|hours?|days?|heures?|jours?)\b",
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
    "a",
    "an",
    "at",
    "can",
    "de",
    "does",
    "how",
    "if",
    "is",
    "le",
    "la",
    "on",
    "the",
    "was",
    "what",
    "when",
    "which",
    "whose",
    "who",
    "where",
    "combien",
    "why",
    "may",
    "might",
    "will",
    "would",
    "qui",
    "quand",
    "comment",
    "pourquoi",
    "ou",
    "les",
    "un",
    "master",
    "services",
    "service",
    "incident",
    "data",
    "processing",
    "personal",
    "security",
    "review",
    "platform",
    "une",
    "des",
    "du",
    "au",
    "aux",
    "quel",
    "quelle",
    "quels",
    "quelles",
}

_DOCUMENT_TYPE_TERMS = {
    "agreement": {"agreement", "contract", "contrat"},
    "addendum": {"addendum", "annex", "amendment"},
    "report": {"report"},
    "register": {"register"},
    "permit": {"permit", "permis"},
    "plan": {"plan"},
}

_SOURCE_IDENTITY_NOUNS = re.compile(
    r"\b(?:agreement|annex|case|change\s+record|contract|delivery\s+record|"
    r"delivery|document|file|handover|invoice|log|note|order|plan|permit|purchase\s+order|"
    r"receipt|record|register|report|service\s+record|sheet|shipment|tally|"
    r"work\s+order)\b",
    re.IGNORECASE,
)

_RELATION_EQUIVALENTS = {
    "entry": {"entry", "entree", "entrance"},
    "start": {
        "start",
        "starts",
        "starting",
        "begin",
        "begins",
        "beginning",
        "commence",
        "commences",
        "commencer",
        "debut",
        "debute",
        "demarre",
    },
    "finish": {
        "finish",
        "finishes",
        "finished",
        "complete",
        "completes",
        "completed",
        "end",
        "ends",
        "ending",
        "termine",
        "terminer",
        "terminee",
        "acheve",
    },
    "replace": {
        "replace",
        "replaces",
        "replaced",
        "replacing",
        "replacement",
        "replac",
        "remplace",
        "remplacer",
        "remplacement",
        "remplacee",
        "remplac",
    },
    "battery": {"battery", "batteries", "batterie", "accumulator"},
    "reserve": {"reserve", "reserves", "fund", "funds", "fond", "fonds", "pool"},
    "maintenance": {"maintenance", "entretien"},
    "coverage": {"coverage", "cover", "couverture"},
    "precedence": {
        "control",
        "controls",
        "govern",
        "governs",
        "prevail",
        "prevails",
        "precedence",
        "priority",
        "replace",
        "replaces",
        "supersede",
        "supersedes",
    },
    "amend": {
        "amend",
        "amends",
        "amended",
        "amendment",
        "change",
        "changes",
        "changed",
        "modify",
        "modifies",
        "modified",
        "alter",
        "alters",
        "altered",
    },
    "access": {"access", "accesses", "accessed", "accessing", "acces"},
    "storage": {"storage", "stockage", "archivage", "archiving"},
    "restoration": {
        "restore",
        "restores",
        "restored",
        "restoration",
        "recovery",
        "recover",
        "retablissement",
    },
    "monthly": {
        "monthly",
        "month",
        "months",
        "mois",
        "mensuel",
        "mensuelle",
        "mensuels",
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


def _has_relation_token_overlap(left: set[str], right: set[str]) -> bool:
    return any(
        first == second
        or (len(first) >= 5 and second.startswith(first))
        or (len(second) >= 5 and first.startswith(second))
        for first in left
        for second in right
    )


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()


def _requested_document_type(question: str) -> str | None:
    folded = _fold(question)
    occurrences = sorted(
        [
            (match.start(), match.end(), kind, term)
            for kind, terms in _DOCUMENT_TYPE_TERMS.items()
            for term in terms
            for match in re.finditer(rf"\b{re.escape(term)}\b", folded)
        ]
    )
    if not occurrences:
        return None
    eligible_occurrences = []
    source_cues = []
    for start, end, kind, term in occurrences:
        prefix = folded[max(0, start - 32) : start]
        suffix = folded[end : end + 32]
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
        return next(kind for kind, terms in _DOCUMENT_TYPE_TERMS.items() if term in terms)
    if not eligible_occurrences:
        return None
    if len({kind for _, _, kind, _ in eligible_occurrences}) == 1:
        start, end, kind, _ = eligible_occurrences[0]
        prefix = folded[:start]
        # In "Which register does the field log accompany?", register is the
        # requested answer type, not a source restriction. Keep other implicit
        # source modifiers such as "the contract operator" or "invoice total".
        answer_type_head = re.match(
            r"^\s*(?:which|what|quel|quelle|quels|quelles)\s+"
            r"(?:(?:the|le|la|les|l)\s+)?(?:[a-z0-9-]+\s+){0,2}$",
            prefix,
        ) and not re.match(
            r"^\s*(?:what|quel|quelle|quels|quelles)\s+"
            r"(?:is|are|was|were|est|sont|etait|etaient)\b",
            prefix,
        )
        if answer_type_head:
            return None
        if kind == "report":
            # "Report" is also a common verb; a bare verb does not set source scope.
            return None
        suffix = folded[end : end + 32]
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
        kind = next(kind for kind, terms in _DOCUMENT_TYPE_TERMS.items() if source_term in terms)
        return {kind}
    phrase = match.group(1).strip()
    terms = set(re.findall(r"[a-z0-9]+", phrase))
    for identifier in _EXPLICIT_ID.finditer(phrase):
        terms.difference_update(re.findall(r"[a-z0-9]+", _fold(identifier.group(1))))
    normalized_types = {kind for kind, aliases in _DOCUMENT_TYPE_TERMS.items() if terms & aliases}
    if normalized_types:
        terms = {
            term
            for term in terms
            if not any(term in aliases for aliases in _DOCUMENT_TYPE_TERMS.values())
        }
        terms.update(normalized_types)
    return terms - {
        "document",
        "source",
        "itself",
        "lui",
        "meme",
        "directly",
        "explicitly",
        "the",
        "le",
        "la",
        "les",
        "un",
        "une",
    }


def _requested_source_ids(question: str) -> set[str]:
    folded = _fold(question)
    match = re.search(
        r"\b(?:according\s+to|per|selon|d['\u2019]apres|"
        r"in\s+(?:the\s+)?(?:agreement|contract|invoice|handover|report|register|"
        r"permit|plan|shipment|delivery\s+record|field\s+log)|"
        r"in|from|under)\s+"
        r"(?:(?:the|le|la|les|l['\u2019])\s+)?"
        r"(?:the\s+|le\s+|la\s+|l['\u2019])?([a-z0-9 _/-]{1,48})",
        folded,
    )
    phrase_ids: set[str] = set()
    if match is not None:
        phrase = re.split(
            r"\b(?:itself|directly|explicitly|states?|says?|lists?|indique|mentionne|"
            r"what|which|who|whose|is|are|was|were|does|do|did|has|have|that|the)\b",
            match.group(1),
            maxsplit=1,
        )[0]
        phrase_ids = {
            _fold(identifier.group(1)) for identifier in _EXPLICIT_ID.finditer(phrase)
        }
        if match.group(0).startswith(("in ", "from ", "under ")) and not re.match(
            r"\s*(?:the\s+)?(?:agreement|contract|invoice|handover|report|register|"
            r"permit|plan|shipment|delivery\s+record|field\s+log)\b",
            phrase,
        ):
            # Bare prepositions often introduce a record constraint rather
            # than a source document ("under WorkOrderID ...", "in lot ...").
            # Promote them to source scope only when a source-type noun makes
            # that reading explicit, or when the ID follows the preposition
            # directly. Explicit markers such as "according to" remain valid.
            first_id = _EXPLICIT_ID.search(phrase)
            prefix = phrase[: first_id.start()] if first_id is not None else phrase
            local_record_noun = re.search(
                r"\b(?:record|lot|batch|sample|sensor|work\s*order|check|account|asset)\b",
                prefix,
            )
            if local_record_noun or (
                prefix.strip()
                and _SOURCE_IDENTITY_NOUNS.search(prefix) is None
            ):
                phrase_ids.clear()
    if phrase_ids:
        return phrase_ids
    # A named source can be the grammatical subject of a reporting verb:
    # “what does handover HX-24 report?” In that form, the source ID is
    # the sole typed ID in the subject, before the predicate.
    source_subject = re.search(
        r"\b(?:does|did|do)\s+(?P<subject>[a-z0-9 /_-]{1,100}?)\s+"
        r"(?:reports?|states?|says?|lists?|records?|indicates?|shows?|"
        r"confirms?|describes?|documents?)\b",
        folded,
    )
    if source_subject is None or _SOURCE_IDENTITY_NOUNS.search(
        source_subject.group("subject")
    ) is None:
        return set()
    subject_ids = {
        _fold(identifier.group(1))
        for identifier in _EXPLICIT_ID.finditer(source_subject.group("subject"))
    }
    return subject_ids if len(subject_ids) == 1 else set()


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


def _principal_identity(text: str) -> str:
    """Keep only an ID attached directly to the first source-type noun."""
    first_id = _EXPLICIT_ID.search(text)
    if first_id is not None:
        source_noun = _SOURCE_IDENTITY_NOUNS.search(text[: first_id.start()])
        if source_noun is None:
            return ""
        gap = text[source_noun.end() : first_id.start()]
        if not re.fullmatch(
            r"[\s,;:#()\-]*(?:(?:id|number|no\.?)\s*)?", gap, re.IGNORECASE
        ):
            return ""
        # The first typed ID belongs to the document/source head. IDs later in
        # the sentence can be linked agreements, shipments, or references.
        return text[: first_id.end()].strip(" \t,;:-")
    # Without an ID, only a short heading can establish a textual source name;
    # a sentence that mentions another document cannot set primary identity.
    if len(text.split()) <= 8 and not re.search(r"[.!?]", text):
        return text.strip()
    return ""


def _named_subject_tokens(question: str) -> set[str]:
    tokens: set[str] = set()
    requested_field = _requested_field_words(question)
    # Keep a slash-separated identifier intact so its base cannot also match
    # a competing record with a different suffix.
    for match in re.finditer(
        r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ']*(?:-[A-Za-z0-9]+)*(?:/[A-Za-z0-9-]+)*\b",
        question,
    ):
        raw = match.group(0)
        token = re.sub(r"(?:'s|s')$", "", _fold(raw))
        expanded = _camel_words(raw)
        typed_label = bool(re.search(r"(?:ID|Code|Tag|Ref|Number)$", raw))
        field_word = bool(
            typed_label and requested_field and set(_relation_tokens(expanded)) & requested_field
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
    return set(_relation_tokens(" ".join(_requested_field_head(question))))


def _requested_field_head(question: str) -> list[str]:
    """Remove predicate words from the requested field phrase."""
    parts = _requested_field_parts(question)
    predicate_words = {
        "applies",
        "apply",
        "used",
        "use",
        "reported",
        "reports",
        "recorded",
        "records",
        "listed",
        "lists",
        "entered",
        "issued",
        "given",
        "assigned",
        "accompany",
        "accompanies",
    }
    while parts and parts[-1] in predicate_words:
        parts.pop()
    return parts


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
            "a",
            "an",
            "the",
            "le",
            "la",
            "les",
            "de",
            "du",
            "des",
            "d",
            "for",
            "in",
            "on",
            "at",
            "to",
            "under",
            "from",
            "by",
            "with",
            "of",
            "is",
            "are",
            "was",
            "were",
            "does",
            "do",
            "did",
            "has",
            "have",
            "est",
            "sont",
            "ont",
            "il",
            "elle",
            "ils",
            "elles",
            "lequel",
            "laquelle",
            "printed",
            "listed",
            "recorded",
            "approved",
            "approve",
            "set",
            "sets",
            "indicated",
            "begins",
            "commence",
            "reported",
            "donne",
            "indique",
            "indiquee",
            "indiquees",
            "note",
            "notee",
            "explicitly",
            "directly",
            "is-il",
            "est-il",
            "est-elle",
            "sont-ils",
            "sont-elles",
            "donne-t-il",
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


def _record_constraint_values(question: str) -> set[str]:
    """Prefer IDs explicitly attached to record types over product/model codes."""
    typed_prefix = re.compile(
        r"(?:agreement|contract|receipt|invoice|work\s*order|change\s*request|"
        r"service\s*order|sensor(?:\s*id)?|record(?:\s*id)?|batch\s*id|"
        r"permit(?:\s*id)?|plan\s*id|report\s*id|under)\s*(?:id\s*)?$",
        re.IGNORECASE,
    )
    typed_values: set[str] = set()
    for match in _EXPLICIT_ID.finditer(question):
        prefix = question[max(0, match.start() - 40) : match.start()]
        if typed_prefix.search(prefix):
            typed_values.add(_fold(match.group(1)))
    return typed_values or _requested_identifier_values(question)


def _requested_clock_values(question: str) -> set[str]:
    values: set[str] = set()
    for match in _CLOCK.finditer(question):
        value = match.group(1)
        zone = match.group(2)
        values.add(f"{value} {zone}".casefold() if zone else value.casefold())
    return values


def _requested_conditions(question: str) -> set[str]:
    requested_ids = _requested_identifier_values(question)
    conditions = set()
    for match in _CONDITION.finditer(question):
        if requested_ids & {
            _fold(identifier.group(1)) for identifier in _EXPLICIT_ID.finditer(match.group(0))
        }:
            continue
        conditions.add(re.sub(r"\s+", " ", _fold(match.group(1))).strip())
    return {
        condition
        for condition in conditions
        if not re.search(
            r"\b(?:agreement|contract|addendum|policy|accord|contrat|avenant)\b",
            condition,
        )
        and not re.search(r"\b(?:one|must|selected|controls|which|deadline|delai)\b", condition)
    }


def _requested_person_terms(question: str) -> set[str]:
    """Return named people asked about, excluding typed record identifiers."""
    terms = set()
    for match in _PERSON.finditer(question):
        name = match.group(1)
        if re.search(r"(?:ID|Code|Tag|Ref|Number)\b", name, re.IGNORECASE):
            continue
        if _EXPLICIT_ID.search(name):
            continue
        terms.update(_relation_tokens(name))
    return terms


def _provisional_amount_status(evidence: str, document_context: str) -> str | None:
    """Find a same-record source statement that explicitly leaves an amount open."""
    evidence_ids = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(evidence)}
    if not evidence_ids or not _AMOUNT.search(evidence):
        return None
    active_ids: set[str] = set()
    for sentence in _segments(document_context):
        sentence_ids = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(sentence)}
        same_record_reference = bool(
            re.search(
                r"\b(?:same\s+(?:note|record|report|reconciliation)|its\s+author|"
                r"it\s+(?:does\s+not|doesn't)|this\s+note)\b",
                _fold(sentence),
            )
        )
        if sentence_ids:
            if same_record_reference:
                active_ids.update(sentence_ids)
            else:
                active_ids = sentence_ids
        elif not same_record_reference and not re.match(
            r"\s*(?:it|its|the author|this note|both figures|both amounts)\b",
            _fold(sentence),
        ):
            active_ids = set()
        if not re.search(
            r"\b(?:to\s+confirm|pending\s+confirmation|awaiting\s+confirmation|"
            r"unconfirmed|provisional|not\s+final|a\s+confirmer|a\s+valider|non\s+confirme|"
            r"does\s+not\s+explain\s+the\s+difference|doesn't\s+explain\s+the\s+difference|"
            r"n'explique\s+pas\s+la\s+difference)\b",
            _fold(sentence),
        ):
            continue
        if re.search(
            r"\b(?:both\s+(?:figures|amounts)|les\s+deux\s+(?:chiffres|montants))\b",
            _fold(sentence),
        ):
            if evidence_ids & active_ids:
                return sentence
        elif evidence_ids & (
            sentence_ids if sentence_ids else active_ids if same_record_reference else set()
        ):
            return sentence
    return None


def _related_amount_proof(
    question: str,
    provisional_value: str,
    provisional_item: RankedChunk,
    ranked: list[RankedChunk],
    document_contexts: dict[str, str],
) -> tuple[RankedChunk, str] | None:
    """Find a different amount for the same typed record and requested role."""
    requested_role = _amount_role(question)
    requested_ids = _requested_identifier_values(question)
    if requested_role is None or not requested_ids:
        return None
    for candidate in ranked:
        if candidate.chunk.document_id == provisional_item.chunk.document_id:
            continue
        context = document_contexts.get(candidate.chunk.id, candidate.chunk.text)
        for segment in _segments(candidate.chunk.text):
            matches = list(_AMOUNT.finditer(segment))
            if (
                len(matches) != 1
                or _amount_role(segment) != requested_role
                or not _constraints_present(question, segment, context)
            ):
                continue
            value = matches[0].group(0).strip()
            if _fold(value) == _fold(provisional_value):
                continue
            excerpt = _local_value_clause(segment, matches[0].start(), matches[0].end())
            if excerpt:
                return candidate, excerpt
    return None


def _amount_role(text: str) -> str | None:
    """Classify the economic role of a monetary value from local wording."""
    folded = _fold(text)
    if re.search(
        r"\b(?:unit\s+price|price\s+per|cost\s+per|per\s+unit|per\s+item|"
        r"per\s+component|prix\s+unitaire|cout\s+par\s+unite)\b",
        folded,
    ):
        return "unit_price"
    if re.search(
        r"\b(?:invoice|invoiceid|facture)\b.{0,90}\b(?:bill(?:s|ed)?|total|amount\s+due|"
        r"pre[- ]?(?:tax|vat)|before\s+(?:tax|vat))\b|"
        r"\b(?:total\s+(?:value|amount)|pre[- ]?(?:tax|vat)\s+total|"
        r"invoice\s+total|amount\s+due|(?:amount\s+)?(?:billed|invoiced|charged)\s+"
        r"(?:to|for)?|total\s+billed|billed\s+amount|"
        r"valeur\s+totale|montant\s+total)\b",
        folded,
    ):
        return "invoice_total"
    return None


def _amount_intent_supported(question: str, evidence: str) -> bool:
    """Require local wording for requested contractual or transaction status."""
    folded_question = _fold(question)
    folded_evidence = _fold(evidence)
    if re.search(r"\b(?:agreed|contractual|stipulated|convenu|convenue)\b", folded_question):
        status_words = r"(?:agreed|contractual|stipulated|convenu|convenue)"
        if re.search(
            rf"\b(?:not|never|without|pas|non)\b.{{0,24}}\b{status_words}\b|"
            rf"\b{status_words}\b.{{0,24}}\b(?:not|never|without|pas|non)\b",
            folded_evidence,
        ):
            return False
        return bool(
            re.search(
                r"\b(?:agreed|contractual|stipulated|convenu|convenue|"
                r"price\s+was\s+set|prix\s+a\s+ete\s+fixe)\b",
                folded_evidence,
            )
        )
    if re.search(
        r"\bactually\s+(?:invoiced|billed|charged)\b|"
        r"\bamount\s+(?:that\s+was\s+)?(?:invoiced|billed|charged)\b|"
        r"\b(?:amount|total)\s+(?:was|has\s+been)\s+(?:invoiced|billed|charged)\b",
        folded_question,
    ):
        event_words = r"(?:bill(?:s|ed)?|invoic(?:e|ed|es)|charg(?:e|ed|es))"
        if re.search(
            rf"\b(?:not|never|without|pas|non)\b.{{0,24}}\b{event_words}\b|"
            rf"\b(?:was|were|did|does|has|have)\s+not\b.{{0,16}}\b{event_words}\b",
            folded_evidence,
        ):
            return False
        event = re.search(
            r"\bbills?\b|\b(?:was|were|has\s+been|have\s+been)\s+"
            r"(?:billed|invoiced|charged)\b|"
            r"\b(?:billed|invoiced|charged)\s+(?:to|for)\b|"
            r"\b(?:amount|total)\s+(?:was\s+)?(?:billed|invoiced|charged)\b",
            folded_evidence,
        )
        amount = _AMOUNT.search(evidence)
        # A transaction verb needs a nearby monetary value; an invoice
        # mentioned in prose or a note that merely gives a total is not proof
        # that the amount was actually billed.
        return bool(
            event is not None
            and amount is not None
            and abs(amount.start() - event.start()) <= 100
        )
    return True


def _identifier_prefix(value: str) -> str:
    return re.split(r"[-/]", value, maxsplit=1)[0]


def _constraints_present(
    question: str, evidence: str, constraint_context: str | None = None
) -> bool:
    # Explicit source IDs are enforced by source-document scoping. Do not
    # require them to be repeated in each fact sentence; retain other IDs as
    # record/object constraints.
    source_ids = _requested_source_ids(question)
    requested_ids = _record_constraint_values(question) - source_ids
    all_requested_ids = _requested_identifier_values(question) - source_ids
    object_ids = all_requested_ids - requested_ids
    evidence_ids = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(evidence)}
    context = constraint_context or evidence
    missing_local_id = False
    for requested_id in requested_ids:
        local_matches = {
            value
            for value in evidence_ids
            if _identifier_prefix(value) == _identifier_prefix(requested_id)
        }
        if requested_id in local_matches:
            if local_matches != {requested_id} and not _explicit_same_record_reference(
                evidence, requested_id, local_matches - {requested_id}
            ):
                return False
        else:
            missing_local_id = True
    for object_id in object_ids:
        # Product/model IDs are object constraints, not dossier anchors. Their
        # exact value must remain in the same record context and cannot be
        # replaced by a neighboring product with a similar identifier.
        context_ids = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(context)}
        if object_id not in evidence_ids and object_id not in context_ids:
            return False
        local_variants = {
            value
            for value in evidence_ids
            if _identifier_prefix(value) == _identifier_prefix(object_id)
        }
        if local_variants and object_id not in local_variants:
            return False
    if missing_local_id and _context_id_proofs(question, evidence, context) is None:
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


def _explicit_same_record_reference(
    evidence: str, requested_id: str, companion_ids: set[str]
) -> bool:
    """Allow companion identifiers only when the source explicitly equates records."""
    if not companion_ids:
        return False
    folded = _fold(evidence)
    if not re.search(r"\bsame\b.{0,48}\b(?:reading|record|entry|shipment|order)\b", folded):
        return False
    ids = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(evidence)}
    return requested_id in ids and companion_ids <= ids and bool(
        re.search(r"\b(?:as|equivalent\s+to|refers\s+to)\b", folded)
    )


def _field_value(
    question: str, segment: str, constraint_context: str | None = None
) -> tuple[str, str] | None:
    """Extract one explicitly labeled value with all query constraints present."""
    field_parts = _requested_field_parts(question)
    if not field_parts or not _constraints_present(question, segment, constraint_context):
        return None
    # A colon assignment may contain a linking noun omitted from the query
    # (supplier account code for supplier code). Require every requested slot
    # term in order and an explicit assignment before extracting an ID.
    requested_head = _requested_field_head(question)
    if requested_head and requested_head[-1] in {"code", "tag", "ref", "reference", "number"}:
        label = r"\s+".join(
            re.escape(part) + r"(?:\s+[a-z][a-z-]*){0,2}" for part in requested_head
        )
        assignment = re.search(
            rf"(?i)(?<![A-Za-z0-9]){label}\s*:\s*({_EXPLICIT_ID.pattern})",
            segment,
        )
        if assignment:
            value_match = _EXPLICIT_ID.fullmatch(assignment.group(1))
            if value_match:
                return value_match.group(1), segment.strip()
    aliases = {
        "modele": "model",
        "turbidite": "turbidity",
        "maximum": "peak",
        "maximale": "peak",
        "maximal": "peak",
        "laboratoire": "laboratory",
        "armoire": "cupboard",
        "humidite": "humidity",
        "concentration": "",
        "level": "",
        "reading": "",
        "readings": "",
        "value": "",
    }
    target_options = [field_parts]
    canonical_parts = [aliases.get(part, part) for part in field_parts]
    canonical_parts = [part for part in canonical_parts if part]
    if "humidity" in canonical_parts and "target" in canonical_parts:
        target_options.append(["target", "relative", "humidity"])
    if "cupboard" in canonical_parts:
        target_options.append(["conservation", "cupboard"])
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
        field_matches.extend(re.finditer(label_pattern(canonical_parts), segment, re.IGNORECASE))
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
    if not field_matches and set(canonical_parts) & {"temperature", "temp"}:
        # A temperature reading can be directly identified by its physical
        # unit even when the source uses the verb "read" instead of repeating
        # the field label.
        temperature_values = list(
            re.finditer(
                r"(?<![\w/.-])(\d+(?:[.,]\d+)?)\s*(°\s*[CF]|degrees?\s+"
                r"(?:Celsius|Fahrenheit)|deg(?:rees?)?\s*[CF])(?![\w])",
                segment,
                re.IGNORECASE,
            )
        )
        if (
            len(temperature_values) == 1
            and re.search(
                r"\b(?:read|reads|recorded|measured|measurement|reading)\b",
                segment,
                re.I,
            )
            and (
                not _requested_clock_values(question)
                or _requested_clock_values(question)
                & {
                    _normalize_clock(match.group(1), match.group(2))
                    for match in _CLOCK.finditer(segment)
                }
            )
        ):
            measured = temperature_values[0]
            return measured.group(0).strip(), segment.strip()
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
    requested_head = _requested_field_head(question)
    identifier_field = bool(
        re.match(
            r"^\s*(?:what|which|quel|quelle|quels|quelles)\s+"
            r"(?:(?:is|are|was|were|est|etait|etaient)\s+)?"
            r"(?:(?:the|le|la|les|l)\s+)?[A-Z][A-Za-z0-9]*ID\b",
            question,
            re.IGNORECASE,
        )
        or (requested_head and requested_head[-1] in {"code", "tag", "ref", "reference", "number"})
    )
    identifier_match = _EXPLICIT_ID.search(tail) if identifier_field else None
    if measurement:
        value = measurement.group(0).strip()
    elif amount := _AMOUNT.match(tail):
        # Keep a currency amount together, including thousands separators,
        # when it appears as a labeled text-field value.
        value = amount.group(0).strip()
    elif identifier_match:
        value = identifier_match.group(1)
    else:
        number = re.match(r"\s*(\d+(?:[.,]\d+)?)\b", tail)
        if number and set(field_parts) & {"ph", "score", "rating"}:
            value = number.group(1)
        else:
            text_match = re.match(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9 .'\u2019/-]*", tail)
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
    source_measurement, unknown_measurement_precedes = _measurement_after_label(original_tail)
    if unknown_measurement_precedes:
        return None
    if source_measurement:
        value = source_measurement.group(0).strip()
    original_value = re.search(re.escape(value), original[original_match.end() :], re.IGNORECASE)
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
    row_ids = _requested_identifier_values(original)
    row_link = (
        "|" in original
        and requested_ids <= row_ids
        and all(
            len(
                {
                    identifier
                    for identifier in row_ids
                    if _identifier_prefix(identifier) == _identifier_prefix(requested_id)
                }
            )
            == 1
            for requested_id in requested_ids
        )
    )
    context_proofs = _context_id_proofs(question, assertion, constraint_context)
    if not requested_ids <= assertion_ids and context_proofs is None and not row_link:
        return None
    if not _requested_conditions(question) <= {
        condition for condition in _requested_conditions(question) if condition in assertion_folded
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
    end = offset + max(original_match.end() + original_value.end(), assertion_end)
    assertion_source_start = offset + assertion_start
    assertion_source_end = offset + assertion_end
    for requested_id in _requested_identifier_values(question):
        id_match = re.search(
            rf"(?<![A-Z0-9]){re.escape(requested_id)}(?![A-Z0-9])",
            source,
            re.IGNORECASE,
        )
        if id_match and (
            assertion_source_start <= id_match.start() < assertion_source_end or row_link
        ):
            clause = _local_value_clause(source, id_match.start(), id_match.end())
            clause_start = source.find(clause)
            start = min(start, clause_start if clause_start >= 0 else id_match.start())
            end = max(
                end,
                clause_start + len(clause) if clause_start >= 0 else id_match.end(),
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
    if ":" in source_excerpt:
        colon = source_excerpt.find(":")
        requested_in_prefix = requested_ids & {
            _fold(identifier.group(1))
            for identifier in _EXPLICIT_ID.finditer(source_excerpt[:colon])
        }
        if requested_in_prefix and match.start() > colon:
            source_excerpt = source_excerpt[colon + 1 :].strip()
            source_excerpt = source_excerpt.removesuffix(".").strip()
    return value, source_excerpt


def _requested_relation_family(question: str) -> str | None:
    folded = _fold(question)
    reference_verb = re.search(
        r"\b(?:refer(?:ence(?:s|d)?|s|red|ring)?|cite(?:s|d|ing)?|"
        r"renvoie(?:nt|r)?|renvoyer)\b",
        folded,
    )
    if reference_verb and (
        reference_verb.group(0) != "reference"
        or re.search(r"\b(?:does|do|did)\b.{0,80}$", folded[: reference_verb.start()])
        or re.search(r"\bexplicitly\b.{0,50}$", folded[: reference_verb.start()])
    ):
        return "reference"
    if re.search(r"\b(?:assign\w*|attribu\w*)\b", folded):
        return "assign"
    if re.search(r"\b(?:bind\w*|lier\w*|relier\w*)\b", folded):
        return "bind"
    return None


def _directed_field_relation(question: str, segment: str, value: str) -> bool:
    """Keep an explicitly requested record relation and its direction local."""
    family = _requested_relation_family(question)
    if family is None:
        return True
    if (
        family == "reference"
        and re.search(
            r"\b(?:is|are|was|were|been)?\s*(?:referenced|cited)\s+by\b",
            _fold(segment),
        )
        and re.search(r"\b(?:does|do|did)\b", _fold(question))
    ):
        return False
    predicates = {
        "reference": (
            r"\b(?:refer(?:ence(?:s|d)?|s|red|ring)?|cite(?:s|d|ing)?|"
            r"renvoie(?:nt|r)?|renvoyer)\b"
        ),
        "assign": r"\b(?:assign\w*|attribu\w*)\b",
        "bind": r"\b(?:bind\w*|lier\w*|relier\w*)\b",
    }
    field_parts = _requested_field_parts(question)
    if not field_parts:
        return False
    label = (
        r"(?<![A-Za-z0-9])"
        + r"[\s_-]*".join(re.escape(part) for part in field_parts)
        + r"(?:[\s_-]*id)?(?![A-Za-z0-9])"
    )
    clauses = _assertion_clauses(segment)
    for clause in clauses:
        predicate = re.search(predicates[family], clause, re.IGNORECASE)
        field = re.search(label, clause, re.IGNORECASE)
        found_value = re.search(re.escape(value), clause, re.IGNORECASE)
        if (
            predicate is not None
            and field is not None
            and found_value is not None
            and predicate.start() < field.start() <= found_value.start()
        ):
            return True
    return False


def _subject_matches(subject: str, contents: str) -> bool:
    folded_subject = _fold(subject)
    folded_contents = _fold(contents)
    if re.fullmatch(r"[a-z]{1,8}-\d+[a-z0-9/-]*", folded_subject):
        return (
            re.search(
                rf"(?<![a-z0-9]){re.escape(folded_subject)}(?![a-z0-9/-])",
                folded_contents,
            )
            is not None
        )
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
    context_by_document: dict[str, str] = {}
    primary_text_by_document: dict[str, str] = {}
    for document_id, items in grouped.items():
        ordered, source_context = _ordered_context(items)
        context_by_document[document_id] = "\n".join(
            [items[0].chunk.document_name, source_context]
        )
        # An opaque UUID is not source order. Use a text header as the
        # document identity only when every retrieved chunk has an explicit
        # ordinal/source chunk index, or the document has a single chunk.
        has_source_order = len(items) == 1 or all(
            item.chunk.ordinal is not None
            or re.search(r"(?:^|:)c\d+$", item.chunk.id) is not None
            for item in items
        )
        primary_text_by_document[document_id] = (
            ordered[0].chunk.text if ordered and has_source_order else ""
        )
    allowed: set[str] = set()
    subjects_found_globally: set[str] = set()
    fully_scoped_documents: set[str] = set()
    eligible_documents: set[str] = set()
    link_keys_by_document: dict[str, set[tuple[str, str]]] = {}
    for document_id, items in grouped.items():
        contents = context_by_document[document_id]
        identity_lines = [items[0].chunk.document_name]
        title = primary_text_by_document[document_id]
        if title:
            principal_identity = _principal_identity(title)
            if principal_identity:
                identity_lines.append(principal_identity)
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
            overlap = {subject for subject in subject_tokens if _subject_matches(subject, contents)}
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
        clause for part in re.split(r"[;,]", _fold(segment)) for clause in _assertion_clauses(part)
    ]
    return any(
        relation_tokens & _relation_tokens(clause) and _has_explicit_absence(clause)
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
        clause for part in re.split(r"[;,]", folded_segment) for clause in _assertion_clauses(part)
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
        (boundary_text.rfind(mark, 0, start) + 1 for mark in (";", "|", ".", "!", "?", "\n")),
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
    remainder = re.sub(r"\b[A-Za-z][A-Za-z0-9]*(?:ID|Code|Tag|Ref|Number)\b", "", remainder)
    return re.sub(r"[\s|:;,./-]+", "", remainder) == ""


def _has_unique_context_header(question: str, segment: str, constraint_context: str | None) -> bool:
    return _unique_context_header(question, segment, constraint_context) is not None


def _unique_context_header(
    question: str, segment: str, constraint_context: str | None
) -> str | None:
    proofs = _context_id_proofs(question, segment, constraint_context)
    return proofs[0] if proofs else None


_ANAPHOR_TYPED_ID = re.compile(
    r"\b(agreement|contract|receipt|invoice|work\s*order|workorder|"
    r"change\s*request|changerequest|service\s*order|serviceorder|"
    r"sensor|log|record)\s*(?:id\s*)?"
    r"([A-Z]{1,8}(?:[-/][A-Z0-9]+)+)\b",
    re.IGNORECASE,
)


def _anaphoric_record_proof(
    question: str, evidence: str, context: str, evidence_start: int
) -> str | None:
    """Resolve only a uniquely typed, immediately preceding record reference."""
    target = evidence.strip()
    definite = re.match(
        r"(?i)^the\s+(agreement|contract|receipt|invoice|work\s*order|"
        r"change\s*request|service\s*order|sensor|log|record)\b",
        target,
    )
    pronoun = re.match(r"(?i)^it\b", target)
    if not definite and not pronoun:
        return None

    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in _segments(context):
        position = context.find(sentence, cursor)
        if position < 0:
            return None
        spans.append((position, position + len(sentence), sentence))
        cursor = position + len(sentence)
    target_index = next(
        (
            index
            for index, (start, end, _) in enumerate(spans)
            if start <= evidence_start < end
        ),
        None,
    )
    if target_index is None or target_index == 0:
        return None
    _, previous_end, previous = spans[target_index - 1]
    if context[previous_end:evidence_start].strip():
        return None
    if _UNVERIFIED_SOURCE_GAP in previous:
        return None

    requested_ids = _record_constraint_values(question)
    anchors = [
        (re.sub(r"\s+", "", _fold(match.group(1))), _fold(match.group(2)))
        for match in _ANAPHOR_TYPED_ID.finditer(previous)
    ]
    if not requested_ids or not anchors:
        return None
    canonical = {
        "agreement": "agreement",
        "contract": "agreement",
        "receipt": "receipt",
        "invoice": "invoice",
        "workorder": "workorder",
        "changerequest": "changerequest",
        "serviceorder": "serviceorder",
        "sensor": "sensor",
        "log": "log",
        "record": "record",
    }
    if pronoun is not None:
        compatible = anchors if len(anchors) == 1 else []
    else:
        assert definite is not None
        wanted_type = re.sub(r"\s+", "", _fold(definite.group(1)))
        wanted_type = canonical.get(wanted_type, wanted_type)
        compatible = [
            (kind, identifier)
            for kind, identifier in anchors
            if canonical.get(kind, kind) == wanted_type
        ]
        if len(compatible) != 1:
            return None
    if len(compatible) != 1 or compatible[0][1] not in requested_ids:
        return None
    return previous


def _context_id_proofs(
    question: str, evidence: str, constraint_context: str | None
) -> tuple[str, ...] | None:
    """Return the minimal preceding record assertions that bind requested IDs.

    A unique record scope may carry an ID to a later field, but identifiers in
    separate, unrelated assertions are not joined by mere co-occurrence.
    """
    requested = _record_constraint_values(question)
    if not requested or requested <= {
        _fold(match.group(1)) for match in _EXPLICIT_ID.finditer(evidence)
    }:
        return ()
    if not constraint_context or constraint_context.count(evidence) != 1:
        return None
    evidence_start = constraint_context.find(evidence)
    sentence_spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in _segments(constraint_context):
        position = constraint_context.find(sentence, cursor)
        if position < 0:
            return None
        sentence_spans.append((position, position + len(sentence), sentence))
        cursor = position + len(sentence)
    preceding = [row for row in sentence_spans if row[1] <= evidence_start]
    containing = next((row for row in sentence_spans if row[0] <= evidence_start < row[1]), None)
    if containing is not None:
        relative_colon = containing[2].find(":")
        if relative_colon >= 0 and containing[0] + relative_colon < evidence_start:
            prefix = constraint_context[containing[0] : containing[0] + relative_colon].strip()
            if _EXPLICIT_ID.search(prefix):
                preceding.append((containing[0], containing[0] + relative_colon, prefix))
                preceding.sort(key=lambda row: row[0])
    if containing is not None:
        anaphoric_proof = _anaphoric_record_proof(
            question, evidence, constraint_context, evidence_start
        )
        if anaphoric_proof is not None:
            return (anaphoric_proof,)
        if re.match(
            r"(?i)^\s*(?:it\b|the\s+(?:agreement|contract|receipt|invoice|work\s*order|"
            r"change\s*request|service\s*order|sensor|log|record)\b)",
            evidence,
        ):
            return None
    anchor_indices = [
        index
        for index, (_, _, sentence) in enumerate(preceding)
        if requested & {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(sentence)}
    ]
    if not anchor_indices:
        return None
    for anchor_index in reversed(anchor_indices):
        anchor_start = preceding[anchor_index][0]
        scope = constraint_context[anchor_start : evidence_start + len(evidence)]
        scope_ids = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(scope)}
        if not requested <= scope_ids:
            continue
        if any(
            {
                value
                for value in scope_ids
                if _identifier_prefix(value) == _identifier_prefix(requested_id)
            }
            != {requested_id}
            for requested_id in requested
        ):
            continue

        context_rows = [row for row in preceding[anchor_index:]]
        local_ids = {
            _fold(match.group(1)) for match in _EXPLICIT_ID.finditer(evidence)
        } - (_requested_identifier_values(question) - requested)
        nodes = [
            (sentence, clause)
            for _, _, sentence in context_rows
            for part in re.split(r"[;]+", sentence)
            for clause in _assertion_clauses(part)
        ]
        anchor_start = preceding[anchor_index][0]
        root_node = next(
            (
                index
                for index, (_, clause) in enumerate(nodes)
                if requested & {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(clause)}
                and constraint_context.find(clause, anchor_start) < evidence_start
            ),
            None,
        )
        if root_node is None:
            continue
        connected = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(nodes[root_node][1])}
        proofs = [nodes[root_node][0]]
        for sentence, clause in nodes[root_node + 1 :]:
            clause_ids = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(clause)}
            if clause_ids & connected:
                added = clause_ids - connected
                connected.update(clause_ids)
                if added & requested:
                    proofs.append(sentence)
        if local_ids & connected:
            connected.update(local_ids)
        if requested <= connected and local_ids <= connected:
            normalized_proofs = []
            for proof in proofs:
                candidate = proof.strip()
                if re.fullmatch(
                    r"[A-Za-z][A-Za-z0-9]*ID\s+[A-Z0-9][A-Z0-9/-]*\.?",
                    candidate,
                ):
                    candidate = candidate.removesuffix(".")
                normalized_proofs.append(candidate)
            return tuple(dict.fromkeys(normalized_proofs))
    return None


def _question_excluded_terms(question: str) -> set[str]:
    folded = _fold(question)
    excluded: set[str] = set()
    for match in re.finditer(r"\b(?:except|excluding|other than|hors|sauf)\b", folded):
        exclusion_text = re.split(
            r",|\b(?:under|for|with|according\s+to)\b",
            folded[match.end() :],
            maxsplit=1,
        )[0]
        excluded.update(_relation_tokens(exclusion_text) - _QUESTION_NOISE)
    if re.search(r"\b(?:hors\s+(?:la\s+)?tva|excluding\s+vat)\b", folded):
        excluded -= {"vat", "tva"}
    return excluded


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
            "amount",
            "cost",
            "price",
            "prix",
            "how",
            "much",
            "apply",
            "charg",
            "constitue",
            "combien",
            "coute",
            "cout",
            "par",
            "hors",
            "sauf",
            "record",
            "procure",
            "state",
            "report",
            "file",
            "invoice",
            "invoic",
            "before",
            "pre",
            "tax",
            "vat",
            "tva",
        },
        "duration": {"duration", "deadline", "period", "long", "soon", "quickly", "many"},
        "count": {"how", "many", "number", "combien"},
        "status": {"status", "statut", "recorded", "record", "consigne"},
        "severity": {"severity"},
        "responsible_people": {"owner"},
        "organization_name": set(),
        "boolean": {
            "is",
            "are",
            "was",
            "were",
            "does",
            "do",
            "did",
            "can",
            "could",
            "may",
            "might",
            "will",
            "would",
            "has",
            "have",
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


def _explicit_labeled_field_value(question: str, segment: str, value: str) -> bool:
    parts = _requested_field_head(question)
    if not parts:
        return False
    label = r"[\s_-]*".join(re.escape(part) for part in parts)
    pattern = (
        r"(?<![A-Za-z0-9])"
        + label
        + r"(?:[\s_-]*id)?(?![A-Za-z0-9])\s*[:=]\s*"
        + re.escape(value)
        + r"(?=$|[\s,.;])"
    )
    return re.search(pattern, segment, re.IGNORECASE) is not None


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
        r"\b(?:how\s+many\s+(?!minutes?\b|hours?\b|days?\b|weeks?\b|months?\b|years?\b)|"
        r"what\s+number\s+of|combien\s+de\s+(?!temps\b))",
        folded,
    ):
        return "count"
    if re.search(
        r"\b(?:when|quand|date|milestone|effect|applicable|commence|start|ouvre)\b",
        folded,
    ):
        return "date"
    if re.search(
        r"\b(?:how much|amount|budget|cost|charge|fee|tariff|price|prix|montant|"
        r"total\s+(?:value|amount|cost)|pre[- ]?tax\s+total|invoice\s+total|"
        r"total\s+of\s+(?:the\s+)?invoice|"
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
    field_parts = _requested_field_parts(question)
    if field_parts and field_parts[-1] in {
        "code",
        "tag",
        "ref",
        "reference",
        "number",
        "id",
    }:
        return "field_value"
    if (
        re.search(
            r"\b(?:organization|organisation|entity|entite|supplier|custodian|mutuelle|"
            r"beneficiary|beneficiaire|account holder|operator|issued)\b",
            folded,
        )
        or "whose account" in folded
    ):
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
        for delimiter in (";", "|", "\n", *((",",) if len(matches) > 1 else ()))
    ]
    end_candidates = [
        position
        for delimiter in (";", "|", "\n", *((",",) if len(matches) > 1 else ()))
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
        if extracted is not None:
            if not _directed_field_relation(question, segment, extracted[0]) and not (
                _explicit_labeled_field_value(question, segment, extracted[0])
            ):
                return None
            return extracted[0]
        # Some factual questions have a clear requested value in prose, but
        # no locally labeled field for the structured extractor. Preserve the
        # complete assertion only when the record constraints and ordinary
        # relation checks both match; this avoids guessing a scalar from a
        # neighboring field or another record.
        if _prose_field_fallback_supported(question, segment, constraint_context):
            return segment
        return None
    if kind == "time":
        if not _constraints_present(question, segment, constraint_context):
            return None
        selected = _time_value_context(
            question, segment, constraint_context=constraint_context
        )
        return selected[0] if selected else None
    if kind == "count":
        if not _constraints_present(question, segment, constraint_context):
            return None
        object_match = re.search(
            r"\b(?:how\s+many|what\s+number\s+of|combien\s+de)\s+(.+?)"
            r"(?=\s+(?:did|do|does|has|have|had|is|are|was|were|will|can|could|"
            r"arrived|arrive|remain|remained|include|included|cover|covered|"
            r"sont|est|ont|a|ont|arrive|arrives|restent|reste|contient|contiennent)\b|[?]|$)",
            _fold(question),
        )
        if object_match is None:
            return None
        object_terms = _relation_tokens(object_match.group(1)) - {
            "of",
            "de",
            "the",
            "a",
            "an",
            "le",
            "la",
            "les",
            "des",
            "du",
        }
        if not object_terms:
            return None
        relation_terms = _question_relation_tokens(question, "count") - object_terms
        relation_terms -= _relation_tokens("how many what number did do does has have had "
                                           "is are was were will can could")
        relation_terms -= _relation_tokens("record workorder agreement contract")
        for requested_id in _requested_identifier_values(question):
            relation_terms -= _relation_tokens(requested_id)
        number_pattern = re.compile(
            r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|dozen|un|une|deux|trois|quatre|cinq|"
            r"six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|"
            r"seize|dix-sept|dix-huit|dix-neuf|vingt|\d+)\b",
            re.IGNORECASE,
        )
        identifiers = [match.span() for match in _EXPLICIT_ID.finditer(segment)]
        date_spans = [match.span() for match in _DATE.finditer(segment)]
        clock_spans = [match.span() for match in _CLOCK.finditer(segment)]
        candidates: list[tuple[str, bool]] = []
        count_relation_groups = (
            {"arriv", "deliver", "deliveri", "delivery", "receiv", "receipt"},
            {"plan", "schedul", "expect"},
        )

        def count_relation_matches(local_clause: str) -> bool:
            local_relation = _relation_tokens(local_clause)
            return any(
                bool(relation_terms & group) and bool(local_relation & group)
                for group in count_relation_groups
            )

        for number in number_pattern.finditer(segment):
            excluded_spans = identifiers + date_spans + clock_spans
            if any(start <= number.start() < end for start, end in excluded_spans):
                continue
            left = max(
                (
                    segment.rfind(mark, 0, number.start()) + 1
                    for mark in (";", "|", ",", " and ", " et ")
                ),
                default=0,
            )
            right_candidates = [
                position
                for mark in (";", "|", ",", " and ", " et ", ".", "!", "?", "\n")
                if (position := segment.find(mark, number.end())) >= 0
            ]
            right = min(right_candidates) if right_candidates else len(segment)
            local_terms = _relation_tokens(segment[left:right])
            if object_terms <= local_terms:
                relation_clause = _local_value_clause(segment, number.start(), number.end())
                relation_clause_terms = _relation_tokens(relation_clause)
                value_prefix = re.search(
                    r"(?:at\s+least|no\s+fewer\s+than|minimum\s+of|at\s+most|"
                    r"no\s+more\s+than|up\s+to|about|approximately|around|"
                    r"au\s+moins|au\s+plus|environ)\s+$",
                    segment[max(0, number.start() - 32) : number.start()],
                    re.IGNORECASE,
                )
                value = (
                    f"{value_prefix.group(0).strip()} {number.group(0)}"
                    if value_prefix
                    else number.group(0)
                )
                candidates.append(
                    (
                        value,
                        _has_relation_token_overlap(relation_terms, relation_clause_terms)
                        or count_relation_matches(relation_clause),
                    )
                )
        if relation_terms:
            candidates = [candidate for candidate in candidates if candidate[1]]
        if len(candidates) != 1:
            return None
        return candidates[0][0]
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
                requested_values = {_fold(match.group(0)) for match in requested_matches}
                observed_values = {_fold(match.group(0)) for match in observed_matches}
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
        requested_role = _amount_role(question)
        query_terms = _question_relation_tokens(question, kind)
        excluded = _question_excluded_terms(question)
        scored = []
        for match in matches:
            context = _local_value_clause(segment, match.start(), match.end())
            if requested_role is not None and _amount_role(context) != requested_role:
                continue
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
    unit = re.search(
        r"\d+(?:[.,]\d+)?\s+([A-Za-zµμ°%][A-Za-z0-9µμ°/%]*)",
        unknown_measurement.group(0),
    )
    if unit is not None:
        unit_text = unit.group(1)
        if (unit_text[:1].isupper() and not unit_text.isupper()) or unit_text.casefold() in {
            "a",
            "an",
            "and",
            "as",
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "year",
            "years",
            "hour",
            "hours",
            "for",
            "of",
            "on",
            "in",
            "to",
            "per",
            "is",
            "was",
            "were",
        }:
            return False
    measurement_context = segment[: unknown_measurement.start()]
    return bool(_question_relation_tokens(question, "text") & _relation_tokens(measurement_context))


def _normalize_clock(hour: str, suffix: str | None) -> str:
    return f"{hour} {suffix}".casefold().strip() if suffix else hour.casefold().strip()


def _normalize_clock_text(value: str) -> str:
    match = _CLOCK.search(value)
    return _normalize_clock(match.group(1), match.group(2)) if match else value.casefold().strip()


def _confirmed_cross_sentence_time(
    question: str, context: str
) -> tuple[str, str, str] | None:
    """Resolve a prior reading only when an adjacent signed correction names it."""
    requested_ids = _record_constraint_values(question)
    event_terms = _question_relation_tokens(question, "time") - {
        "time", "record", "recorded", "list", "listed", "indicate", "indicated",
        "show", "shown", "date", "morning", "start", "end",
    }
    segments = _segments(context)
    for index, correction in enumerate(segments):
        folded = _fold(correction)
        correction_clocks = list(_CLOCK.finditer(correction))
        if (
            len(correction_clocks) != 2
            or not re.search(
                r"\b(?:signed\s+correction|corrected\s+and\s+confirmed|"
                r"confirmed\s+correction)\b.{0,100}\b(?:confirms?|corrects?|"
                r"same\s+(?:morning\s+)?reading)\b",
                folded,
            )
            or not re.search(r"\b(?:same\s+.+\s+as|rather\s+than|instead\s+of)\b", folded)
            or re.search(
                r"\b(?:not\s+confirmed|unconfirmed|no\s+signed\s+correction|"
                r"without\s+(?:a\s+)?signed\s+correction|pending\s+confirmation)\b",
                folded,
            )
            or "[unverified source gap]" in folded
            or index == 0
        ):
            continue
        anchor = segments[index - 1]
        if "[unverified source gap]" in _fold(anchor):
            continue
        anchor_ids = {_fold(match.group(1)) for match in _EXPLICIT_ID.finditer(anchor)}
        if requested_ids and not requested_ids <= anchor_ids:
            continue
        anchor_clocks = list(_CLOCK.finditer(anchor))
        if len(anchor_clocks) != 1:
            continue
        anchor_terms = _relation_tokens(anchor)
        correction_terms = _relation_tokens(correction)
        if event_terms and not (event_terms & anchor_terms and event_terms & correction_terms):
            continue
        if requested_ids and not _constraints_present(question, anchor, context):
            continue
        corrected, replaced = correction_clocks
        corrected_value = (
            f"{corrected.group(1)} {corrected.group(2)}"
            if corrected.group(2)
            else corrected.group(1)
        )
        replaced_value = (
            f"{replaced.group(1)} {replaced.group(2)}"
            if replaced.group(2)
            else replaced.group(1)
        )
        if _normalize_clock(anchor_clocks[0].group(1), anchor_clocks[0].group(2)) != (
            _normalize_clock(replaced.group(1), replaced.group(2))
        ):
            continue
        return corrected_value, correction, replaced_value
    return None


def _time_value_context(
    question: str, segment: str, *, constraint_context: str | None = None
) -> tuple[str, str] | None:
    requested = _requested_clock_values(question)
    candidates: list[tuple[int, int, str, str]] = []
    query_terms = _question_relation_tokens(question, "time")
    event_terms = query_terms - {
        "time",
        "record",
        "recorded",
        "list",
        "listed",
        "enter",
        "entered",
        "indicate",
        "indicated",
        "show",
        "shown",
        "date",
        "start",
        "end",
    }

    def event_is_supported(clause: str) -> bool:
        requested_subjects = _requested_person_terms(question)
        if requested_subjects and not requested_subjects <= _relation_tokens(clause):
            return False
        if event_terms and not event_terms & _relation_tokens(clause):
            return False
        if re.search(
            r"\b(?:complete\w*|finish\w*|termine\w*|acheve\w*)\b",
            _fold(question),
        ):
            completed_action = re.search(
                r"\b(?:complete\w*|finish\w*|termine\w*|acheve\w*|"
                r"replac(?:e|ed|es|ing)|remplac(?:e|er|ee|ees|ent|ent)|"
                r"was\s+replac(?:e|ed)|a\s+ete\s+remplacee?)\b",
                _fold(clause),
            )
            if completed_action is None:
                return False
        return True

    for match in _CLOCK.finditer(segment):
        value = (
            f"{match.group(1)} {match.group(2)}"
            if match.group(2)
            else match.group(1)
        )
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
    resolved_correction = _confirmed_cross_sentence_time(
        question, constraint_context or segment
    )
    if resolved_correction is not None:
        corrected_value, correction_excerpt, replaced_value = resolved_correction
        segment_clocks = {
            _normalize_clock(match.group(1), match.group(2))
            for match in _CLOCK.finditer(segment)
        }
        if (
            correction_excerpt not in segment
            and _normalize_clock_text(replaced_value) in segment_clocks
        ):
            return None
        if correction_excerpt in segment:
            correction_values = _explicit_time_alternatives(
                question, segment, constraint_context=constraint_context
            )
            if correction_values:
                return corrected_value, correction_values[0][1]
    if not candidates:
        return None
    if requested:
        if len(candidates) != 1 or not event_is_supported(candidates[0][3]):
            return None
        return candidates[0][2], candidates[0][3]
    alternatives = _explicit_time_alternatives(
        question, segment, constraint_context=constraint_context
    )
    if alternatives:
        return alternatives[0]
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


def _explicit_time_alternatives(
    question: str, segment: str, *, constraint_context: str | None = None
) -> list[tuple[str, str]]:
    """Return clock values explicitly contrasted for the same requested event."""
    if _requested_clock_values(question):
        return []
    clocks = list(_CLOCK.finditer(segment))
    if len(clocks) < 2:
        return []
    query_terms = _question_relation_tokens(question, "time")
    event_terms = query_terms - {
        "time", "record", "recorded", "list", "listed", "indicate", "indicated",
        "show", "shown", "date", "morning", "start", "end",
    }
    named_subjects = _requested_person_terms(question)
    for left, right in pairwise(clocks):
        connector = _fold(segment[left.end() : right.start()])
        if not re.search(r"\b(?:rather\s+than|instead\s+of|not\s+.*\bbut)\b", connector):
            continue
        left_clause = _local_value_clause(segment, left.start(), left.end())
        right_clause = _local_value_clause(segment, right.start(), right.end())
        if left_clause != right_clause:
            continue
        clause = left_clause
        resolved_prefix = _fold(segment[: left.start()])
        if re.search(
            r"\b(?:corrected|confirmed|confirms?|approved|validated|signed\s+(?:off|correction)|"
            r"rectifie|confirme|approuve|valide)\b",
            resolved_prefix,
        ) and not re.search(
            r"\b(?:not\s+confirmed|unconfirmed|not\s+approved|not\s+corrected|"
            r"no\s+signed\s+correction|without\s+(?:a\s+)?signed\s+correction|"
            r"pending\s+confirmation|"
            r"non\s+confirme|pas\s+confirme)\b",
            resolved_prefix,
        ):
            value = f"{left.group(1)} {left.group(2)}" if left.group(2) else left.group(1)
            return [(value, clause)]
        clause_terms = _relation_tokens(clause)
        if event_terms and not event_terms & clause_terms:
            continue
        if named_subjects and not named_subjects <= clause_terms:
            continue
        if not _constraints_present(question, clause, constraint_context or segment):
            continue
        values = []
        for match in (left, right):
            value = (
                f"{match.group(1)} {match.group(2)}" if match.group(2) else match.group(1)
            )
            values.append((value, clause))
        return values
    return []


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
        return (
            _value_for(kind, segment, question, constraint_context=constraint_context) is not None
        )
    if kind == "amount":
        requested_role = _amount_role(question)
        if requested_role is not None:
            value = _value_for(kind, segment, question, constraint_context=constraint_context)
            return value is not None
    noise_by_kind = {
        "date": {"date", "effective", "deadline"},
        "amount": {
            "amount",
            "cost",
            "price",
            "prix",
            "how",
            "much",
            "apply",
            "charg",
            "constitue",
            "combien",
            "coute",
            "cout",
            "par",
            "hors",
            "sauf",
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
        value = _value_for(kind, segment, question, constraint_context=constraint_context)
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
    if kind == "count":
        value = _value_for(kind, segment, question, constraint_context=constraint_context)
        if value is None:
            return False
        return _has_relation_token_overlap(
            _question_relation_tokens(question, kind), _relation_tokens(segment)
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
        return (
            bool(overlap)
            and _value_for(kind, segment, question, constraint_context=constraint_context)
            is not None
        )
    if kind == "multi_slot":
        return False
    if kind == "text":
        return _text_relation_supported(question, segment)
    if kind == "duration" and re.search(r"\b(?:duration|duree)\b", _fold(segment)):
        return True
    if kind == "severity" and _SEVERITY.search(segment):
        return True
    return bool(overlap)


def _text_relation_supported(question: str, segment: str) -> bool:
    """Require lexical support for a complete prose assertion."""
    question_tokens = _question_relation_tokens(question, "text")
    segment_tokens = _relation_tokens(segment)
    requested_head = _requested_field_head(question)
    head_terms = [word for word in requested_head if word not in _QUESTION_NOISE]
    # This fallback returns prose, so keep the slot and its value inside one
    # assertion. Treat punctuation and conjunctions as hard boundaries; a
    # conservative abstention is safer than joining a neighboring field.
    clauses = re.split(
        r";|\r?\n|(?<=[.!?])\s+|\b(?:and|while|whereas|but|however|"
        r"alors\s+que|tandis\s+que)\b",
        segment,
        flags=re.IGNORECASE,
    )
    slot_present = any(
        head_terms
        and all(
            re.search(rf"(?<![a-z0-9]){re.escape(term)}s?(?![a-z0-9])", _fold(clause))
            for term in head_terms
        )
        for clause in clauses
    )
    return (
        len(question_tokens & segment_tokens) >= 2
        and slot_present
        and bool(_requested_field_words(question) & segment_tokens)
    )


def _prose_field_fallback_supported(
    question: str, segment: str, constraint_context: str | None
) -> bool:
    if (
        not _constraints_present(question, segment, constraint_context)
        or _unknown_measurement_for_relation(question, segment)
        or not _text_relation_supported(question, segment)
    ):
        return False
    conditions = _requested_conditions(question)
    if not conditions:
        return True
    if len(list(_MEASUREMENT.finditer(segment))) > 1:
        return False
    field_head = _requested_field_words(question)
    head_matches = [
        match
        for token in field_head
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", _fold(segment))
    ]
    if not head_matches:
        return False
    start, end = _assertion_bounds(segment, head_matches[-1].start(), head_matches[-1].end())
    local_clause = _fold(segment[start:end])
    return all(condition in local_clause for condition in conditions)


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
    grouped: dict[tuple[str, int], list[RankedChunk]] = {}
    for item in ranked:
        grouped.setdefault((item.chunk.document_id, item.chunk.page), []).append(item)
    contexts: dict[str, str] = {}
    for items in grouped.values():
        ordered, context = _ordered_context(items)
        for item in ordered:
            contexts[item.chunk.id] = context
    return contexts


def _document_contexts(ranked: list[RankedChunk]) -> dict[str, str]:
    """Build a same-document context while preserving source page/chunk order."""
    grouped: dict[str, list[RankedChunk]] = {}
    for item in ranked:
        grouped.setdefault(item.chunk.document_id, []).append(item)
    contexts: dict[str, str] = {}
    for items in grouped.values():
        ordered, context = _ordered_context(items)
        for item in ordered:
            contexts[item.chunk.id] = context
    return contexts


def _ordered_context(items: list[RankedChunk]) -> tuple[list[RankedChunk], str]:
    """Join retrieved chunks in source order and mark every unverifiable gap."""
    def position(item: RankedChunk) -> tuple[int, int, int] | None:
        if item.chunk.ordinal is not None:
            return (0, item.chunk.page, item.chunk.ordinal)
        match = re.search(r"(?:^|:)c(\d+)$", item.chunk.id)
        if match:
            return (1, item.chunk.page, int(match.group(1)))
        return None

    positioned = [(position(item), item) for item in items]
    ordered = [item for _, item in sorted(
        positioned,
        key=lambda row: row[0] if row[0] is not None else (2, 0, 0),
    )]
    pieces: list[str] = []
    prior: RankedChunk | None = None
    for item in ordered:
        if prior is not None:
            left, right = position(prior), position(item)
            contiguous = False
            if left is not None and right is not None and left[0] == right[0]:
                if left[0] == 0:
                    contiguous = right[2] == left[2] + 1
                elif prior.chunk.page == item.chunk.page:
                    contiguous = right[2] == left[2] + 1
            if not contiguous:
                pieces.append(_UNVERIFIED_SOURCE_GAP)
        pieces.append(item.chunk.text)
        prior = item
    return ordered, "\n".join(pieces)


def _evidence_citations(
    item: RankedChunk,
    excerpt: str,
    context_headers: dict[str, tuple[tuple[RankedChunk, str], ...]],
) -> tuple[Citation, ...]:
    headers = context_headers.get(item.chunk.id, ())
    if not headers:
        return (_citation(item, excerpt),)
    citations = [_citation(header_item, header_excerpt) for header_item, header_excerpt in headers]
    citations.append(_citation(item, excerpt))
    return tuple(dict.fromkeys(citations))


def _context_citations(
    question: str,
    evidence: str,
    context: str,
    item: RankedChunk,
    ranked: list[RankedChunk],
) -> tuple[tuple[RankedChunk, str], ...]:
    proofs = _context_id_proofs(question, evidence, context)
    citations: list[tuple[RankedChunk, str]] = []
    for proof in proofs or ():
        proof_item = next(
            (
                candidate
                for candidate in ranked
                if candidate.chunk.document_id == item.chunk.document_id
                and proof in candidate.chunk.text
            ),
            None,
        )
        if proof_item is None:
            return ()
        citations.append((proof_item, proof))
    requested_source_ids = _requested_source_ids(question)
    evidence_ids = {
        _fold(match.group(1)) for match in _EXPLICIT_ID.finditer(evidence)
    }
    missing_source_ids = requested_source_ids - evidence_ids
    if missing_source_ids:
        source_proof: tuple[RankedChunk, str] | None = None
        for line in context.splitlines():
            primary_identity = _principal_identity(line)
            primary_ids = {
                _fold(match.group(1))
                for match in _EXPLICIT_ID.finditer(primary_identity)
            }
            if not missing_source_ids <= primary_ids:
                continue
            source_item = next(
                (
                    candidate
                    for candidate in ranked
                    if candidate.chunk.document_id == item.chunk.document_id
                    and line in candidate.chunk.text
                ),
                None,
            )
            if source_item is not None:
                source_proof = (source_item, line)
                break
        if source_proof is not None:
            citations.append(source_proof)
    unique: list[tuple[RankedChunk, str]] = []
    seen: set[tuple[str, str]] = set()
    for citation_item, proof in citations:
        key = (citation_item.chunk.id, proof)
        if key not in seen:
            seen.add(key)
            unique.append((citation_item, proof))
    return tuple(unique)


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
        "count": "count",
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
        if chunk.document_id == answer.supporting_document and chunk.page == answer.supporting_page
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
    if answer.status == "answered" and _fold(answer.answer) not in _fold(answer.supporting_excerpt):
        boolean_supported = (
            answer.answer.startswith("Yes. ") and answer.answer[5:] == answer.supporting_excerpt
        ) or (answer.answer.startswith("No. ") and answer.answer[4:] == answer.supporting_excerpt)
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
                row for row in rows if row[0] >= self.config.partial_support_threshold and row[3]
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
        provisional_amounts: list[
            tuple[float, str, str, RankedChunk, str, RankedChunk | None]
        ] = []
        context_headers: dict[str, tuple[tuple[RankedChunk, str], ...]] = {}
        field_absences: list[tuple[str, RankedChunk, float]] = []
        document_contexts = _document_contexts(ranked)
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
                        document_contexts.get(item.chunk.id, parent_content.text)
                        if kind in {"field_value", "date", "time", "amount", "duration", "count"}
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
                    if kind == "amount" and not _amount_intent_supported(
                        question, evidence_text
                    ):
                        evaluated.setdefault(item.chunk.id, []).append(
                            (0.0, evidence_text, None, False)
                        )
                        continue
                    if kind == "field_value" and _has_requested_field_absence(
                        question, evidence_text
                    ):
                        if _constraints_present(question, evidence_text, constraint_context):
                            field_absences.append((evidence_text, item, 0.5))
                            proofs = _context_citations(
                                question, evidence_text, constraint_context, item, ranked
                            )
                            if proofs:
                                context_headers[item.chunk.id] = proofs
                        evaluated.setdefault(item.chunk.id, []).append(
                            (0.0, evidence_text, None, False)
                        )
                        continue
                    explicit_time_conflict = (
                        kind == "time"
                        and len(
                            _explicit_time_alternatives(
                                question,
                                evidence_text,
                                constraint_context=constraint_context,
                            )
                        )
                        > 1
                    )
                    if (
                        kind not in {"boolean", "status"}
                        and _has_negated_relation(question, evidence_text, kind)
                        and not explicit_time_conflict
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
                        extracted = _field_value(question, evidence_text, constraint_context)
                        if extracted is not None:
                            citation_text = extracted[1]
                            proofs = _context_citations(
                                question, citation_text, constraint_context, item, ranked
                            )
                            if proofs:
                                context_headers[item.chunk.id] = proofs
                        elif value is not None:
                            proofs = _context_citations(
                                question, evidence_text, constraint_context, item, ranked
                            )
                            if proofs:
                                context_headers[item.chunk.id] = proofs
                    elif kind in {"time", "date", "amount", "duration", "count"}:
                        if kind == "time":
                            time_context = _time_value_context(question, evidence_text)
                            if time_context is not None:
                                citation_text = time_context[1]
                        else:
                            date_match = _date_match_for_question(evidence_text, question)
                            if date_match is not None:
                                citation_text = _date_context(evidence_text, date_match).strip()
                        proofs = _context_citations(
                            question, citation_text, constraint_context, item, ranked
                        )
                        if proofs:
                            context_headers[item.chunk.id] = proofs
                    evaluated.setdefault(item.chunk.id, []).append(
                        (score, citation_text, value, relation)
                    )
                    if value is not None and relation:
                        if kind == "time":
                            alternatives = _explicit_time_alternatives(
                                question,
                                evidence_text,
                                constraint_context=constraint_context,
                            )
                            if len(alternatives) > 1:
                                candidates.extend(
                                    (score, alternative_excerpt, alternative_value, item)
                                    for alternative_value, alternative_excerpt in alternatives
                                )
                                continue
                        provisional_status = (
                            _provisional_amount_status(evidence_text, constraint_context)
                            if kind == "amount"
                            else None
                        )
                        if provisional_status is not None:
                            status_item = next(
                                (
                                    candidate
                                    for candidate in ranked
                                    if candidate.chunk.document_id == item.chunk.document_id
                                    and provisional_status in candidate.chunk.text
                                ),
                                None,
                            )
                            provisional_amounts.append(
                                (
                                    score,
                                    citation_text,
                                    value,
                                    item,
                                    provisional_status,
                                    status_item,
                                )
                            )
                            continue
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
                        "The same requested field is both stated and explicitly marked absent."
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
        if provisional_amounts:
            provisional_amounts.sort(key=lambda row: (-row[0], row[3].chunk.id))
            score, excerpt, _, item, status_excerpt, status_item = provisional_amounts[0]
            provisional_citations: list[Citation] = list(
                _evidence_citations(item, excerpt, context_headers)
            )
            if status_item is not None:
                provisional_citations.append(_citation(status_item, status_excerpt))
            competing = next(
                (candidate for candidate in supported if candidate[2] != provisional_amounts[0][2]),
                None,
            )
            if competing is not None:
                provisional_citations.extend(
                    _evidence_citations(competing[3], competing[1], context_headers)
                )
            else:
                related_amount = _related_amount_proof(
                    question,
                    provisional_amounts[0][2],
                    item,
                    ranked,
                    document_contexts,
                )
                if related_amount is not None:
                    related_item, related_excerpt = related_amount
                    related_context = document_contexts.get(
                        related_item.chunk.id, related_item.chunk.text
                    )
                    provisional_citations.extend(
                        _evidence_citations(
                            related_item,
                            related_excerpt,
                            {
                                related_item.chunk.id: _context_citations(
                                    question,
                                    related_excerpt,
                                    related_context,
                                    related_item,
                                    ranked,
                                )
                            },
                        )
                    )
            unique_citations: list[Citation] = []
            seen_citations: set[tuple[str, str]] = set()
            for citation in provisional_citations:
                key = (citation.chunk_id, citation.excerpt)
                if key not in seen_citations:
                    seen_citations.add(key)
                    unique_citations.append(citation)
            result = GroundedAnswer(
                status="ambiguous",
                answerable=False,
                answer="The cited amount is marked provisional or awaiting confirmation.",
                confidence=score,
                supporting_document=item.chunk.document_id,
                supporting_page=item.chunk.page,
                supporting_excerpt=excerpt,
                ambiguity_reason=(
                    "The source explicitly leaves this amount unresolved, "
                    "so it cannot be returned as final."
                ),
                extracted_fields={},
                supporting_chunk_ids=tuple(
                    dict.fromkeys(citation.chunk_id for citation in unique_citations)
                ),
                citations=tuple(unique_citations),
                candidate_assessments=assessments,
            )
            validate_grounded_answer(result, ranked)
            return result
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
            "count",
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
            extracted_fields=({field_name: best[2] or best[1]} if kind != "boolean" else {}),
            supporting_chunk_ids=tuple(
                dict.fromkeys(
                    citation.chunk_id
                    for citation in _evidence_citations(best[3], best[1], context_headers)
                )
            ),
            citations=_evidence_citations(best[3], best[1], context_headers),
            candidate_assessments=assessments,
        )
        validate_grounded_answer(result, ranked)
        return result
