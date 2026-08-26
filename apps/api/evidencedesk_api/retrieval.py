import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from evidencedesk_api.providers import EmbeddingProvider

DEFAULT_RETRIEVAL_LIMIT = 20
RRF_K = 60
NAMED_ENTITY_DOCUMENT_BOOST = 0.01

_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "de",
    "d",
    "des",
    "does",
    "est",
    "how",
    "combien",
    "ete",
    "il",
    "is",
    "la",
    "le",
    "les",
    "quel",
    "quelle",
    "quelles",
    "qui",
    "l",
    "requi",
    "requis",
    "of",
    "on",
    "the",
    "to",
    "t",
    "what",
    "when",
    "who",
}


_CANONICAL: dict[str, str] = {
    "accord": "agreement",
    "agree": "agreement",
    "contrat": "agreement",
    "annuel": "annual",
    "annuelle": "annual",
    "approbation": "approval",
    "autorisation": "approval",
    "cout": "amount",
    "cost": "amount",
    "fee": "amount",
    "montant": "amount",
    "effect": "effective",
    "effet": "effective",
    "heures": "deadline",
    "hours": "deadline",
    "hour": "deadline",
    "delai": "deadline",
    "duree": "duration",
    "maintient": "maintain",
    "flux": "feed",
    "mensuel": "monthly",
    "mensuelle": "monthly",
    "notify": "notification",
    "notification": "notification",
    "owner": "owner",
    "possede": "owner",
    "proprietaire": "owner",
    "preavis": "notice",
    "preavi": "notice",
    "renouvellement": "renewal",
    "renouvelle": "renewal",
    "renew": "renewal",
    "retarde": "delayed",
    "delay": "delayed",
    "retard": "delayed",
    "retention": "retention",
    "risque": "risk",
    "temps": "duration",
    "temp": "duration",
    "minutes": "duration",
    "minut": "duration",
    "sev": "severity",
    "severite": "severity",
    "sous": "subcontractor",
    "subcontractors": "subcontractor",
    "traitants": "subcontractor",
    "tva": "vat",
    "adresse": "address",
    "disponibilite": "availability",
    "penalite": "penalty",
    "paiement": "payment",
    "tribunal": "court",
    "retablissement": "restoration",
    "budget": "budget",
    "bitcoin": "bitcoin",
}


_REQUIRED_EVIDENCE_TOKENS = {
    "address",
    "availability",
    "bitcoin",
    "budget",
    "court",
    "exchange",
    "penalty",
    "restoration",
    "vat",
}


_FIELD_TOKENS = {
    "amount",
    "annual",
    "approval",
    "deadline",
    "delayed",
    "duration",
    "effective",
    "maintain",
    "monthly",
    "notice",
    "notification",
    "owner",
    "renewal",
    "retention",
    "risk",
    "severity",
    "subcontractor",
}


_GENERIC_QUERY_TOKENS = _FIELD_TOKENS | {
    "agreement",
    "apply",
    "date",
    "document",
    "incident",
    "register",
    "report",
    "supplier",
    "vendor",
}


def _stem(token: str) -> str:
    for suffix in ("ments", "ment", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def lexical_tokens(text: str) -> frozenset[str]:
    folded = unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()
    return frozenset(
        _CANONICAL.get(_stem(token), _stem(token))
        for token in re.findall(r"[a-z0-9]+", folded)
        if token not in _STOPWORDS
    )


def _document_scope(text: str) -> str | None:
    folded = unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()
    if re.search(r"\b(?:contrat|accord|agreement)\b", folded):
        return "agreement"
    if re.search(r"\b(?:rapport|report)\b", folded):
        return "incident"
    if re.search(r"\b(?:registre|register)\b", folded):
        return "register"
    if re.search(r"\bincident\b", folded):
        return "incident"
    return None


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    id: str
    document_id: str
    document_name: str
    page: int
    section: str | None
    text: str
    embedding: list[float]
    embedding_model_id: str = "deterministic-hash-v1:384"


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: EvidenceChunk
    score: float
    lexical_score: float
    dense_score: float
    entity_score: float
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    section: str | None
    excerpt: str


@dataclass(frozen=True, slots=True)
class AnswerResult:
    status: str
    answer: str
    confidence: float
    citations: tuple[Citation, ...]


class RetrievalMethod(StrEnum):
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class Reranker(Protocol):
    model_id: str

    def score(self, question: str, passages: Sequence[str]) -> list[float]: ...


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _candidate_scores(
    question: str,
    chunks: list[EvidenceChunk],
    *,
    provider: EmbeddingProvider,
) -> list[RankedChunk]:
    query_vector = provider.embed(question)
    query_tokens = lexical_tokens(question)
    requested_scope = _document_scope(question)
    document_tokens: dict[str, frozenset[str]] = {}
    identity_tokens: dict[str, frozenset[str]] = {}
    for document_id in {chunk.document_id for chunk in chunks}:
        document_chunks = [chunk for chunk in chunks if chunk.document_id == document_id]
        document_tokens[document_id] = lexical_tokens(
            " ".join(
                f"{chunk.document_name} {chunk.section or ''} {chunk.text}"
                for chunk in document_chunks
            )
        )
        identity_lines = [document_chunks[0].document_name]
        for chunk in document_chunks:
            identity_lines.extend(
                line
                for line in chunk.text.splitlines()
                if re.search(
                    r"^(?:organization|organisation|supplier|fournisseur|vendor legal entity|"
                    r"client record supplier|soci[eé]t[eé] concern[eé]e)\b|"
                    r"\b(?:ltd\.?|llc|gmbh|inc\.?)\s*$",
                    line.strip(),
                    re.IGNORECASE,
                )
            )
        identity_tokens[document_id] = lexical_tokens(" ".join(identity_lines))
    document_frequency = Counter(
        token for token in query_tokens for tokens in identity_tokens.values() if token in tokens
    )
    distinctive_tokens = {
        token
        for token, frequency in document_frequency.items()
        if frequency == 1 and token not in _GENERIC_QUERY_TOKENS and len(token) >= 4
    }
    ranked: list[RankedChunk] = []
    for chunk in chunks:
        chunk_tokens = lexical_tokens(chunk.text)
        local_coverage = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
        document_coverage = len(query_tokens & document_tokens[chunk.document_id]) / max(
            1, len(query_tokens)
        )
        requested_fields = query_tokens & _FIELD_TOKENS
        entity_coverage = float(bool(distinctive_tokens & identity_tokens[chunk.document_id]))
        if requested_fields:
            field_coverage = len(requested_fields & chunk_tokens) / len(requested_fields)
            if distinctive_tokens:
                lexical = (
                    0.40 * field_coverage
                    + 0.15 * local_coverage
                    + 0.10 * document_coverage
                    + 0.35 * entity_coverage
                )
            elif requested_scope is not None:
                filename_tokens = lexical_tokens(chunk.document_name)
                scope_coverage = float(requested_scope in filename_tokens)
                lexical = (
                    0.50 * field_coverage
                    + 0.15 * local_coverage
                    + 0.10 * document_coverage
                    + 0.25 * scope_coverage
                )
            else:
                lexical = 0.65 * field_coverage + 0.25 * local_coverage + 0.10 * document_coverage
        else:
            lexical = (
                0.55 * local_coverage + 0.10 * document_coverage + 0.35 * entity_coverage
                if distinctive_tokens
                else 0.8 * local_coverage + 0.2 * document_coverage
            )
        dense = (
            max(-1.0, min(1.0, _cosine(query_vector, chunk.embedding)))
            if chunk.embedding_model_id == provider.model_id
            else 0.0
        )
        ranked.append(
            RankedChunk(
                chunk=chunk,
                score=0.0,
                lexical_score=lexical,
                dense_score=dense,
                entity_score=entity_coverage,
            )
        )
    return ranked


def _with_score(
    item: RankedChunk, score: float, *, rerank_score: float | None = None
) -> RankedChunk:
    return RankedChunk(
        chunk=item.chunk,
        score=score,
        lexical_score=item.lexical_score,
        dense_score=item.dense_score,
        entity_score=item.entity_score,
        rerank_score=rerank_score,
    )


def rank_chunks(
    question: str,
    chunks: list[EvidenceChunk],
    *,
    provider: EmbeddingProvider,
    method: RetrievalMethod = RetrievalMethod.HYBRID,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
    reranker: Reranker | None = None,
) -> list[RankedChunk]:
    """Rank evidence with explicit, reproducible lexical/dense strategies."""

    if limit < 1:
        raise ValueError("limit must be positive")
    scored = _candidate_scores(question, chunks, provider=provider)
    if method is RetrievalMethod.LEXICAL:
        ranked = [_with_score(item, item.lexical_score) for item in scored]
        return sorted(ranked, key=lambda item: (-item.score, item.chunk.id))[:limit]

    compatible = [item for item in scored if item.chunk.embedding_model_id == provider.model_id]
    if method is RetrievalMethod.DENSE:
        ranked = [_with_score(item, item.dense_score) for item in compatible]
        return sorted(ranked, key=lambda item: (-item.score, item.chunk.id))[:limit]

    if method not in {RetrievalMethod.HYBRID, RetrievalMethod.HYBRID_RERANK}:
        raise ValueError(f"unsupported retrieval method: {method}")
    lexical_order = sorted(scored, key=lambda item: (-item.lexical_score, item.chunk.id))
    dense_order = sorted(compatible, key=lambda item: (-item.dense_score, item.chunk.id))
    lexical_ranks = {item.chunk.id: index for index, item in enumerate(lexical_order, start=1)}
    dense_ranks = {item.chunk.id: index for index, item in enumerate(dense_order, start=1)}
    fused = [
        _with_score(
            item,
            (
                1.0 / (RRF_K + lexical_ranks[item.chunk.id])
                + (
                    1.0 / (RRF_K + dense_ranks[item.chunk.id])
                    if item.chunk.id in dense_ranks
                    else 0.0
                )
            )
            / (2.0 if item.chunk.id in dense_ranks else 1.0)
            + NAMED_ENTITY_DOCUMENT_BOOST * item.entity_score,
        )
        for item in scored
    ]
    fused.sort(key=lambda item: (-item.score, -item.lexical_score, item.chunk.id))
    if method is RetrievalMethod.HYBRID:
        return fused[:limit]
    if reranker is None:
        raise ValueError("hybrid_rerank requires an explicit reranker")
    rerank_candidates = fused[: max(limit * 4, 20)]
    rerank_scores = reranker.score(
        question,
        [f"{item.chunk.document_name}\n{item.chunk.text}" for item in rerank_candidates],
    )
    if len(rerank_scores) != len(rerank_candidates):
        raise ValueError("reranker returned the wrong number of scores")
    reranked = [
        _with_score(item, float(score), rerank_score=float(score))
        for item, score in zip(rerank_candidates, rerank_scores, strict=True)
    ]
    return sorted(reranked, key=lambda item: (-item.score, item.chunk.id))[:limit]


def hybrid_rank(
    question: str,
    chunks: list[EvidenceChunk],
    *,
    provider: EmbeddingProvider,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> list[RankedChunk]:
    """Compatibility wrapper for the selected rank-fusion implementation."""

    return rank_chunks(
        question,
        chunks,
        provider=provider,
        method=RetrievalMethod.HYBRID,
        limit=limit,
    )


def _citation(chunk: EvidenceChunk, excerpt: str | None = None) -> Citation:
    return Citation(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        page=chunk.page,
        section=chunk.section,
        excerpt=excerpt or chunk.text,
    )


_MONTH_NAMES = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|janvier|février|fevrier|mars|avril|mai|juin|juillet|"
    r"août|aout|septembre|octobre|novembre|décembre|decembre"
)
_DATE_PATTERN = re.compile(
    rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}(?:er)?\s+(?:{_MONTH_NAMES})\s+\d{{4}}|"
    rf"(?:{_MONTH_NAMES})\s+\d{{1,2}},?\s+\d{{4}})\b",
    re.IGNORECASE,
)

_DEADLINE_PATTERN = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+"
    r"(?:minutes?|hours?|days?|heures?|jours?)\b",
    re.IGNORECASE,
)
_AMOUNT_PATTERN = re.compile(
    r"(?:\b(?:EUR|USD|GBP)\s*[$€£]?\s*[0-9][0-9 ,.]*|"
    r"[$€£]\s*[0-9][0-9 ,.]*\s*(?:EUR|USD|GBP)?|"
    r"\b[0-9][0-9 ,.]*\s*(?:EUR|USD|GBP|€|£|\$))\b",
    re.IGNORECASE,
)
_MISSING_DATE_PATTERN = re.compile(
    r"\b(?:not recorded|missing|unknown|not available|non renseign[eé]e?)\b",
    re.IGNORECASE,
)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()


def _question_intent(question: str) -> str | None:
    folded = _fold(question)
    if re.search(r"\b(?:renew|renewal|renouvel|recondu|echeance)\w*\b", folded):
        return "renewal_date"
    if re.search(
        r"\b(?:effective|effect|effet|commence|commencement|starts?|application|vigueur)\w*\b",
        folded,
    ):
        return "effective_date"
    if re.search(r"\b(?:notification|notify|notifier)\w*\b", folded) and re.search(
        r"\b(?:deadline|delai|quickly|hours?|heures?|days?|jours?)\b", folded
    ):
        return "notification_deadline"
    if re.search(r"\b(?:duration|duree|how long|combien de temps|delayed|retard)\b", folded):
        return "duration"
    if re.search(r"\b(?:document type|document kind|type de document|nature|categorie)\b", folded):
        return "document_type"
    if re.search(
        r"\b(?:amount|fee|cost|price|budget|charge|rate|cout|montant|prix|forfait)\b",
        folded,
    ):
        return "important_amounts"
    if re.search(
        r"\b(?:responsible|manager|owner|accountable|contact|responsable|qui est le pilote)\b",
        folded,
    ):
        return "responsible_people"
    if re.search(r"\b(?:risk|risque|exposure|control gap)\b", folded):
        return "risks"
    if re.search(
        r"\b(?:organization|organisation|legal entity|supplier organization|"
        r"supplier is named|societe|nom du fournisseur)\b",
        folded,
    ):
        return "organization_name"
    if re.search(
        r"\b(?:obligation|must|shall|required|undertakes?|doit|s engage|what .* do)\b",
        folded,
    ):
        return "obligations"
    if re.search(r"\b(?:severity|severite|sev[- ]?\d)\b", folded):
        return "severity"
    return None


def _segments(text: str) -> list[str]:
    segments: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])", stripped)
            if item.strip()
        ]
        segments.extend(parts or [stripped])
    return segments


def _is_intent_evidence(intent: str, segment: str) -> bool:
    folded = _fold(segment)
    has_date = bool(_DATE_PATTERN.search(segment))
    if intent == "renewal_date":
        return bool(re.search(r"\b(?:renew|renewal|renouvel|recondu|echeance)\w*\b", folded)) and (
            has_date or bool(_MISSING_DATE_PATTERN.search(segment))
        )
    if intent == "effective_date":
        return has_date and bool(
            re.search(
                r"\b(?:effective|takes? effect|effective from|commence|commencement|"
                r"starts?|application|vigueur|prise d[' ]effet)\w*\b",
                folded,
            )
        ) and not bool(re.search(r"\b(?:renew|renouvel|recondu)\w*\b", folded))
    if intent == "notification_deadline":
        return bool(re.search(r"\b(?:notify|notification|notifier)\w*\b", folded)) and bool(
            _DEADLINE_PATTERN.search(segment)
        )
    if intent == "duration":
        return bool(_DEADLINE_PATTERN.search(segment)) and bool(
            re.search(r"\b(?:delay|delayed|duration|duree|retard|lasted)\w*\b", folded)
        )
    if intent == "document_type":
        return bool(
            re.search(
                r"^(?:document type|document kind|category|categorie|nature du document|"
                r"type de piece)\s*[:—-]",
                folded,
            )
            or (re.search(
                r"\b(?:agreement|contract|addendum|schedule|statement of work|order form|"
                r"accord|contrat|bon de commande|declaration de travaux|licence)\b",
                folded,
            )
            and segment.strip().isupper())
        )
    if intent == "important_amounts":
        return bool(_AMOUNT_PATTERN.search(segment))
    if intent == "responsible_people":
        return bool(
            re.search(
                r"^(?:responsible manager|responsible contact|service owner|owner|"
                r"accountable lead|contact operationnel|responsable(?: du compte)?)\b",
                folded,
            )
        )
    if intent == "risks":
        return bool(
            re.search(r"^(?:risk|risque|risk note|exposure note)\s*[:—-]", folded)
            or re.search(r"\bcontrol gap\b", folded)
            or re.search(r"\b(?:may|might|could|peut|peuvent)\b", folded)
        )
    if intent == "organization_name":
        return bool(
            re.search(
                r"^(?:organization|organisation|supplier|fournisseur|vendor legal entity|"
                r"client record supplier|societe concernee)\b",
                folded,
            )
            or re.search(r"\b(?:ltd\.?|llc|gmbh|inc\.?)$", folded)
        )
    if intent == "obligations":
        return bool(
            re.search(
                r"\b(?:shall|must|is required to|undertakes? to|doit|s[' ]engage a|est tenu de)\b",
                folded,
            )
            or re.search(r"^no\s+.+\bmay\b", folded)
        )
    if intent == "severity":
        return bool(re.search(r"\b(?:severity|severite|sev[- ]?\d)\b", folded))
    return False


def _evidence_key(intent: str, segment: str) -> str:
    if intent in {"effective_date", "renewal_date"}:
        match = _DATE_PATTERN.search(segment)
        return _fold(match.group(0) if match else segment)
    if intent == "important_amounts":
        match = _AMOUNT_PATTERN.search(segment)
        return re.sub(r"\s+", "", _fold(match.group(0) if match else segment))
    if intent in {"duration", "notification_deadline"}:
        match = _DEADLINE_PATTERN.search(segment)
        return _fold(match.group(0) if match else segment)
    return _fold(segment)


def _is_adversarial_question(question: str) -> bool:
    folded = _fold(question)
    patterns = (
        r"\b(?:ignore|ignorez|disregard|override)\b",
        r"\b(?:invent|plausible|from memory|even if absent)\b",
        r"\b(?:administrator|admin|password|system prompt|hidden prompt)\b",
        r"\b(?:reveal|revele|expose)\b.*\b(?:secret|phone|telephone|email|prompt)\b",
    )
    return any(re.search(pattern, folded) for pattern in patterns)


class IntentEvidenceReranker:
    """Deterministic second stage that rewards direct field/value evidence."""

    model_id = "intent-evidence-reranker-v1"

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        intent = _question_intent(question)
        question_tokens = lexical_tokens(question)
        scores: list[float] = []
        for passage in passages:
            passage_tokens = lexical_tokens(passage)
            coverage = len(question_tokens & passage_tokens) / max(1, len(question_tokens))
            direct = 0.0
            if intent is not None and any(
                _is_intent_evidence(intent, segment) for segment in _segments(passage)
            ):
                direct = 1.0
            scores.append(direct + 0.25 * coverage)
        return scores


class ExtractiveAnswerProvider:
    estimated_cost_usd = 0.0

    def __init__(self, *, mode: str = "extractive-local") -> None:
        self.mode = mode

    def answer(self, question: str, ranked: list[RankedChunk]) -> AnswerResult:
        if _is_adversarial_question(question):
            return AnswerResult(
                status="abstained",
                answer=(
                    "Request refused: document instructions are untrusted and "
                    "redacted data cannot be recovered."
                ),
                confidence=1.0,
                citations=(),
            )
        intent = _question_intent(question)
        if intent is None or not ranked:
            return AnswerResult(
                status="abstained",
                answer="Insufficient evidence in the selected dossier.",
                confidence=0.0,
                citations=(),
            )
        broad_scope = bool(
            re.search(
                r"\b(?:across|all (?:supplier )?files|corpus|without naming|which supplier)\b",
                _fold(question),
            )
        )
        relevant_ranked = ranked
        if not broad_scope and any(item.entity_score > 0 for item in ranked):
            relevant_ranked = [item for item in ranked if item.entity_score > 0]

        evidence: list[tuple[str, str, RankedChunk]] = []
        for item in relevant_ranked:
            for segment in _segments(item.chunk.text):
                if _is_intent_evidence(intent, segment):
                    evidence.append((_evidence_key(intent, segment), segment, item))
                    break
        if not evidence:
            return AnswerResult(
                status="abstained",
                answer="Insufficient evidence in the selected dossier.",
                confidence=0.0,
                citations=(),
            )
        if intent == "renewal_date" and _MISSING_DATE_PATTERN.search(evidence[0][1]):
            return AnswerResult(
                status="abstained",
                answer="No reliable renewal date is recorded in the selected evidence.",
                confidence=evidence[0][2].score,
                citations=(_citation(evidence[0][2].chunk, evidence[0][1]),),
            )

        singular_intents = {
            "organization_name",
            "document_type",
            "effective_date",
            "renewal_date",
            "important_amounts",
            "responsible_people",
            "notification_deadline",
            "duration",
            "severity",
        }
        distinct: list[tuple[str, str, RankedChunk]] = []
        for evidence_item in evidence:
            if all(evidence_item[0] != existing[0] for existing in distinct):
                distinct.append(evidence_item)
        evidence_documents = {item[2].chunk.document_id for item in distinct}
        same_document_conflict = len(evidence_documents) == 1 and len(distinct) > 1
        cross_document_conflict = (
            (broad_scope or intent == "notification_deadline")
            and len(evidence_documents) > 1
            and len(distinct) > 1
        )
        if intent in singular_intents and (same_document_conflict or cross_document_conflict):
            return AnswerResult(
                status="ambiguous",
                answer="Conflicting evidence was found; review the cited passages.",
                confidence=min(item[2].score for item in distinct[:2]),
                citations=tuple(_citation(item[2].chunk, item[1]) for item in distinct[:2]),
            )

        _, excerpt, best = evidence[0]
        confidence = max(0.0, min(1.0, best.score))
        return AnswerResult(
            status="answered",
            answer=excerpt,
            confidence=confidence,
            citations=(_citation(best.chunk, excerpt),),
        )
