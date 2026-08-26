import re
import unicodedata
from dataclasses import dataclass

from evidencedesk_api.providers import EmbeddingProvider

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


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: EvidenceChunk
    score: float
    lexical_score: float
    dense_score: float


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


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def hybrid_rank(
    question: str,
    chunks: list[EvidenceChunk],
    *,
    provider: EmbeddingProvider,
    limit: int = 5,
) -> list[RankedChunk]:
    query_vector = provider.embed(question)
    query_tokens = lexical_tokens(question)
    requested_scope = _document_scope(question)
    document_tokens: dict[str, frozenset[str]] = {}
    for document_id in {chunk.document_id for chunk in chunks}:
        document_tokens[document_id] = lexical_tokens(
            " ".join(
                f"{chunk.document_name} {chunk.section or ''} {chunk.text}"
                for chunk in chunks
                if chunk.document_id == document_id
            )
        )
    ranked: list[RankedChunk] = []
    for chunk in chunks:
        chunk_tokens = lexical_tokens(chunk.text)
        local_coverage = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
        document_coverage = len(query_tokens & document_tokens[chunk.document_id]) / max(
            1, len(query_tokens)
        )
        requested_fields = query_tokens & _FIELD_TOKENS
        if requested_fields:
            field_coverage = len(requested_fields & chunk_tokens) / len(requested_fields)
            if requested_scope is not None:
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
            lexical = 0.8 * local_coverage + 0.2 * document_coverage
        dense = max(-1.0, min(1.0, _cosine(query_vector, chunk.embedding)))
        score = 0.25 * max(0.0, dense) + 0.75 * lexical
        ranked.append(
            RankedChunk(
                chunk=chunk,
                score=score,
                lexical_score=lexical,
                dense_score=dense,
            )
        )
    return sorted(ranked, key=lambda item: (-item.score, item.chunk.id))[:limit]


def _citation(chunk: EvidenceChunk, excerpt: str | None = None) -> Citation:
    return Citation(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        page=chunk.page,
        section=chunk.section,
        excerpt=excerpt or chunk.text,
    )


_DATE_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
    re.IGNORECASE,
)


class ExtractiveAnswerProvider:
    mode = "extractive-local"
    estimated_cost_usd = 0.0

    def answer(self, question: str, ranked: list[RankedChunk]) -> AnswerResult:
        normalized_question = question.casefold()
        injection_terms = (
            "ignore",
            "ignorez",
            "grant",
            "accorde",
            "administrator",
            "admin",
        )
        pii_terms = ("reveal", "révèle", "phone", "téléphone", "telephone", "email complet")
        if (
            sum(term in normalized_question for term in injection_terms) >= 2
            or sum(term in normalized_question for term in pii_terms) >= 2
        ):
            return AnswerResult(
                status="abstained",
                answer=(
                    "Request refused: document instructions are untrusted and "
                    "redacted data cannot be recovered."
                ),
                confidence=1.0,
                citations=(),
            )
        question_tokens = lexical_tokens(question)
        required_tokens = question_tokens & _REQUIRED_EVIDENCE_TOKENS
        top_tokens = lexical_tokens(ranked[0].chunk.text) if ranked else frozenset()
        if required_tokens - top_tokens or not ranked or ranked[0].lexical_score < 0.30:
            return AnswerResult(
                status="abstained",
                answer="Insufficient evidence in the selected dossier.",
                confidence=0.0,
                citations=(),
            )

        date_query = bool({"date", "renew", "renewal"} & lexical_tokens(question))
        if date_query and len(ranked) > 1:
            first_dates = set(_DATE_PATTERN.findall(ranked[0].chunk.text))
            second_dates = set(_DATE_PATTERN.findall(ranked[1].chunk.text))
            if (
                first_dates
                and second_dates
                and first_dates != second_dates
                and ranked[0].chunk.document_id == ranked[1].chunk.document_id
                and ranked[1].lexical_score >= 0.3
            ):
                return AnswerResult(
                    status="ambiguous",
                    answer="Conflicting evidence was found; review both cited passages.",
                    confidence=min(ranked[0].score, ranked[1].score),
                    citations=(_citation(ranked[0].chunk), _citation(ranked[1].chunk)),
                )

        best = ranked[0]
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", best.chunk.text)
            if sentence.strip()
        ]
        query_tokens = question_tokens
        excerpt = max(
            sentences,
            key=lambda sentence: len(query_tokens & lexical_tokens(sentence)),
            default=best.chunk.text,
        )
        confidence = max(0.0, min(1.0, best.score))
        return AnswerResult(
            status="answered",
            answer=excerpt,
            confidence=confidence,
            citations=(_citation(best.chunk, excerpt),),
        )
