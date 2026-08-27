"""Strict, local Qwen JSON adapter for grounded answer experiments.

The module deliberately depends on a tiny injected backend protocol.  Production
can adapt ``llama.cpp`` without making the test suite load model weights.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from evidencedesk_api.answering import AnswerStatus, GroundedAnswer, validate_grounded_answer
from evidencedesk_api.retrieval import Citation, RankedChunk
from evidencedesk_api.trust_boundaries import (
    SYSTEM_INSTRUCTIONS,
    TrustSeparatedInput,
    UntrustedDocumentContent,
    is_untrusted_document_instruction,
    untrusted_document_fragment_indexes,
)


@dataclass(frozen=True, slots=True)
class QwenModelManifest:
    model_id: str
    revision: str
    filename: str
    file_sha256: str
    file_size_bytes: int
    parameter_count: int
    license: str
    source: str


QWEN_05B_INSTRUCT_Q4_K_M = QwenModelManifest(
    model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    revision="9217f5db79a29953eb74d5343926648285ec7e67",
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    file_sha256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
    file_size_bytes=491400032,
    parameter_count=490000000,
    license="Apache-2.0",
    source="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
)


@dataclass(frozen=True, slots=True)
class QwenRunConfig:
    seed: int
    max_output_tokens: int = 160
    temperature: float = 0.0
    context_passages: int = 3

    def __post_init__(self) -> None:
        if self.temperature != 0.0:
            raise ValueError("Qwen adapter requires temperature=0.0")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.context_passages <= 0:
            raise ValueError("context_passages must be positive")


class QwenBackend(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> str: ...


class QwenDecisionError(ValueError):
    """The local model returned an unusable or ungrounded decision."""


class QwenModelIntegrityError(RuntimeError):
    """The local GGUF file does not match the pinned public model."""


FieldValue = str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QwenDecision:
    answerable: bool
    answer: str | None
    confidence: float
    supporting_document: str | None
    supporting_page: int | None
    supporting_excerpt: str | None
    ambiguity_reason: str | None
    extracted_fields: dict[str, FieldValue]

    @property
    def status(self) -> AnswerStatus:
        if self.answerable:
            return "answered"
        if self.ambiguity_reason:
            return "ambiguous"
        return "abstained"


_REQUIRED_KEYS = frozenset(
    {
        "answerable",
        "answer",
        "confidence",
        "supporting_document",
        "supporting_page",
        "supporting_excerpt",
        "ambiguity_reason",
        "extracted_fields",
    }
)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_qwen_model(
    manifest_path: Path,
    model_path: Path,
    *,
    expected: QwenModelManifest = QWEN_05B_INSTRUCT_Q4_K_M,
) -> Path:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenModelIntegrityError("invalid Qwen model manifest") from exc
    if not isinstance(payload, dict):
        raise QwenModelIntegrityError("invalid Qwen model manifest")
    identity = {
        "model_id": expected.model_id,
        "repository": expected.model_id,
        "repository_revision": expected.revision,
        "license": expected.license,
        "parameter_count": expected.parameter_count,
        "file": expected.filename,
        "file_size_bytes": expected.file_size_bytes,
        "file_sha256": expected.file_sha256,
    }
    if any(payload.get(key) != value for key, value in identity.items()):
        raise QwenModelIntegrityError("unexpected pinned Qwen model identity")
    root = model_path.resolve()
    candidate = (root / expected.filename).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise QwenModelIntegrityError("missing pinned Qwen model file")
    if candidate.stat().st_size != expected.file_size_bytes:
        raise QwenModelIntegrityError("Qwen model size mismatch")
    if _sha256(candidate) != expected.file_sha256:
        raise QwenModelIntegrityError("Qwen model sha256 mismatch")
    return candidate


def _text_or_none(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise QwenDecisionError(f"{key} must be a string or null")
    return value


def _field_values(value: object) -> dict[str, FieldValue]:
    if not isinstance(value, dict):
        raise QwenDecisionError("extracted_fields must be an object")
    result: dict[str, FieldValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise QwenDecisionError("extracted_fields keys must be non-empty strings")
        if isinstance(item, str):
            result[key] = item
        elif isinstance(item, list) and all(isinstance(part, str) for part in item):
            result[key] = tuple(item)
        else:
            raise QwenDecisionError("extracted_fields values must be strings or string arrays")
    return result


def _parse_decision(raw: str) -> QwenDecision:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QwenDecisionError("model response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise QwenDecisionError("model response must be a JSON object")
    keys = frozenset(value)
    if keys - _REQUIRED_KEYS:
        raise QwenDecisionError("model response contains unexpected keys")
    if _REQUIRED_KEYS - keys:
        raise QwenDecisionError("model response is missing required keys")
    answerable = value["answerable"]
    confidence = value["confidence"]
    page = value["supporting_page"]
    if not isinstance(answerable, bool):
        raise QwenDecisionError("answerable must be a boolean")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise QwenDecisionError("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise QwenDecisionError("confidence must be between 0 and 1")
    if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
        raise QwenDecisionError("supporting_page must be a positive integer or null")
    decision = QwenDecision(
        answerable=answerable,
        answer=_text_or_none(value["answer"], "answer"),
        confidence=float(confidence),
        supporting_document=_text_or_none(value["supporting_document"], "supporting_document"),
        supporting_page=page,
        supporting_excerpt=_text_or_none(value["supporting_excerpt"], "supporting_excerpt"),
        ambiguity_reason=_text_or_none(value["ambiguity_reason"], "ambiguity_reason"),
        extracted_fields=_field_values(value["extracted_fields"]),
    )
    if decision.answerable and (not decision.answer or not decision.answer.strip()):
        raise QwenDecisionError("answerable decisions require a non-empty answer")
    if decision.answerable and decision.ambiguity_reason is not None:
        raise QwenDecisionError("answerable decisions cannot have an ambiguity reason")
    if not decision.answerable and decision.ambiguity_reason is None:
        if decision.answer is not None or any(
            value is not None
            for value in (
                decision.supporting_document,
                decision.supporting_page,
                decision.supporting_excerpt,
            )
        ):
            raise QwenDecisionError("abstained decisions cannot contain answer or support")
        if decision.extracted_fields:
            raise QwenDecisionError("abstained decisions cannot contain extracted fields")
    if (
        not decision.answerable
        and decision.ambiguity_reason is not None
        and decision.answer is not None
    ):
        raise QwenDecisionError("ambiguous decisions cannot contain an answer")
    return decision


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _values_are_supported(values: Mapping[str, FieldValue], excerpt: str) -> bool:
    normalised_excerpt = _normalise(excerpt)
    for value in values.values():
        parts = (value,) if isinstance(value, str) else value
        if any(_normalise(part) not in normalised_excerpt for part in parts if part.strip()):
            return False
    return True


def validate_qwen_decision(
    decision: QwenDecision | Mapping[str, Any],
    ranked: Sequence[RankedChunk],
) -> QwenDecision:
    """Validate exact schema and grounding against retrieved passages only."""

    parsed = _parse_decision(json.dumps(decision)) if isinstance(decision, Mapping) else decision
    if not isinstance(parsed, QwenDecision):
        raise QwenDecisionError("decision must use the Qwen schema")
    support = (
        parsed.supporting_document,
        parsed.supporting_page,
        parsed.supporting_excerpt,
    )
    if any(item is not None for item in support) and any(item is None for item in support):
        raise QwenDecisionError("supporting document/page/excerpt must be supplied together")
    if parsed.supporting_excerpt is None:
        if parsed.answerable or parsed.extracted_fields or parsed.ambiguity_reason:
            raise QwenDecisionError("answerable decisions require supporting evidence")
        return parsed
    if is_untrusted_document_instruction(parsed.supporting_excerpt):
        raise QwenDecisionError("instruction-like document content cannot support an answer")
    matching = [
        item.chunk
        for item in ranked
        if item.chunk.document_id == parsed.supporting_document
        and item.chunk.page == parsed.supporting_page
    ]
    if not matching:
        raise QwenDecisionError("supporting document/page is not among retrieved passages")
    if not any(parsed.supporting_excerpt in item.text for item in matching):
        raise QwenDecisionError("supporting excerpt is not present in the cited passage")
    if any(
        parsed.supporting_excerpt in line
        for item in matching
        for index, line in enumerate(item.text.splitlines())
        if index in untrusted_document_fragment_indexes(item.text.splitlines())
    ):
        raise QwenDecisionError("instruction-like document content cannot support an answer")
    if (
        parsed.answerable
        and parsed.answer is not None
        and _normalise(parsed.answer) not in _normalise(parsed.supporting_excerpt)
    ):
        raise QwenDecisionError("answer is not supported by the cited excerpt")
    if not _values_are_supported(parsed.extracted_fields, parsed.supporting_excerpt):
        raise QwenDecisionError("extracted fields are not supported by the cited excerpt")
    return parsed


def _build_prompt(question: str, ranked: Sequence[RankedChunk]) -> str:
    separated = TrustSeparatedInput.build(
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
    return separated.user_payload()


class QwenAnswerProvider:
    """Adapter with strict output and grounding gates; it never fabricates a fallback."""

    mode = "local-qwen-constrained-json-v3"
    estimated_cost_usd = 0.0

    def __init__(self, backend: QwenBackend, *, config: QwenRunConfig) -> None:
        self._backend = backend
        self._config = config

    def answer(self, question: str, ranked: list[RankedChunk]) -> GroundedAnswer:
        context = ranked[: self._config.context_passages]
        raw = self._backend.complete(
            _build_prompt(question, context),
            temperature=self._config.temperature,
            seed=self._config.seed,
            max_tokens=self._config.max_output_tokens,
        )
        if not isinstance(raw, str):
            raise QwenDecisionError("local backend must return JSON text")
        decision = validate_qwen_decision(_parse_decision(raw), context)
        support = next(
            (
                item
                for item in context
                if item.chunk.document_id == decision.supporting_document
                and item.chunk.page == decision.supporting_page
                and decision.supporting_excerpt is not None
                and decision.supporting_excerpt in item.chunk.text
            ),
            None,
        )
        citation = (
            Citation(
                chunk_id=support.chunk.id,
                document_id=support.chunk.document_id,
                document_name=support.chunk.document_name,
                page=support.chunk.page,
                section=support.chunk.section,
                excerpt=decision.supporting_excerpt or "",
            )
            if support is not None
            else None
        )
        status = decision.status
        answer = decision.answer or (
            "Conflicting evidence was found; review the cited passage."
            if status == "ambiguous"
            else "Insufficient evidence in the selected dossier."
        )
        grounded = GroundedAnswer(
            status=status,
            answerable=decision.answerable,
            answer=answer,
            confidence=decision.confidence,
            supporting_document=decision.supporting_document,
            supporting_page=decision.supporting_page,
            supporting_excerpt=decision.supporting_excerpt,
            ambiguity_reason=decision.ambiguity_reason,
            extracted_fields=decision.extracted_fields,
            supporting_chunk_ids=(support.chunk.id,) if support is not None else (),
            citations=(citation,) if citation is not None else (),
        )
        validate_grounded_answer(grounded, context)
        return grounded


class LlamaLike(Protocol):
    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]: ...


class LlamaCppBackend:
    """Verified CPU-only llama.cpp backend with grammar-constrained JSON output."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        model_path: Path,
        n_threads: int,
        n_ctx: int = 4096,
    ) -> None:
        from llama_cpp import Llama  # type: ignore[import-not-found]

        verified = verify_qwen_model(manifest_path, model_path)
        self._model = cast(
            LlamaLike,
            Llama(
                model_path=str(verified),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=0,
                verbose=False,
            ),
        )

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> str:
        string_or_null = {"type": ["string", "null"]}
        schema = {
            "type": "object",
            "properties": {
                "answerable": {"type": "boolean"},
                "answer": string_or_null,
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "supporting_document": string_or_null,
                "supporting_page": {"type": ["integer", "null"], "minimum": 1},
                "supporting_excerpt": string_or_null,
                "ambiguity_reason": string_or_null,
                "extracted_fields": {
                    "type": "object",
                    "additionalProperties": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                },
            },
            "required": sorted(_REQUIRED_KEYS),
            "additionalProperties": False,
        }
        output = self._model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTIONS,
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object", "schema": schema},
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        try:
            content = output["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenDecisionError("llama.cpp returned an invalid completion envelope") from exc
        if not isinstance(content, str):
            raise QwenDecisionError("llama.cpp returned no JSON content")
        return content
