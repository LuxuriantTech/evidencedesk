"""Fail-closed local ONNX NLI scoring for EvidenceDesk answer experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from evidencedesk_api.answering import (
    DeterministicGroundedAnswerProvider,
    GroundedAnswer,
    validate_grounded_answer,
)
from evidencedesk_api.retrieval import RankedChunk

DEFAULT_EXPERIMENT_PLAN = Path("evals/configs/answer-v3-experiment-plan.json")
_REQUIRED_LABEL_ORDER = ("entailment", "neutral", "contradiction")


class NliModelIntegrityError(RuntimeError):
    """Raised before scoring when a local NLI model cannot be attested."""


class TokenizerLike(Protocol):
    def encode(self, sequence: str, pair: str | None = None) -> Any: ...


class OnnxSessionLike(Protocol):
    def get_inputs(self) -> Sequence[Any]: ...

    def run(
        self, output_names: Sequence[str] | None, input_feed: dict[str, Any]
    ) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class NliScores:
    contradiction: float
    neutral: float
    entailment: float

    @property
    def label(self) -> str:
        scores = {
            "contradiction": self.contradiction,
            "neutral": self.neutral,
            "entailment": self.entailment,
        }
        return max(scores, key=scores.__getitem__)


@dataclass(frozen=True, slots=True)
class NliDecisionConfig:
    entailment_threshold: float
    contradiction_threshold: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (self.entailment_threshold, self.contradiction_threshold)
        ):
            raise ValueError("NLI thresholds must be finite values between zero and one")


class NliScorer(Protocol):
    def score_question_hypothesis(
        self, *, question: str, hypothesis: str, passage: str
    ) -> NliScores: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NliModelIntegrityError(f"invalid {label}") from exc
    if not isinstance(loaded, dict):
        raise NliModelIntegrityError(f"invalid {label}")
    return cast(dict[str, Any], loaded)


def _expected_from_plan(path: Path) -> dict[str, Any]:
    plan = _load_object(path, label="answer experiment plan")
    strategies = plan.get("strategies")
    if not isinstance(strategies, list):
        raise NliModelIntegrityError("invalid answer experiment plan")
    selected = next(
        (
            item
            for item in strategies
            if isinstance(item, dict) and item.get("id") == "multilingual-nli-v3"
        ),
        None,
    )
    if not isinstance(selected, dict) or selected.get("license") != "MIT":
        raise NliModelIntegrityError("NLI model license is not pre-registered as MIT")
    model = selected.get("model")
    required = ("id", "revision", "onnx_file", "onnx_sha256", "onnx_size_bytes")
    if not isinstance(model, dict) or any(key not in model for key in required):
        raise NliModelIntegrityError("invalid pre-registered NLI model")
    if (
        not isinstance(model["id"], str)
        or not isinstance(model["revision"], str)
        or not isinstance(model["onnx_file"], str)
        or not isinstance(model["onnx_sha256"], str)
        or not isinstance(model["onnx_size_bytes"], int)
        or re.fullmatch(r"[0-9a-f]{64}", model["onnx_sha256"]) is None
        or model["onnx_size_bytes"] < 1
    ):
        raise NliModelIntegrityError("invalid pre-registered NLI model")
    return cast(dict[str, Any], model)


def _verified_file(root: Path, item: dict[str, Any]) -> Path:
    relative = Path(str(item.get("path", "")))
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise NliModelIntegrityError(f"missing model file: {relative.as_posix()}")
    expected_size = item.get("bytes")
    expected_hash = item.get("sha256")
    if (
        not isinstance(expected_size, int)
        or expected_size < 1
        or candidate.stat().st_size != expected_size
    ):
        raise NliModelIntegrityError(f"size mismatch: {relative.as_posix()}")
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise NliModelIntegrityError(f"invalid sha256: {relative.as_posix()}")
    if _sha256(candidate) != expected_hash:
        raise NliModelIntegrityError(f"sha256 mismatch: {relative.as_posix()}")
    return candidate


def _tokenizer(path: Path) -> TokenizerLike:
    from tokenizers import Tokenizer

    return cast(TokenizerLike, Tokenizer.from_file(str(path)))


def _session(path: Path) -> OnnxSessionLike:
    import onnxruntime  # type: ignore[import-untyped]

    return cast(
        OnnxSessionLike,
        onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]),
    )


class LocalOnnxNliScorer:
    """Scores a passage-premise against a hypothesis with a verified CPU model."""

    estimated_cost_usd = 0.0

    def __init__(
        self,
        *,
        manifest_path: Path,
        model_path: Path,
        experiment_plan_path: Path = DEFAULT_EXPERIMENT_PLAN,
        tokenizer_factory: Callable[[Path], TokenizerLike] | None = None,
        session_factory: Callable[[Path], OnnxSessionLike] | None = None,
    ) -> None:
        expected = _expected_from_plan(experiment_plan_path)
        manifest = _load_object(manifest_path, label="NLI model manifest")
        if (
            manifest.get("model_id") != expected["id"]
            or manifest.get("repository") != expected["id"]
            or manifest.get("repository_revision") != expected["revision"]
            or manifest.get("license") != "MIT"
            or manifest.get("onnx_file") != expected["onnx_file"]
            or manifest.get("onnx_sha256") != expected["onnx_sha256"]
            or manifest.get("onnx_size_bytes") != expected["onnx_size_bytes"]
        ):
            raise NliModelIntegrityError("unexpected pre-registered NLI model identity")
        label_order = manifest.get("label_order")
        if (
            not isinstance(label_order, list)
            or not all(isinstance(label, str) for label in label_order)
            or tuple(label_order) != _REQUIRED_LABEL_ORDER
        ):
            raise NliModelIntegrityError("NLI label order is not confirmed")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise NliModelIntegrityError("invalid NLI model manifest")
        root = model_path.resolve()
        verified: dict[str, Path] = {}
        for item in files:
            if not isinstance(item, dict):
                raise NliModelIntegrityError("invalid NLI model manifest")
            candidate = _verified_file(root, item)
            verified[str(item["path"])] = candidate
        onnx_path = verified.get(str(expected["onnx_file"]))
        if onnx_path is None:
            raise NliModelIntegrityError("NLI ONNX file is absent from manifest")
        if (
            onnx_path.stat().st_size != expected["onnx_size_bytes"]
            or _sha256(onnx_path) != expected["onnx_sha256"]
        ):
            raise NliModelIntegrityError("NLI ONNX file differs from pre-registered model")
        config_path = verified.get("config.json")
        tokenizer_path = verified.get("tokenizer.json")
        if config_path is None or tokenizer_path is None:
            raise NliModelIntegrityError("NLI config.json and tokenizer.json must be verified")
        config = _load_object(config_path, label="NLI config")
        id2label = config.get("id2label")
        if not isinstance(id2label, dict):
            raise NliModelIntegrityError("NLI config is missing id2label")
        observed = tuple(str(id2label.get(str(index), "")).casefold() for index in range(3))
        if observed != _REQUIRED_LABEL_ORDER:
            raise NliModelIntegrityError("NLI label order differs from verified config")

        self.model_id = f"{expected['id']}@{expected['revision']}"
        self._tokenizer = (tokenizer_factory or _tokenizer)(tokenizer_path)
        self._session = (session_factory or _session)(onnx_path)
        self._input_names = {str(item.name) for item in self._session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(self._input_names):
            raise NliModelIntegrityError("NLI session has unsupported input names")

    def score(self, *, passage: str, hypothesis: str) -> NliScores:
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError("passage must be a non-empty string")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("hypothesis must be a non-empty string")
        encoded = self._tokenizer.encode(passage, hypothesis)
        inputs: dict[str, Any] = {
            "input_ids": np.asarray([encoded.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.asarray([encoded.type_ids], dtype=np.int64)
        outputs = self._session.run(None, inputs)
        if not outputs:
            raise NliModelIntegrityError("NLI session returned no logits")
        logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if logits.size != 3 or not np.isfinite(logits).all():
            raise NliModelIntegrityError("NLI session returned invalid logits")
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        if not np.isfinite(probabilities).all() or not math.isclose(
            float(np.sum(probabilities)), 1.0
        ):
            raise NliModelIntegrityError("NLI logits cannot be normalized")
        return NliScores(
            contradiction=float(probabilities[2]),
            neutral=float(probabilities[1]),
            entailment=float(probabilities[0]),
        )

    def score_question_hypothesis(
        self, *, question: str, hypothesis: str, passage: str
    ) -> NliScores:
        """Score a proposed answer claim while retaining the question for provenance."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        return self.score(passage=passage, hypothesis=f"Question: {question}\nClaim: {hypothesis}")


class NliGroundedAnswerProvider:
    """Use multilingual NLI as a fail-closed gate over extracted answer candidates."""

    estimated_cost_usd = 0.0

    def __init__(
        self,
        *,
        base: DeterministicGroundedAnswerProvider,
        scorer: NliScorer,
        config: NliDecisionConfig,
        mode: str,
    ) -> None:
        self.base = base
        self.scorer = scorer
        self.config = config
        self.mode = mode

    def _partial(self, draft: GroundedAnswer, confidence: float) -> GroundedAnswer:
        return replace(
            draft,
            status="partially_supported",
            answerable=False,
            answer="The passage does not entail the proposed answer strongly enough.",
            confidence=confidence,
            ambiguity_reason="The candidate answer is only partially supported by local NLI.",
            extracted_fields={},
        )

    def answer(self, question: str, ranked: list[RankedChunk]) -> GroundedAnswer:
        draft = self.base.answer(question, ranked)
        if draft.status == "answered" and draft.supporting_excerpt is not None:
            scores = self.scorer.score_question_hypothesis(
                question=question,
                hypothesis=draft.answer,
                passage=draft.supporting_excerpt,
            )
            if scores.entailment >= self.config.entailment_threshold:
                result = replace(draft, confidence=scores.entailment)
            elif scores.contradiction >= self.config.contradiction_threshold:
                result = GroundedAnswer(
                    status="abstained",
                    answerable=False,
                    answer="The proposed answer conflicts with the retrieved passage.",
                    confidence=scores.contradiction,
                    supporting_document=None,
                    supporting_page=None,
                    supporting_excerpt=None,
                    ambiguity_reason=None,
                    extracted_fields={},
                    supporting_chunk_ids=(),
                    citations=(),
                )
            else:
                result = self._partial(draft, scores.entailment)
            validate_grounded_answer(result, ranked)
            return result
        if draft.status == "ambiguous" and len(draft.citations) >= 2:
            first, second = draft.citations[:2]
            scores = self.scorer.score_question_hypothesis(
                question=question,
                hypothesis=second.excerpt,
                passage=first.excerpt,
            )
            if scores.contradiction >= self.config.contradiction_threshold:
                result = replace(draft, confidence=scores.contradiction)
            else:
                result = self._partial(draft, scores.entailment)
            validate_grounded_answer(result, ranked)
            return result
        return draft
