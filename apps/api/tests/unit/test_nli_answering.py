from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest
from evidencedesk_api.answering import DecisionConfig, DeterministicGroundedAnswerProvider
from evidencedesk_api.nli_answering import (
    LocalOnnxNliScorer,
    NliDecisionConfig,
    NliGroundedAnswerProvider,
    NliModelIntegrityError,
    NliScores,
)
from evidencedesk_api.providers import DeterministicEmbeddingProvider
from evidencedesk_api.retrieval import EvidenceChunk, RankedChunk


class FakeEncoding:
    ids: ClassVar[tuple[int, ...]] = (101, 11, 102)
    attention_mask: ClassVar[tuple[int, ...]] = (1, 1, 1)
    type_ids: ClassVar[tuple[int, ...]] = (0, 0, 0)


class FakeTokenizer:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def encode(self, premise: str, hypothesis: str) -> FakeEncoding:
        self.pairs.append((premise, hypothesis))
        return FakeEncoding()


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.requests: list[dict[str, object]] = []

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("input_ids"), FakeInput("attention_mask"), FakeInput("token_type_ids")]

    def run(self, output_names: object, inputs: dict[str, object]) -> list[object]:
        del output_names
        self.requests.append(inputs)
        return [[[self.logits[0], self.logits[1], self.logits[2]]]]


def _write_verified_model(tmp_path: Path, *, labels: list[str] | None = None) -> tuple[Path, Path]:
    model = tmp_path / "onnx" / "model_quantized.onnx"
    model.parent.mkdir()
    model.write_bytes(b"synthetic-onnx")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}},
        ),
        encoding="utf-8",
    )
    files = []
    for path in (model, tokenizer, config):
        files.append(
            {
                "path": path.relative_to(tmp_path).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "model_id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "repository": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "repository_revision": "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c",
        "license": "MIT",
        "onnx_file": "onnx/model_quantized.onnx",
        "onnx_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "onnx_size_bytes": model.stat().st_size,
        "label_order": labels or ["entailment", "neutral", "contradiction"],
        "files": files,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan = {
        "strategies": [
            {
                "id": "multilingual-nli-v3",
                "license": "MIT",
                "model": {
                    "id": manifest["model_id"],
                    "revision": manifest["repository_revision"],
                    "onnx_file": manifest["onnx_file"],
                    "onnx_sha256": manifest["onnx_sha256"],
                    "onnx_size_bytes": manifest["onnx_size_bytes"],
                },
            }
        ]
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return manifest_path, plan_path


def test_verified_local_nli_scores_model_card_label_order_on_cpu(tmp_path: Path) -> None:
    manifest, plan = _write_verified_model(tmp_path)
    tokenizer = FakeTokenizer()
    session = FakeSession([3.0, 0.0, -2.0])

    scorer = LocalOnnxNliScorer(
        manifest_path=manifest,
        model_path=tmp_path,
        experiment_plan_path=plan,
        tokenizer_factory=lambda _path: tokenizer,
        session_factory=lambda _path: session,
    )

    scores = scorer.score(passage="The service begins on 1 May.", hypothesis="It begins on 1 May.")

    assert scorer.model_id == (
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli@"
        "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
    )
    assert scores.entailment > scores.neutral > scores.contradiction
    assert scores.label == "entailment"
    assert tokenizer.pairs == [("The service begins on 1 May.", "It begins on 1 May.")]
    assert set(session.requests[0]) == {"input_ids", "attention_mask", "token_type_ids"}


def test_verified_local_nli_rejects_label_order_that_disagrees_with_config(tmp_path: Path) -> None:
    manifest, plan = _write_verified_model(
        tmp_path, labels=["contradiction", "neutral", "entailment"]
    )

    with pytest.raises(NliModelIntegrityError, match="label order"):
        LocalOnnxNliScorer(
            manifest_path=manifest,
            model_path=tmp_path,
            experiment_plan_path=plan,
            tokenizer_factory=lambda _path: FakeTokenizer(),
            session_factory=lambda _path: FakeSession([0.0, 0.0, 0.0]),
        )


def test_verified_local_nli_fails_closed_on_manifest_file_hash_mismatch(tmp_path: Path) -> None:
    manifest, plan = _write_verified_model(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NliModelIntegrityError, match="sha256 mismatch"):
        LocalOnnxNliScorer(
            manifest_path=manifest,
            model_path=tmp_path,
            experiment_plan_path=plan,
            tokenizer_factory=lambda _path: FakeTokenizer(),
            session_factory=lambda _path: FakeSession([0.0, 0.0, 0.0]),
        )


def test_verified_local_nli_rejects_invalid_logits_shape(tmp_path: Path) -> None:
    manifest, plan = _write_verified_model(tmp_path)

    class BadSession(FakeSession):
        def run(self, output_names: object, inputs: dict[str, object]) -> list[object]:
            del output_names, inputs
            return [[0.0, 1.0]]

    scorer = LocalOnnxNliScorer(
        manifest_path=manifest,
        model_path=tmp_path,
        experiment_plan_path=plan,
        tokenizer_factory=lambda _path: FakeTokenizer(),
        session_factory=lambda _path: BadSession([0.0, 0.0, 0.0]),
    )

    with pytest.raises(NliModelIntegrityError, match="logits"):
        scorer.score(passage="p", hypothesis="h")


def _ranked(text: str) -> list[RankedChunk]:
    chunk = EvidenceChunk(
        id="chunk",
        document_id="doc",
        document_name="doc.md",
        page=1,
        section=None,
        text=text,
        embedding=[0.0] * 384,
    )
    return [RankedChunk(chunk, 0.9, 0.9, 0.9, 0.0)]


class FixedScorer:
    def __init__(self, scores: NliScores) -> None:
        self.scores = scores

    def score_question_hypothesis(
        self, *, question: str, hypothesis: str, passage: str
    ) -> NliScores:
        del question, hypothesis, passage
        return self.scores


def _nli_provider(scores: NliScores) -> NliGroundedAnswerProvider:
    embeddings = DeterministicEmbeddingProvider(dimension=384)
    base = DeterministicGroundedAnswerProvider(
        embeddings=embeddings,
        config=DecisionConfig(0.0, 0.0, 0.08),
    )
    return NliGroundedAnswerProvider(
        base=base,
        scorer=FixedScorer(scores),
        config=NliDecisionConfig(entailment_threshold=0.62, contradiction_threshold=0.55),
        mode="test-nli-v3",
    )


def test_nli_gate_preserves_entailed_grounded_answer() -> None:
    result = _nli_provider(NliScores(0.01, 0.04, 0.95)).answer(
        "When does service begin?",
        _ranked("The service begins on 14 October 2031."),
    )

    assert result.status == "answered"
    assert result.answer == "14 October 2031"
    assert result.confidence == 0.95


def test_nli_gate_marks_neutral_candidate_as_only_partially_supported() -> None:
    result = _nli_provider(NliScores(0.05, 0.75, 0.20)).answer(
        "When does service begin?",
        _ranked("The service begins on 14 October 2031."),
    )

    assert result.status == "partially_supported"
    assert result.answerable is False
    assert result.citations
