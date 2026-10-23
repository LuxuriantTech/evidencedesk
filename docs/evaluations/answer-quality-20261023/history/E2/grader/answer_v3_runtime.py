"""Fail-closed construction of the single frozen answer/extraction v3 runtime."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from evidencedesk_api.answering import DecisionConfig, DeterministicGroundedAnswerProvider
from evidencedesk_api.extraction import SupplierExtraction, extract_supplier_fields_v3
from evidencedesk_api.providers import EmbeddingProvider
from evidencedesk_api.retrieval import EvidenceChunk

from evals.runner import AnswerProviderLike


class AnswerRuntimeConfigError(ValueError):
    """The frozen configuration does not name exactly the pre-registered v3 runtime."""


@dataclass(frozen=True, slots=True)
class FrozenAnswerRuntime:
    answer_provider: AnswerProviderLike
    extraction_provider: Callable[[list[EvidenceChunk]], SupplierExtraction]


_ANSWER_KEYS = frozenset(
    {
        "strategy_id",
        "configuration_id",
        "support_threshold",
        "partial_support_threshold",
        "contradiction_margin",
    }
)
_EXTRACTION_KEYS = frozenset({"strategy_id"})
_REGISTERED_DETERMINISTIC_CONFIGURATIONS = {
    "deterministic-v3-a": {
        "support_threshold": 0.48,
        "partial_support_threshold": 0.38,
        "contradiction_margin": 0.08,
    },
    "deterministic-v3-b": {
        "support_threshold": 0.56,
        "partial_support_threshold": 0.42,
        "contradiction_margin": 0.12,
    },
}


def _frozen_number(value: object, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AnswerRuntimeConfigError(f"answer_engine.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise AnswerRuntimeConfigError(f"answer_engine.{key} must be between zero and one")
    return number


def _object(config: dict[str, Any], key: str, required: frozenset[str]) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise AnswerRuntimeConfigError(f"frozen {key} is missing")
    if frozenset(value) != required:
        raise AnswerRuntimeConfigError(f"frozen {key} has unsupported or missing fields")
    return value


def build_frozen_answer_runtime(
    config: dict[str, Any],
    embedding: EmbeddingProvider,
) -> FrozenAnswerRuntime:
    """Build only the explicit v3 engine recorded in the final frozen config."""

    decision = validate_frozen_answer_config(config)

    def extract(chunks: list[EvidenceChunk]) -> SupplierExtraction:
        return extract_supplier_fields_v3(chunks, embeddings=embedding)

    return FrozenAnswerRuntime(
        answer_provider=cast(
            AnswerProviderLike,
            DeterministicGroundedAnswerProvider(embeddings=embedding, config=decision),
        ),
        extraction_provider=extract,
    )


def validate_frozen_answer_config(config: dict[str, Any]) -> DecisionConfig:
    """Validate the selected engine identity without loading model weights."""

    answer = _object(config, "answer_engine", _ANSWER_KEYS)
    extraction = _object(config, "extraction_engine", _EXTRACTION_KEYS)
    if answer["strategy_id"] != "deterministic-evidence-v3":
        raise AnswerRuntimeConfigError("unsupported frozen answer strategy")
    configuration_id = answer["configuration_id"]
    if not isinstance(configuration_id, str) or (
        configuration_id not in _REGISTERED_DETERMINISTIC_CONFIGURATIONS
    ):
        raise AnswerRuntimeConfigError("unsupported frozen answer configuration")
    if extraction["strategy_id"] != "supplier-extraction-v3":
        raise AnswerRuntimeConfigError("unsupported frozen extraction strategy")
    expected_thresholds = _REGISTERED_DETERMINISTIC_CONFIGURATIONS[configuration_id]
    observed_thresholds = {
        key: _frozen_number(answer[key], key) for key in expected_thresholds
    }
    if observed_thresholds != expected_thresholds:
        raise AnswerRuntimeConfigError(
            f"frozen answer thresholds differ from {configuration_id}"
        )
    return DecisionConfig(**observed_thresholds)
