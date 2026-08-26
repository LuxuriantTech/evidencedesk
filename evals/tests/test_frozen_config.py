import hashlib
import importlib.metadata
import json
from pathlib import Path

from evidencedesk_api.retrieval import (
    DEFAULT_RETRIEVAL_LIMIT,
    NAMED_ENTITY_DOCUMENT_BOOST,
    RRF_K,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_config_matches_code_and_immutable_development_inputs() -> None:
    config = json.loads(
        (ROOT / "evals/configs/semantic-v2-frozen.json").read_text(encoding="utf-8")
    )

    assert config["retrieval_method"] == "hybrid"
    assert config["embedding"]["fastembed_version"] == importlib.metadata.version("fastembed")
    assert config["embedding"]["onnxruntime_version"] == importlib.metadata.version("onnxruntime")
    assert config["answer_candidate_limit"] == DEFAULT_RETRIEVAL_LIMIT
    assert config["rank_fusion"]["rrf_k"] == RRF_K
    assert config["rank_fusion"]["named_entity_document_boost"] == NAMED_ENTITY_DOCUMENT_BOOST
    assert config["development_manifest_sha256"] == _sha256(
        ROOT / "datasets/development_v2/evaluation_cases.json"
    )
    assert config["development_corpus_sha256"] == _sha256(
        ROOT / "datasets/development_v2/corpus_manifest.json"
    )
    assert config["development_comparison_sha256"] == _sha256(
        ROOT / "artifacts/evaluations/development_v2_comparison/iteration-07-final.json"
    )
    assert config["selected_evaluation_artifact_sha256"] == _sha256(
        ROOT / "artifacts/evaluations/development-v2-semantic-v2-frozen-final.json"
    )
    assert config["development_recalculation_sha256"] == _sha256(
        ROOT / "artifacts/evaluations/development-v2-semantic-v2-recalculated-final.json"
    )
