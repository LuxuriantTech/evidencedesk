import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from evidencedesk_api.retrieval import (
    DEFAULT_RETRIEVAL_LIMIT,
    NAMED_ENTITY_DOCUMENT_BOOST,
    RRF_K,
)

from evals.freeze import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FREEZE_PATH,
    _assert_paths_tracked_and_clean,
    _development_target_gate,
    _validate_comparison_raw,
    _validate_strategy_comparison,
    _validate_v3_freeze_config,
)
from evals.runner import _engine_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def test_v3_freeze_defaults_are_generic_and_schema_validation_is_fail_closed(
    tmp_path: Path,
) -> None:
    assert DEFAULT_CONFIG_PATH.name == "answer-v3-frozen-v7.json"
    assert DEFAULT_FREEZE_PATH.name == "answer-v3-freeze-v7.json"
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    assert _validate_v3_freeze_config(path)["parameters_version"] == config[
        "parameters_version"
    ]
    config.pop("answer_engine")
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported or missing top-level fields"):
        _validate_v3_freeze_config(path)


@pytest.mark.parametrize(
    ("field", "weakened"),
    [
        ("execution_limit", 2),
        ("independent_author_required", False),
        ("preflight_must_pass_before_lock", False),
        ("protocol_label", "holdout-v8-unregistered"),
    ],
)
def test_v3_freeze_rejects_weakened_holdout_protocol(
    tmp_path: Path,
    field: str,
    weakened: object,
) -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["holdout_protocol"][field] = weakened
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="holdout protocol"):
        _validate_v3_freeze_config(path)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("frozen_at", "not-a-date"),
        ("development_dataset_version", "fabricated"),
        ("mode", "extractive-local-onnx"),
    ],
)
def test_v3_freeze_rejects_falsified_config_identity(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config[field] = invalid
    path = tmp_path / "invalid-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError):
        _validate_v3_freeze_config(path)


def test_v3_freeze_rejects_falsified_metric_definition(tmp_path: Path) -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["evaluation"]["citation_match"] = "fabricated"
    path = tmp_path / "invalid-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation protocol"):
        _validate_v3_freeze_config(path)


def test_v3_freeze_rejects_a_development_hash_that_does_not_match(
    tmp_path: Path,
) -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["development_runtime_summary_sha256"] = "0" * 64
    path = tmp_path / "invalid-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="development artifact hash mismatch"):
        _validate_v3_freeze_config(path)


def test_freeze_provenance_requires_every_input_tracked_and_clean(tmp_path: Path) -> None:
    git = shutil.which("git")
    assert git is not None
    subprocess.run([git, "init", "-q"], cwd=tmp_path, check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [git, "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True
    )
    subprocess.run(  # noqa: S603
        [git, "config", "user.name", "EvidenceDesk test"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "tracked.json"
    tracked.write_text("{}\n", encoding="utf-8")
    subprocess.run(  # noqa: S603
        [git, "add", "tracked.json"], cwd=tmp_path, check=True
    )
    subprocess.run(  # noqa: S603
        [git, "commit", "-qm", "fixture"], cwd=tmp_path, check=True
    )

    _assert_paths_tracked_and_clean(tmp_path, ["tracked.json"])
    untracked = tmp_path / "untracked.json"
    untracked.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not tracked at HEAD"):
        _assert_paths_tracked_and_clean(tmp_path, ["untracked.json"])
    tracked.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="committed and clean"):
        _assert_paths_tracked_and_clean(tmp_path, ["tracked.json"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_target_gate_is_derived_from_all_frozen_targets() -> None:
    passing = {
        "citation_precision": 0.9,
        "citation_recall": 0.9,
        "citation_case_accuracy": 0.9,
        "abstention_accuracy": 0.85,
        "extraction_f1": 0.9,
        "error_rate": 0.0,
        "schema_error_rate": 0.0,
    }

    assert _development_target_gate(passing)["verdict"] == "PASS"
    failed = _development_target_gate(
        {**passing, "citation_recall": 0.8, "abstention_accuracy": 0.8}
    )
    assert failed["verdict"] == "FAIL"
    assert set(failed["failed_targets"]) == {
        "citation_recall",
        "abstention_accuracy",
    }


def test_comparison_raw_is_bound_to_engine_dataset_partition_and_configuration() -> None:
    path = (
        ROOT
        / "artifacts/evaluations/development_v3/strategy_v3/"
        "deterministic-v3-b-selection-r1.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    benchmark = raw["benchmark"]
    expected = {
        "expected_fingerprint": raw["engine_fingerprint"],
        "expected_manifest_sha256": raw["manifest_sha256"],
        "expected_corpus_sha256": raw["corpus_sha256"],
        "expected_configuration": benchmark["configuration"],
        "expected_partition": "selection",
        "expected_repetition": 1,
    }

    _validate_comparison_raw(raw, **expected)
    with pytest.raises(ValueError, match="engine_fingerprint"):
        _validate_comparison_raw(raw, **{**expected, "expected_fingerprint": "0" * 64})
    forged = json.loads(json.dumps(raw))
    forged["citation_precision"] = 0.0
    with pytest.raises(ValueError, match="metric mismatch"):
        _validate_comparison_raw(forged, **expected)
    invalid_timing = json.loads(json.dumps(raw))
    invalid_timing["benchmark"]["total_wall_ms"] = 0.0
    with pytest.raises(ValueError, match="timing"):
        _validate_comparison_raw(invalid_timing, **expected)
    invalid_rss = json.loads(json.dumps(raw))
    invalid_rss["benchmark"]["peak_rss_bytes"] = 0
    with pytest.raises(ValueError, match="RSS"):
        _validate_comparison_raw(invalid_rss, **expected)


def test_strategy_comparison_is_complete_and_bound_before_finalization() -> None:
    path = ROOT / "artifacts/evaluations/development_v3/strategy_v3/comparison.json"
    comparison = json.loads(path.read_text(encoding="utf-8"))
    first_raw_path = ROOT / comparison["comparison"]["deterministic-v3-a"][
        "calibration"
    ]["artifacts"][0]["path"]
    comparison.setdefault(
        "engine_fingerprint",
        json.loads(first_raw_path.read_text(encoding="utf-8"))["engine_fingerprint"],
    )

    entries, evidence_paths = _validate_strategy_comparison(
        comparison,
        root=ROOT,
        expected_fingerprint=comparison["engine_fingerprint"],
    )

    assert len(entries) == 6
    assert evidence_paths
    incomplete = json.loads(json.dumps(comparison))
    incomplete["comparison"].pop("qwen-json-v3-b")
    with pytest.raises(ValueError, match="incomplete"):
        _validate_strategy_comparison(
            incomplete,
            root=ROOT,
            expected_fingerprint=comparison["engine_fingerprint"],
        )


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


def test_v6_recovery_config_binds_the_recomputed_development_artifacts() -> None:
    config = json.loads(
        (ROOT / "evals/configs/semantic-v2-frozen-v6.json").read_text(encoding="utf-8")
    )

    assert config["engine_fingerprint_at_selection"] == (
        "a8371922b582caccc2f4560544e3b1a8261570c0a0dbafffd55c8d1c86289206"
    )
    assert config["development_comparison_sha256"] == _sha256(
        ROOT / "artifacts/evaluations/development_v2_comparison/iteration-08-v6-final.json"
    )
    assert config["selected_evaluation_artifact_sha256"] == _sha256(
        ROOT / "artifacts/evaluations/development-v2-semantic-v2-frozen-v6.json"
    )
    assert config["development_recalculation_sha256"] == _sha256(
        ROOT / "artifacts/evaluations/development-v2-semantic-v2-recalculated-v6.json"
    )


def test_v7_answer_v3_config_binds_frozen_development_and_runtime() -> None:
    config_path = ROOT / "evals/configs/answer-v3-frozen-v7.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert _validate_v3_freeze_config(config_path) == config
    assert config["parameters_version"] == "grounded-local-v3.0-frozen-v7"
    assert config["mode"] == "grounded-local-v3"
    assert config["retrieval_method"] == "hybrid"
    assert config["answer_candidate_limit"] == DEFAULT_RETRIEVAL_LIMIT
    assert config["rank_fusion"] == {
        "algorithm": "reciprocal-rank-fusion",
        "rrf_k": RRF_K,
        "named_entity_document_boost": NAMED_ENTITY_DOCUMENT_BOOST,
    }
    assert config["answer_engine"] == {
        "strategy_id": "deterministic-evidence-v3",
        "configuration_id": "deterministic-v3-a",
        "support_threshold": 0.48,
        "partial_support_threshold": 0.38,
        "contradiction_margin": 0.08,
    }
    assert config["extraction_engine"] == {"strategy_id": "supplier-extraction-v3"}
    assert config["engine_fingerprint_at_selection"] == _engine_fingerprint()
    assert config["embedding"]["fastembed_version"] == importlib.metadata.version(
        "fastembed"
    )
    assert config["embedding"]["onnxruntime_version"] == importlib.metadata.version(
        "onnxruntime"
    )
    assert config["embedding"]["model_manifest_sha256"] == _sha256(
        ROOT / "infra/models/paraphrase-multilingual-minilm-l12-v2.json"
    )
    expected_hashes = {
        "development_manifest_sha256": "datasets/development_v3/evaluation_cases.json",
        "development_corpus_sha256": "datasets/development_v3/corpus_manifest.json",
        "development_generation_manifest_sha256": (
            "datasets/development_v3/generation_manifest.json"
        ),
        "experiment_plan_sha256": "evals/configs/answer-v3-experiment-plan.json",
        "development_comparison_sha256": (
            "artifacts/evaluations/development_v3/strategy_v3/comparison.json"
        ),
        "selected_evaluation_artifact_sha256": (
            "artifacts/evaluations/development_v3/"
            "frozen-runtime-v4-locked-final/selection-r1.json"
        ),
        "development_recalculation_sha256": (
            "artifacts/evaluations/development_v3/frozen-runtime-v4-locked-final/"
            "selection-r1-recalculated.json"
        ),
        "development_runtime_summary_sha256": (
            "artifacts/evaluations/development_v3/"
            "frozen-runtime-v4-locked-final/summary.json"
        ),
    }
    for key, relative in expected_hashes.items():
        assert config[key] == _sha256(ROOT / relative)

    protocol = config["holdout_protocol"]
    assert protocol["protocol_label"] == "holdout-v7-independent-answer-v3"
    assert protocol["execution_limit"] == 1
    assert protocol["preflight_must_pass_before_lock"] is True
    assert protocol["generation_after_engine_freeze"] is True
    assert protocol["independent_author_required"] is True
    assert protocol["gold_may_not_change_after_commit"] is True
