import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evals.holdout_v4 import (
    _verify_runtime_dependencies,
    claim_one_shot,
    run_holdout_once,
    verify_committed_holdout_inputs,
    verify_freeze_attestation,
    verify_holdout_protocol,
)
from evals.recalculate import recalculate_metrics
from evals.runner import ENGINE_FINGERPRINT_PATHS, EvaluationError
from evals.validate_dataset import ValidationReport


def test_one_shot_claim_requires_gate_and_refuses_existing_output_or_lock(tmp_path: Path) -> None:
    lock = tmp_path / "v4.lock"
    output = tmp_path / "v4.json"

    with pytest.raises(EvaluationError, match="explicitly authorized"):
        claim_one_shot(lock, output, allow_holdout=False, evidence={"dataset": "v4"})
    assert not lock.exists()

    output.write_text("already present", encoding="utf-8")
    with pytest.raises(EvaluationError, match="output already exists"):
        claim_one_shot(lock, output, allow_holdout=True, evidence={"dataset": "v4"})
    assert not lock.exists()

    output.unlink()
    claim_one_shot(lock, output, allow_holdout=True, evidence={"dataset": "v4"})
    assert lock.exists()

    with pytest.raises(EvaluationError, match="already opened"):
        claim_one_shot(lock, output, allow_holdout=True, evidence={"dataset": "v4"})


def test_second_holdout_attempt_refuses_before_hashing_any_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "v4.lock"
    lock.write_text("claimed", encoding="utf-8")
    monkeypatch.setattr(
        "evals.holdout_v4._sha256",
        lambda _path: pytest.fail("holdout input was read before the one-shot refusal"),
    )

    with pytest.raises(EvaluationError, match="already opened"):
        run_holdout_once(
            manifest_path=tmp_path / "missing-evaluation.json",
            corpus_path=tmp_path / "missing-corpus.json",
            attestation_path=tmp_path / "missing-attestation.json",
            freeze_path=tmp_path / "missing-freeze.json",
            config_path=tmp_path / "missing-config.json",
            model_manifest_path=tmp_path / "missing-model.json",
            model_path=tmp_path / "missing-model",
            lock_path=lock,
            output_path=tmp_path / "result.json",
            allow_holdout=True,
        )


def test_freeze_attestation_is_bound_to_engine_config_model_and_author() -> None:
    evidence = {
        "engine_fingerprint": "engine-sha",
        "config_sha256": "config-sha",
        "model_manifest_sha256": "model-sha",
        "manifest_sha256": "cases-sha",
        "corpus_sha256": "corpus-sha",
        "freeze_sha256": "freeze-sha",
        "dependency_sha256": {"pyproject.toml": "project-sha", "uv.lock": "lock-sha"},
    }
    freeze = {
        "engine_commit": "20380acd",
        "engine_fingerprint": "engine-sha",
        "config_sha256": "config-sha",
        "model_manifest_sha256": "model-sha",
        "engine_paths": list(ENGINE_FINGERPRINT_PATHS),
        "dependency_sha256": {"pyproject.toml": "project-sha", "uv.lock": "lock-sha"},
    }
    attestation = {
        **freeze,
        "author_role": "independent-holdout-author",
        "generated_after_engine_freeze": True,
        "synthetic_only": True,
        "freeze_sha256": "freeze-sha",
        "development_overlap_check": {
            "method": "sha256-normalized-pages-and-case-id",
            "matching_documents": 0,
            "matching_case_ids": 0,
        },
        "sha256": {
            "corpus_manifest.json": "corpus-sha",
            "evaluation_cases.json": "cases-sha",
        },
    }

    verify_freeze_attestation(attestation, freeze, evidence)
    for key in ("engine_fingerprint", "config_sha256", "model_manifest_sha256"):
        changed = dict(attestation)
        changed[key] = "different"
        with pytest.raises(EvaluationError, match=key.replace("_", " ")):
            verify_freeze_attestation(changed, freeze, evidence)


def test_holdout_protocol_binds_seed_parameters_counts_and_synthetic_status() -> None:
    config = {
        "parameters_version": "semantic-local-v2.0-frozen",
        "holdout_v4_protocol": {
            "seed": 20260827,
            "total_cases": 40,
            "minimum_answerable": 25,
            "minimum_unanswerable": 10,
            "minimum_ambiguous_or_adversarial": 5,
        },
    }
    manifest = {"parameters_version": "semantic-local-v2.0-frozen", "seed": 20260827}
    attestation = {"parameters_version": "semantic-local-v2.0-frozen", "seed": 20260827}
    report = ValidationReport(
        total_cases=40,
        answerable_cases=25,
        unanswerable_cases=10,
        adversarial_or_ambiguous_cases=5,
        development_cases=0,
        holdout_cases=40,
        document_ids_are_known=True,
        citations_are_exact=True,
        extraction_expectations_are_traceable=True,
        synthetic_only=True,
    )

    verify_holdout_protocol(config, manifest, attestation, report)
    manifest["seed"] = 7
    with pytest.raises(EvaluationError, match="seed does not match"):
        verify_holdout_protocol(config, manifest, attestation, report)


def test_runtime_dependencies_must_match_frozen_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "embedding": {
            "fastembed_version": "0.8.0",
            "onnxruntime_version": "1.29.0",
        }
    }
    installed = {"fastembed": "0.8.0", "onnxruntime": "1.29.0"}
    monkeypatch.setattr(
        "evals.holdout_v4.importlib.metadata.version",
        lambda package: installed[package],
    )

    assert _verify_runtime_dependencies(config) == {
        "fastembed_version": "0.8.0",
        "onnxruntime_version": "1.29.0",
    }
    installed["onnxruntime"] = "1.30.0"
    with pytest.raises(EvaluationError, match="onnxruntime version"):
        _verify_runtime_dependencies(config)


def _git(repo: Path, *args: str) -> str:
    git = shutil.which("git")
    assert git is not None
    completed = subprocess.run(  # noqa: S603 - test-only arguments are fixed below
        [git, *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_holdout_inputs_must_be_committed_after_freeze(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "EvidenceDesk test")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    _git(tmp_path, "add", "freeze.json")
    _git(tmp_path, "commit", "-m", "freeze engine")
    freeze_commit = _git(tmp_path, "rev-parse", "HEAD")

    inputs = []
    for name in ("manifest.json", "corpus.json", "attestation.json"):
        path = tmp_path / name
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        inputs.append(path)
    _git(tmp_path, "add", *[path.name for path in inputs])
    _git(tmp_path, "commit", "-m", "add independent holdout")
    dataset_commit = _git(tmp_path, "rev-parse", "HEAD")

    verify_committed_holdout_inputs(
        (*inputs, freeze),
        new_holdout_paths=tuple(inputs),
        dataset_commit=dataset_commit,
        freeze_commit=freeze_commit,
        freeze_path=freeze,
        expected_freeze_sha256=__import__("hashlib").sha256(freeze.read_bytes()).hexdigest(),
        root=tmp_path,
    )

    inputs[0].write_text("modified after commit", encoding="utf-8")
    with pytest.raises(EvaluationError, match="committed and clean"):
        verify_committed_holdout_inputs(
            (*inputs, freeze),
            new_holdout_paths=tuple(inputs),
            dataset_commit=dataset_commit,
            freeze_commit=freeze_commit,
            freeze_path=freeze,
            expected_freeze_sha256=__import__("hashlib").sha256(freeze.read_bytes()).hexdigest(),
            root=tmp_path,
        )

    inputs[0].write_text(json.dumps({"name": inputs[0].name}), encoding="utf-8")
    untracked = tmp_path / "untracked.json"
    untracked.write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluationError, match="committed and clean"):
        verify_committed_holdout_inputs(
            (*inputs, freeze, untracked),
            new_holdout_paths=(*inputs, untracked),
            dataset_commit=dataset_commit,
            freeze_commit=freeze_commit,
            freeze_path=freeze,
            expected_freeze_sha256=__import__("hashlib").sha256(freeze.read_bytes()).hexdigest(),
            root=tmp_path,
        )


def test_freeze_commit_must_strictly_precede_dataset_commit(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "EvidenceDesk test")
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}", encoding="utf-8")
    _git(tmp_path, "add", "freeze.json")
    _git(tmp_path, "commit", "-m", "freeze and holdout together")
    same_commit = _git(tmp_path, "rev-parse", "HEAD")

    with pytest.raises(EvaluationError, match="strictly precede"):
        verify_committed_holdout_inputs(
            (freeze,),
            new_holdout_paths=(),
            dataset_commit=same_commit,
            freeze_commit=same_commit,
            freeze_path=freeze,
            expected_freeze_sha256=__import__("hashlib").sha256(freeze.read_bytes()).hexdigest(),
            root=tmp_path,
        )


def test_holdout_gold_must_not_exist_in_freeze_commit(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "EvidenceDesk test")
    freeze = tmp_path / "freeze.json"
    manifest = tmp_path / "manifest.json"
    corpus = tmp_path / "corpus.json"
    attestation = tmp_path / "attestation.json"
    for path in (freeze, manifest, corpus, attestation):
        path.write_text("{}", encoding="utf-8")
    _git(tmp_path, "add", "freeze.json", "manifest.json", "corpus.json", "attestation.json")
    _git(tmp_path, "commit", "-m", "invalid freeze containing gold")
    freeze_commit = _git(tmp_path, "rev-parse", "HEAD")
    marker = tmp_path / "marker.txt"
    marker.write_text("later commit", encoding="utf-8")
    _git(tmp_path, "add", "marker.txt")
    _git(tmp_path, "commit", "-m", "neutral later commit")
    dataset_commit = _git(tmp_path, "rev-parse", "HEAD")

    with pytest.raises(EvaluationError, match="already existed at freeze commit"):
        verify_committed_holdout_inputs(
            (manifest, corpus, attestation, freeze),
            new_holdout_paths=(manifest, corpus, attestation),
            dataset_commit=dataset_commit,
            freeze_commit=freeze_commit,
            freeze_path=freeze,
            expected_freeze_sha256=__import__("hashlib").sha256(freeze.read_bytes()).hexdigest(),
            root=tmp_path,
        )


def test_recalculation_uses_only_raw_case_and_extraction_decisions() -> None:
    raw = {
        "cases": [
            {
                "kind": "answerable",
                "status": "answered",
                "answer_match": True,
                "citation_matches": [True],
            },
            {
                "kind": "answerable",
                "status": "answered",
                "answer_match": False,
                "citation_matches": [True, False],
            },
            {
                "kind": "unanswerable",
                "status": "abstained",
                "answer_match": False,
                "citation_matches": [],
            },
            {
                "kind": "ambiguous",
                "status": "ambiguous",
                "answer_match": False,
                "citation_matches": [],
            },
        ],
        "extraction_evaluation": [
            {"predicted": 2, "gold": 2, "true_positive": 1},
            {"predicted": 1, "gold": 1, "true_positive": 1},
        ],
    }

    recalculated = recalculate_metrics(raw)

    assert recalculated["metric_counts"] == {
        "citation_correct": 2,
        "citation_returned": 3,
        "citation_expected": 0,
        "citation_expected_matched": 0,
        "answerable_correct": 1,
        "answerable_total": 2,
        "retrieval_hits_at_5": 0,
        "abstention_correct": 2,
        "abstention_total": 2,
        "extraction_true_positive": 2,
        "extraction_predicted": 3,
        "extraction_gold": 3,
        "errors": 0,
    }
    assert recalculated["citation_precision"] == pytest.approx(2 / 3)
    assert recalculated["citation_case_accuracy"] == 0.5
    assert recalculated["abstention_accuracy"] == 1.0
    assert recalculated["extraction_f1"] == pytest.approx(2 / 3)
