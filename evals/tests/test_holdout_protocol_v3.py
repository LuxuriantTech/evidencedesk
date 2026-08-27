import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_runner_python_api_cannot_bypass_attested_holdout_preflight(tmp_path: Path) -> None:
    from evals.runner import EvaluationError, evaluate_manifest

    assert "_holdout_capability" not in inspect.signature(evaluate_manifest).parameters
    with pytest.raises(EvaluationError, match="attested holdout runner"):
        evaluate_manifest(
            tmp_path / "manifest.json",
            tmp_path / "corpus.json",
            split="holdout",
        )


def test_frozen_answer_runtime_accepts_only_the_declared_v3_engines() -> None:
    from evidencedesk_api.providers import DeterministicEmbeddingProvider

    from evals.answer_v3_runtime import AnswerRuntimeConfigError, build_frozen_answer_runtime

    config = {
        "answer_engine": {
            "strategy_id": "deterministic-evidence-v3",
            "configuration_id": "deterministic-v3-a",
            "support_threshold": 0.48,
            "partial_support_threshold": 0.38,
            "contradiction_margin": 0.08,
        },
        "extraction_engine": {"strategy_id": "supplier-extraction-v3"},
    }
    runtime = build_frozen_answer_runtime(
        config,
        DeterministicEmbeddingProvider(dimension=384),
    )

    assert runtime.answer_provider.mode == "deterministic-evidence-v3"
    assert runtime.answer_provider.config.support_threshold == 0.48
    config_b = {
        **config,
        "answer_engine": {
            "strategy_id": "deterministic-evidence-v3",
            "configuration_id": "deterministic-v3-b",
            "support_threshold": 0.56,
            "partial_support_threshold": 0.42,
            "contradiction_margin": 0.12,
        },
    }
    runtime_b = build_frozen_answer_runtime(
        config_b,
        DeterministicEmbeddingProvider(dimension=384),
    )
    assert runtime_b.answer_provider.config.support_threshold == 0.56
    invalid = {**config, "answer_engine": {"strategy_id": "unregistered"}}
    with pytest.raises(AnswerRuntimeConfigError, match="unsupported"):
        build_frozen_answer_runtime(invalid, DeterministicEmbeddingProvider(dimension=384))

    changed_threshold = {
        **config,
        "answer_engine": {**config["answer_engine"], "support_threshold": 0.47},
    }
    with pytest.raises(AnswerRuntimeConfigError, match="thresholds"):
        build_frozen_answer_runtime(
            changed_threshold,
            DeterministicEmbeddingProvider(dimension=384),
        )


def test_preflight_failure_does_not_consume_blind_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import holdout
    from evals.runner import EvaluationError

    lock = tmp_path / "holdout.lock"
    output = tmp_path / "holdout.json"

    def fail_preflight(**_kwargs: object) -> object:
        raise EvaluationError("preflight failed: attestation schema is invalid")

    monkeypatch.setattr(holdout, "preflight_holdout", fail_preflight)
    monkeypatch.setattr(
        holdout,
        "_evaluate_attested_holdout_manifest",
        lambda *_args, **_kwargs: pytest.fail("inference started before preflight passed"),
    )

    with pytest.raises(EvaluationError, match="attestation schema is invalid"):
        holdout.run_holdout_once(
            manifest_path=tmp_path / "evaluation.json",
            corpus_path=tmp_path / "corpus.json",
            attestation_path=tmp_path / "attestation.json",
            freeze_path=tmp_path / "freeze.json",
            config_path=tmp_path / "config.json",
            model_manifest_path=tmp_path / "model.json",
            model_path=tmp_path / "model",
            lock_path=lock,
            output_path=output,
            allow_holdout=True,
        )

    assert not lock.exists()
    assert not output.exists()


def test_real_preflight_error_classes_never_consume_lock_or_start_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import freeze as freeze_module
    from evals import holdout
    from evals.runner import EvaluationError

    valid_config = json.loads(
        (ROOT / "evals/configs/answer-v3-frozen-v7.json").read_text(encoding="utf-8")
    )
    valid_config["engine_fingerprint_at_selection"] = holdout._engine_fingerprint()
    valid_corpus = {
        "dataset_version": "preflight-test-v1",
        "synthetic_only": True,
        "documents": [
            {"id": "doc-1", "filename": "doc-1.md", "pages": ["Evidence."]}
        ],
    }
    base_manifest = {
        "dataset_version": "preflight-test-v1",
        "parameters_version": "answer-v3",
        "mode": "grounded-local-v3",
        "seed": 1,
        "metrics": {},
        "cases": [],
        "extraction_targets": [],
    }
    missing_filename = {
        **valid_corpus,
        "documents": [{"id": "doc-1", "pages": ["Evidence."]}],
    }
    wrong_dataset = {**base_manifest, "dataset_version": "another-version"}
    malformed_gold = {
        **base_manifest,
        "cases": [
            {
                "id": "case-1",
                "split": "holdout",
                "kind": "answerable",
                "question": "What is the evidence?",
                "expected_answer": "Evidence",
                "expected_citations": ["not-an-object"],
            }
        ],
    }
    failures: list[
        tuple[str, object, dict[str, object], dict[str, object], dict[str, object], str]
    ] = [
        (
            "schema",
            ["not-an-object"],
            valid_corpus,
            {"freeze_commit": "f" * 40},
            valid_config,
            r"preflight dataset schema failed",
        ),
        (
            "attestation",
            base_manifest,
            valid_corpus,
            {},
            valid_config,
            r"preflight attestation schema failed",
        ),
        (
            "filename",
            base_manifest,
            missing_filename,
            {"freeze_commit": "f" * 40},
            valid_config,
            r"preflight dataset and gold validation failed:.*filename",
        ),
        (
            "dataset",
            wrong_dataset,
            valid_corpus,
            {"freeze_commit": "f" * 40},
            valid_config,
            r"preflight dataset and gold validation failed:.*versions",
        ),
        (
            "gold",
            malformed_gold,
            valid_corpus,
            {"freeze_commit": "f" * 40},
            valid_config,
            r"preflight dataset and gold validation failed:.*citation.*object",
        ),
        (
            "configuration",
            base_manifest,
            valid_corpus,
            {"freeze_commit": "f" * 40},
            {},
            r"preflight configuration schema failed",
        ),
    ]
    monkeypatch.setattr(holdout, "verify_committed_holdout_inputs", lambda *_a, **_k: None)
    monkeypatch.setattr(holdout, "verify_freeze_attestation", lambda *_a, **_k: None)
    monkeypatch.setattr(holdout, "_verify_engine_commit", lambda *_a, **_k: None)
    monkeypatch.setattr(
        freeze_module,
        "_validate_development_evidence",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        holdout,
        "_evaluate_attested_holdout_manifest",
        lambda *_args, **_kwargs: pytest.fail("inference started before preflight passed"),
    )

    for label, manifest, corpus, attestation, config, message in failures:
        manifest_path = tmp_path / f"{label}-manifest.json"
        corpus_path = tmp_path / f"{label}-corpus.json"
        attestation_path = tmp_path / f"{label}-attestation.json"
        config_path = tmp_path / f"{label}-config.json"
        freeze_path = tmp_path / f"{label}-freeze.json"
        model_manifest_path = tmp_path / f"{label}-model.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
        attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        freeze_path.write_text(json.dumps({"engine_commit": "e" * 40}), encoding="utf-8")
        model_manifest_path.write_text("{}", encoding="utf-8")
        lock = tmp_path / f"{label}.lock"
        output = tmp_path / f"{label}.json"
        with pytest.raises(EvaluationError, match=message):
            holdout.run_holdout_once(
                manifest_path=manifest_path,
                corpus_path=corpus_path,
                attestation_path=attestation_path,
                freeze_path=freeze_path,
                config_path=config_path,
                model_manifest_path=model_manifest_path,
                model_path=tmp_path / "unused-model",
                lock_path=lock,
                output_path=output,
                allow_holdout=True,
            )
        assert not lock.exists()
        assert not output.exists()


def test_successful_preflight_is_completed_before_lock_and_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import holdout

    events: list[str] = []
    lock = tmp_path / "holdout.lock"
    output = tmp_path / "holdout.json"

    prepared = holdout.HoldoutPreflight(
        config={"retrieval_method": "hybrid"},
        manifest={"dataset_version": "blind-v7"},
        freeze={"engine_commit": "a" * 40},
        attestation={"freeze_commit": "b" * 40},
        evidence={"dataset_commit": "c" * 40},
        report=holdout.ValidationReport(
            total_cases=1,
            answerable_cases=1,
            unanswerable_cases=0,
            adversarial_or_ambiguous_cases=0,
            development_cases=0,
            holdout_cases=1,
            document_ids_are_known=True,
            citations_are_exact=True,
            extraction_expectations_are_traceable=True,
            synthetic_only=True,
        ),
        provider=object(),
        method=holdout.RetrievalMethod.HYBRID,
        runtime_metadata={},
        answer_provider="prepared-answer-provider",
        extraction_provider="prepared-extraction-provider",
    )

    def preflight(**_kwargs: object) -> object:
        assert not lock.exists()
        events.append("preflight")
        return prepared

    def claim(*_args: object, **_kwargs: object) -> None:
        events.append("claim")
        lock.write_text("{}", encoding="utf-8")

    def evaluate(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert lock.exists()
        assert _kwargs["answer_provider"] == "prepared-answer-provider"
        assert _kwargs["extraction_provider"] == "prepared-extraction-provider"
        events.append("inference")
        return {
            "schema_version": "evaluation-result-v1",
            "cases": [],
            "extraction_evaluation": [],
        }

    monkeypatch.setattr(holdout, "preflight_holdout", preflight)
    monkeypatch.setattr(holdout, "claim_one_shot", claim)
    monkeypatch.setattr(holdout, "_evaluate_attested_holdout_manifest", evaluate)
    monkeypatch.setattr(holdout, "_sha256", lambda _path: "d" * 64)

    result = holdout.run_holdout_once(
        manifest_path=tmp_path / "evaluation.json",
        corpus_path=tmp_path / "corpus.json",
        attestation_path=tmp_path / "attestation.json",
        freeze_path=tmp_path / "freeze.json",
        config_path=tmp_path / "config.json",
        model_manifest_path=tmp_path / "model.json",
        model_path=tmp_path / "model",
        lock_path=lock,
        output_path=output,
        allow_holdout=True,
    )

    assert events == ["preflight", "claim", "inference"]
    assert result["schema_version"] == "evidencedesk-holdout-raw-v4"
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == (
        "evidencedesk-holdout-raw-v4"
    )


def test_runner_cli_cannot_bypass_attested_holdout_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evals import runner
    from evals.runner import EvaluationError

    monkeypatch.setattr(
        "sys.argv",
        ["runner", "--split", "holdout", "--allow-holdout"],
    )

    with pytest.raises(EvaluationError, match="attested holdout runner"):
        runner.main()


def test_v3_engine_fingerprint_covers_answer_runtime_and_model_manifests() -> None:
    from evals.runner import ENGINE_FINGERPRINT_PATHS

    assert {
        "evals/answer_v3_runtime.py",
        "evals/finalize_answer_v3.py",
        "evals/freeze.py",
        "evals/strategy_benchmark_v3.py",
        "apps/api/evidencedesk_api/answering.py",
        "apps/api/evidencedesk_api/extraction.py",
        "apps/api/evidencedesk_api/nli_answering.py",
        "apps/api/evidencedesk_api/qwen_answering.py",
        "apps/api/evidencedesk_api/redaction.py",
        "infra/models/paraphrase-multilingual-minilm-l12-v2.json",
        "infra/models/mdeberta-v3-base-mnli-xnli.json",
        "infra/models/qwen2.5-0.5b-instruct-q4-k-m.json",
    }.issubset(ENGINE_FINGERPRINT_PATHS)


@pytest.mark.parametrize(
    ("field", "weakened"),
    [
        ("execution_limit", 2),
        ("independent_author_required", False),
        ("preflight_must_pass_before_lock", False),
    ],
)
def test_holdout_preflight_rejects_a_weakened_frozen_protocol(
    field: str,
    weakened: object,
) -> None:
    from evals.holdout import ValidationReport, verify_holdout_protocol
    from evals.runner import EvaluationError

    config = json.loads(
        (ROOT / "evals/configs/answer-v3-frozen-v7.json").read_text(encoding="utf-8")
    )
    config["holdout_protocol"][field] = weakened
    parameters_version = config["parameters_version"]
    seed = config["holdout_protocol"]["seed"]
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

    with pytest.raises(EvaluationError, match="holdout protocol"):
        verify_holdout_protocol(
            config,
            {"parameters_version": parameters_version, "seed": seed},
            {"parameters_version": parameters_version, "seed": seed},
            report,
        )


def test_holdout_protocol_binds_dataset_mode_to_frozen_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evals import holdout
    from evals.runner import EvaluationError

    config = json.loads(
        (ROOT / "evals/configs/answer-v3-frozen-v7.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(holdout, "_validate_v3_config_data", lambda value: value)
    protocol = config["holdout_protocol"]
    report = holdout.ValidationReport(
        total_cases=protocol["total_cases"],
        answerable_cases=protocol["minimum_answerable"],
        unanswerable_cases=protocol["minimum_unanswerable"],
        adversarial_or_ambiguous_cases=protocol[
            "minimum_ambiguous_or_adversarial"
        ],
        development_cases=0,
        holdout_cases=protocol["total_cases"],
        document_ids_are_known=True,
        citations_are_exact=True,
        extraction_expectations_are_traceable=True,
        synthetic_only=True,
    )
    manifest = {
        "parameters_version": config["parameters_version"],
        "seed": protocol["seed"],
        "mode": "extractive-local-onnx",
    }
    attestation = {
        "parameters_version": config["parameters_version"],
        "seed": protocol["seed"],
    }

    with pytest.raises(EvaluationError, match="mode"):
        holdout.verify_holdout_protocol(config, manifest, attestation, report)

    manifest["mode"] = config["mode"]
    manifest["metrics"] = {"citation_match": "weakened"}
    with pytest.raises(EvaluationError, match="metric"):
        holdout.verify_holdout_protocol(config, manifest, attestation, report)
