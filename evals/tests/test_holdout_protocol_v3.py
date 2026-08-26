import json
from pathlib import Path

import pytest


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
        "evaluate_manifest",
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
        events.append("inference")
        return {
            "schema_version": "evaluation-result-v1",
            "cases": [],
            "extraction_evaluation": [],
        }

    monkeypatch.setattr(holdout, "preflight_holdout", preflight)
    monkeypatch.setattr(holdout, "claim_one_shot", claim)
    monkeypatch.setattr(holdout, "evaluate_manifest", evaluate)
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
    assert result["schema_version"] == "evidencedesk-holdout-raw-v3"
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == (
        "evidencedesk-holdout-raw-v3"
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

