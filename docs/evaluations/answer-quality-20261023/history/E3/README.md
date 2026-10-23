# E3: third synthetic comparison

- [Frozen protocol](PROTOCOL.md), [rules](COMPARISON_RULES_FROZEN.json), [freeze attestation](FREEZE_ATTESTATION.json), and [source input hashes](SHA256SUMS.json)
- [Frozen questions](datasets/evaluation_cases.json), [corpus manifest](datasets/corpus_manifest.json), and source documents in `datasets/sources/`
- [Run adapter](adapters/run_baseline.py) and the shared [runner](shared-toolchain/grader/runner.py), [schema rules](shared-toolchain/grader/schema_versions.py), [dataset validator](shared-toolchain/grader/validate_dataset.py), [answer runtime](shared-toolchain/grader/answer_v3_runtime.py), [answer config](shared-toolchain/grader/evals/configs/answer-v3-frozen-v7.json), [model manifest](shared-toolchain/config/model-manifest.json), and [locked dependencies](shared-toolchain/config/uv.lock)
- [Machine comparison record](COMPARISON_REPORT.machine.json) and [run provenance](RUN_PROVENANCE.md)
- [Recovered baseline raw result](baseline-recovery/raw_results.json) and [redacted stdout copy](baseline-recovery/runner_stdout.redacted.txt)
- [First and only candidate raw result](candidate-first-run/raw_results.json) and [redacted stdout copy](candidate-first-run/runner_stdout.redacted.txt)

The first baseline raw JSON was overwritten by the first candidate attempt. Its stdout was overwritten during the baseline recovery. Those original bytes are unavailable. The retained baseline recovery reproduces the original answer/citation counts and score, but not its timing. The candidate output survives from the first and only candidate inference. Its original output metadata points to the adapter’s misrouted baseline location; it was retained without edits.

The source-side [artifact hash manifest](ARTIFACTS_SHA256SUMS.json) keeps its original source paths. In this staged packet, baseline files are under `baseline-recovery/`, and the candidate files are under `candidate-first-run/`; see [source-to-copy hashes](../ARTIFACT_MANIFEST.json) for exact mapping. The original empty retry stdout is not a result; the `NOT_RAW` reconstruction is excluded. See [exclusions and redactions](../EXCLUSIONS_AND_REDACTIONS.md).

Baseline: 10/17 (58.82%), `FAIL`. Candidate: 7/17 (41.18%), `FAIL`. This synthetic suite was intended for evaluation work but is no longer a fresh independent holdout after diagnostic review. No E4 suite or result was found.
