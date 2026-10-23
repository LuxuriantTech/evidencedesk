# Reproduction and scope

This public package contains a synthetic 19-case evidence-grounding evaluation. Corpus, questions, gold answers, expected citations, frozen runner criteria, baseline/candidate raw results, and per-case comparison are included. No real customer or personal documents are used.

- Repository: https://github.com/LuxuriantTech/evidencedesk
- Baseline commit: `0ae029757bfae524034d07335d0c518423b4afa9`
- Candidate commit: `123535b9b1c45603bce1ee1ccfd5adec07bb7c38`
- Model: `paraphrase-multilingual-minilm-l12-v2-onnx-q@faf4aa4225822f3bc6376869cb1164e8e3feedd0` (Apache-2.0); model files are not redistributed. The pinned model manifest and file hashes are included.
- Answer configuration: `answer-v3-frozen-v7.json`; its SHA-256 is `a3987fdb7bb2db037a4089044438a79d150db4b6bcc5ab136b868cd6c54c7a38`.
- Dependency lock: `uv.lock` included. Runtime used for both E2 runs: Linux CPython 3.12.14 with locked dependencies.
- Score source: frozen `evals/runner.py` at baseline commit; primary metric is `metric_counts.answerable_correct / answerable_total`, with no post-run alias or gold adjustments.
- Baseline result: `6/13`; candidate result: `4/13`.

The dataset is small and synthetic, not an independent third-party blind benchmark, legal-quality assessment, or production qualification. It does not evaluate extraction (no extraction golds), deployment security, availability, or real-world generalization. Scores are limited to these frozen cases and pinned engine/model versions. The test runner stdout and machine-local artifact manifests are intentionally omitted because they may contain local paths; the included raw JSON remains unchanged unless the public manifest marks it redacted.
