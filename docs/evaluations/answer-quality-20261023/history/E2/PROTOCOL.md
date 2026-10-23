# E2 comparison protocol

Version `e2-fixed-runner-v1`; frozen before any baseline or candidate run. The corpus and cases are newly authored, synthetic, and contain four realistic documents with narrative, contractual distractors, and multilingual supplier records. No outputs have been inspected.

The suite has 19 unique cases: 13 answerable (including English/French paraphrases, close values, and negative clauses), 4 unanswerable, and 2 ambiguous (an MSA/addendum collision and an unspecified supplier identity). All cases are labelled `development` only to invoke the repository’s non-holdout evaluator; this is not a claim of third-party holdout certification.

Gold answers and citation excerpts are fixed in `datasets/evaluation_cases.json`; the corpus is in `datasets/corpus_manifest.json` and matching source text under `datasets/sources/`. No case or gold may change after baseline execution. No extraction targets are included.

**Primary score:** use only the frozen runner’s `metric_counts.answerable_correct / answerable_total` (also reported as `citation_case_accuracy`), exactly as emitted. Keep its exact `answer_match`, citation, status, abstention, and error semantics; do not add aliases, hand-scored overrides, or post-run tolerances. Secondary metrics: native citation precision/recall, retrieval recall@5/MRR@5, abstention accuracy, error and schema-error counts. Frozen runner thresholds remain in `COMPARISON_RULES_FROZEN.json`; extraction F1 is non-applicable because there are no extraction golds.

The single adapter is byte-identical to E1’s `adapters/run_baseline.py`. Its bytes, grader files, model manifest, suite, corpus and protocol are hashed in `SHA256SUMS.json` before baseline execution. For candidate comparison, use the identical adapter and change only `EVIDENCEDESK_REPO` to the candidate’s detached clone. No candidate test is authorized until a later explicit freeze instruction.

Both E2 runs must use native Linux, CPython 3.12.14, locked dependencies, the same verified ONNX cache, and the same answer config. Capture complete raw JSON and per-case outputs.
