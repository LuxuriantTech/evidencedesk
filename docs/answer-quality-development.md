# Grounded answer quality and evaluation notes

## Current answer mechanism

The active `grounded-local-v3` answer provider uses a deterministic evidence pipeline. It classifies
the requested answer shape, scopes candidates to an explicitly requested source where the question
distinguishes one, and checks whether a candidate passage supports the requested relation under the
applicable local condition. For example, two duration values are not treated as contradictory merely
because both occur in a dossier: the event type and the relation must also match. A document named
as the subject of a question is not automatically treated as the source. Negation is attached to the
relevant relation; coordinated subjects sharing one predicate retain that predicate's scope, while
separate clauses with their own predicates remain distinct. Unsupported, compound, or genuinely
conflicting questions can be left unanswered or marked ambiguous. Returned answers retain a cited
passage that is checked against the retrieved evidence.

These rules and aliases are intentionally limited. This is not a full language parser: implicit
references, unfamiliar phrasing, and complex cross-document conditions can still be missed. The
internal support score is **not calibrated** as a probability of correctness. The interface labels it
as source support (`Appui des sources`, x/100); it must not be read as a confidence percentage.

## Results and status of the evaluation sets

The independent v7 holdout remains unchanged and negative: `answerable_correct` is **9/25 (36%)**.
Its other previously recorded results remain in the root README and
[`evaluation.md`](evaluation.md); v7 was not rerun or used to choose these changes.

E1 and E2 have now been read and used during development. They are **development comparisons, not
independent or holdout evaluations**.

| Comparison | Answerable correct (`metric_counts`) | Other recorded counters | Interpretation |
|---|---:|---|---|
| First E1 baseline/candidate comparison | 7/13 / 7/13 | citations P/R 0.615385/0.615385 for both; abstention 1/7 for both | No measured delta in that frozen comparison. This original result is retained. |
| Latest E1 development run (`candidate-run-v6`, source `1ffbaff`) | 10/13 | expected citations matched 13/13; retrieval hits@5 12/13; abstention 7/7; errors/schema errors 0/0 | Later development result; do not substitute protocol-outcome totals for `answerable_correct/total`. |
| First E2 baseline / candidate (`123535b`) | 6/13 / 4/13 | candidate abstention 4/6; errors/schema errors 0/0 | Negative initial comparison; retained as recorded. |
| Later E2 development run (source `1ffbaff`) | 11/13 | expected citations matched 13/13; retrieval hits@5 12/13; abstention 5/6; errors/schema errors 0/0 | Development-only comparison; not an independent result. |
| Development_v3 calibration (source `1ffbaff`) | 16/16 | citations 16/16; abstention 8/8; extraction TP/gold 44/44; errors/schema errors 0/0 | No delta from the prior saved calibration run. |
| Development_v3 selection (source `1ffbaff`) | 16/16 | citations 16/16; abstention 7/8; extraction TP/gold 44/45; errors/schema errors 0/0 | No delta from the prior saved selection run; the abstention target is still missed. |

Counts above are the runner's `metric_counts`, especially `answerable_correct` and
`answerable_total`; they are not manually reclassified using secondary outcome labels. E1, E2 and
development_v3 are synthetic suites and provide limited regression evidence. They do not establish
performance on new documents, real contracts, or production traffic. No E3 candidate result or
independent conclusion is reported here; that evaluation is being prepared separately. The listed
E1/E2/development_v3 development runs predate the focused follow-up fix `8d709d0`; its focused and
full unit gates passed, but these suites were not rerun after that fix.

## Reproduction

Run checks from a native Linux checkout with LF line endings and the locked environment. Do not
reuse a Linux `.venv` copied into a Windows checkout. The ONNX model path must resolve to the model
verified against `infra/models/paraphrase-multilingual-minilm-l12-v2-text-onnx.json`.

```bash
export REPO=/path/to/native-linux-checkout
export MODEL_DIR=/path/to/verified/paraphrase-multilingual-minilm-l12-v2
# For the saved comparisons only; keep these outside the source checkout.
export E1_DRIVER=/path/to/preserved/evaluation-development-cycle2/run_e1_candidate_v6.py
export E2_RUN=/path/to/fresh/copy/of/evaluation-independent-v2

cd "$REPO"
uv sync --locked
uv run ruff check apps/api apps/worker evals scripts
uv run mypy apps/api apps/worker evals
uv run pytest apps/api/tests/unit apps/worker/tests evals/tests \
  -m 'not integration and not e2e' -q
```

To rerun the saved E1 development driver, set `REPO`, `MODEL_DIR`, and `E1_DRIVER` to the native
Linux checkout, verified ONNX model directory, and preserved driver under
`output/portfolio-product-20261023/evaluation-development-cycle2/` respectively:

```bash
cd "$REPO/apps/api"
EVIDENCEDESK_REPO="$REPO" MINILM_MODEL_DIR="$MODEL_DIR" \
  uv run python "$E1_DRIVER"
```

For E2 development comparisons, use a fresh copy of its preserved development input directory and
its `adapters/run_baseline.py`; set `E2_RUN` to that copy. The adapter writes only inside that copy.
Keep the original frozen E2 baseline/candidate reports and their raw files unchanged. Run the copied
adapter with:

```bash
cd "$REPO/apps/api"
EVIDENCEDESK_REPO="$REPO" MINILM_MODEL_DIR="$MODEL_DIR" \
  uv run python "$E2_RUN/adapters/run_baseline.py"
```

These E1/E2 commands reproduce development comparisons, not independent evaluations. The
integration test that resets and reseeds a database must only run against an isolated database
owned by that test run; do not point it at a shared service.

The frozen answer configuration remains `evals/configs/answer-v3-frozen-v7.json`; decision values
`0.48`, `0.38`, and `0.08` were not changed for this work. Evaluation manifests, expected answers,
thresholds, and historical raw results were not edited.

## Earlier development_v3 calibration snapshot

The earlier recorded cycle selected the deterministic configuration under its predeclared maximin
rule. At that time, development selection measured 15/16 answerable cases, 6/8 abstentions, 16/16
expected citations, extraction P/R/F1 `0.953488/0.911111/0.931818`, Recall@5 `1.0`, and zero
technical or schema errors. The abstention score was below the stated 0.85 target. Later development
work changed the runtime, not those saved historical artifacts; the newer calibration and selection
counts are listed above.
