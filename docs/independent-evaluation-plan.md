# Independent answer-quality evaluation plan

This plan is prospective. It records no new model run, holdout score, or validation claim. The
published v7 result remains the historical frozen result and must not be reused to select rules,
thresholds, prompts, or model settings.

The focused effective-date/renewal-date regression added alongside this plan is a development test
derived from an observed v7 failure. It is not a new evaluation case, a rerun of v7, or independent
performance evidence.

The code change is limited to the `DeterministicGroundedAnswerProvider` used by `grounded-local-v3`.
The default `ExtractiveAnswerProvider` path is unchanged.

## Before collecting evaluation cases

1. Freeze the code revision, dependency lock, answer-provider configuration, extraction schema,
   prompt and thresholds. Record hashes and the runtime/model identifiers where applicable.
2. Define answerability and gold evidence rules before authoring cases. Include explicit effective-date
   versus renewal-date pairs, same-role date conflicts, missing fields, multilingual forms, and
   representative non-date questions. Keep each case tied to a document, page, and exact evidence.
3. Have an author create the cases and a separate reviewer verify expected answers, citations, and
   answerability without access to system outputs. Keep case IDs, gold labels, and source documents
   sealed from the implementation process until the run is frozen.
4. Use a distinct synthetic document corpus and case set. Do not copy v7 cases or tune on this new
   evaluation; if any result is used to change the system, retire it as a holdout and create another.

## Run and report

Run the frozen application once under a recorded configuration. Preserve raw inputs and outputs,
errors, model/provider identifiers, seeds where supported, timings, and hashes. Report answerable
case correctness, citation correctness, abstention correctness, ambiguity handling, extraction
precision/recall/F1, and safety failures with denominators and per-category counts. Separate the
deterministic extractive path from any model-backed path. Manually adjudicate disagreements against
the prewritten rubric and retain both the original and adjudicated labels.

No success threshold is asserted here. Set decision thresholds before opening evaluation outputs,
based on the intended use and reviewed risk; until then, report measurements descriptively and keep
the system's research-prototype limitations visible.
