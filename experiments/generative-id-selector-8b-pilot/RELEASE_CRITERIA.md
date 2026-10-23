# GEN8-F1: separate 8B development and qualification protocol

Declared on 23 October 2026 before any 8B model download, loading, inference,
development scoring or GEN8-F1 authoring. This does not revise the 4B/QA/rule
results or turn any previous failed attempt into a pass. No GEN8-F1 corpus exists.

The model is the official Apache-2.0 Qwen/Qwen3-8B-GGUF Q4_K_M artifact at
revision `7c41481f57cb95916b40956ab2f0b139b296d974`, 5,027,783,488 bytes,
SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
Use the same verified llama.cpp b11798/9febe63d runtime as the 4B experiment,
CPU only, seed 20261023, temperature 0, thinking off, 512 output tokens,
6144 context tokens and exact server-side preflight with a 5632 input-token
ceiling. Reject oversize prompts; do not truncate or silently retry.

Run only on a standard public Ubuntu GitHub runner in a task-owned service:
10 GiB maximum RAM, no swap, four CPUs, nice 10, a 45-minute runtime ceiling
and a 55-minute job ceiling. Require at least 12 GiB host MemAvailable before
loading. The larger bound is declared for the 8B weights and KV/runtime
overhead; it is not a measured capacity claim. No model may run on the user's
PC, and no paid inference API or special paid runner is introduced.

The new strict decision schema, prompt, provider, runners, complete server
command, model/runtime, inputs and environment must be frozen in a new snapshot
before the first 8B smoke. The proposed V5 schema and the model change are two
changes in this separate cohort; no result may attribute a difference solely
to model size. A concurrent 4B V5 experiment is a separate measured variant.

Each job first evaluates the published native baseline on its already exposed
E4/E5/E6 input, before loading the generator. It then loads the model and runs
the four invented component smoke cases. Only if all four pass may the candidate
generator evaluate that same exposed input. A completed baseline does not imply
that any candidate dataset evaluation occurred. The published baseline ffeba12c and native candidate
source 120e233, MiniLM assets, retrieval settings, native validator, scorer,
input bytes, golds and thresholds remain unchanged. Run a same-input baseline
once per dataset/job. Preserve every request, raw response, model validation
error, native verdict, START/FINISH receipt and pre/post hash. A safe fallback
must not hide a malformed model output. All old errors remain errors.

Development qualification requires all three suites to satisfy at least 70%
answerable accuracy, at least +25 percentage points over their same-input
published baselines, no regression in citation precision/recall or abstention
accuracy, zero runtime/schema/model-validation errors, and an independent
audit with no unsupported affirmative. Native verdicts are reported separately.
Do not tune evaluation golds, scorer, retrieval or thresholds to pass.

Only after those development gates pass may one candidate/policy be frozen
for GEN8-F1. A new author who has not seen runtime code, prior cases or results
creates a synthetic supplier corpus plus an unrelated transfer domain, with
multi-page prose, structured facts, explicit links, source-specific requests,
absence and genuine same-scope conflicts: at least 18 answerable, five missing
and five ambiguous cases. An independent blind reviewer checks every answer,
exact quotation and provenance against all source pages before sealing.

Run the unchanged published baseline and frozen candidate once each on the
same sealed inputs with full pre/post identity evidence. The same 70%/+25 pp,
non-regression, zero-error and full actual-answer audit gates apply. No reused
or retrospectively exposed corpus may be called fresh; failed independent
attempts remain failed and require a general fix and a newly authored unseen
evaluation. Existing E7, QA-F1 and GEN-F1 attempts remain distinct.

No merge or product deployment before qualification and meaningful exact-head
unit, integration, full-stack/PDF, frontend, lint, typing, dependency and
actual-model contract checks. Keep model binaries, environments, private paths
and private-document prompts out of public artifacts. The static portfolio
walkthrough remains clearly distinct from a live backend. Passing synthetic
tests is not evidence of broad production readiness.
