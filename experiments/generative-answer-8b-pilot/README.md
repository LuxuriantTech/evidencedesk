# Separate 8B generative development experiment

This tests official Qwen3-8B Q4_K_M with strict decision contract V5 and the
V6 stable-prefix payload against the same-input published baseline on exposed
E4/E5/E6. It is separate from all 4B,
QA and rule runs, and is not deployed. Its native candidate source is 120e233;
baseline is ffeba12c. Retrieval, MiniLM, native scorer and six synthetic CC0
input files retain their original bytes.

The pinned artifact manifest records model/runtime/decoding. The isolated
standard public CPU runner service is bounded to 10 GiB, no swap, four CPUs,
nice 10 and 45 minutes, after a 12 GiB host-memory preflight. The job is bounded
to 55 minutes. These are upper bounds, not completion estimates; preparation
uses part of the job budget. No model runs on the user's PC and no paid API is used. The model
and schema change together in this cohort; differences cannot be attributed
solely to model size. A 4B V5 variant is measured separately.

Four invented component cases must pass before full candidate evaluation.
Model validation errors remain errors even when the returned answer abstains
safely. Complete synthetic requests, responses, native outcomes and integrity
receipts are retained; weights, environments and private data are excluded.

See PROVIDER_V5_POLICY.md for the constrained contract and PROVIDER_V6_POLICY.md
for the new payload experiment. RELEASE_CRITERIA.md defines unchanged 70%/+25 pp, citation/abstention non-regression, zero-error and
actual-answer audit gates. Fresh GEN8-F1 authoring is permitted only after
development qualification. No workflow merges or deploys anything.

V7 keeps the V6 answer provider unchanged and explicitly sets `--cache-ram 0`.
See PROVIDER_V7_RUNTIME_POLICY.md for the confirmed prior OOM and the unproven
cache hypothesis. This is a separately measured runtime cohort, not a repair
retroactively applied to V5/V6 results or a change to the 10 GiB limit.


## V8 combined cohort

V8 combines the generic prompt draft, strict primary-quote anchoring, and a 90-minute service / 105-minute job window. The V5 E4 45-minute timeout remains a failure; no quality gain is attributed to any one change. All quality gates and the 10 GiB, zero-swap, four-CPU, cache-zero, 6144/5632-token, and 180-second request limits remain fixed. See PROVIDER_V8_COMBINED_POLICY.md; V8 is exposed development, not release qualification.
