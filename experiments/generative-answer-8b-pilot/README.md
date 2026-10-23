# Separate 8B generative development experiment

This tests official Qwen3-8B Q4_K_M with strict decision contract V5 against the
same-input published baseline on exposed E4/E5/E6. It is separate from all 4B,
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

See PROVIDER_V5_POLICY.md for the constrained contract and RELEASE_CRITERIA.md
for unchanged 70%/+25 pp, citation/abstention non-regression, zero-error and
actual-answer audit gates. Fresh GEN8-F1 authoring is permitted only after
development qualification. No workflow merges or deploys anything.
