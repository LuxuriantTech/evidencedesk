# 8B ID-selector development cohort

A distinct exposed-development experiment that selects source chunk IDs, then derives native text and citations from those exact chunks. It uses the same 8B model/runtime, retrieval, scorer, MiniLM, baseline and six byte-pinned E4/E5/E6 files as V7. The provider mode and output-token budget differ; this runner also records model-selected IDs separately from the native answer/citations and checks each request's exact dynamic schema fingerprint.

In each answer_calls record, `model_selection` and provider diagnostics contain the model's actual selected IDs/raw content. `response` and `native_derived_answer` are the native GroundedAnswer reconstructed from those IDs, not model-authored prose. The pinned V2 prompt prioritizes unresolved same-scope conflicts as ambiguous, otherwise answers only when supported and abstains when evidence is insufficient.

The four invented smoke cases remain unchanged. This cohort requires service limit 90 minutes and job limit 105 minutes. Results remain exposed development only; release still requires 70% answerable accuracy, at least +25 percentage points over baseline, citation/abstention non-regression, zero technical errors, an independent answer/citation audit, and a separately authored blind corpus. No result is evidence of adoption or production quality.

## Preparation provenance

PREPARATION_SELECTOR_MANIFEST.json describes the original local preparation,
not the installation requirements of this public folder. Its historical V1
manifest is retained locally because it contains a private workspace command.
The V1 provider and V2 report are included for source review. The public file
inventory and workflow are frozen by PILOT_SNAPSHOT_V1.json. No historical
manifest is rewritten or presented as a successful model evaluation.
