# Generative answer development experiment

This folder tests a constrained local reader against the published rule-based
answerer. It is an experiment, not the deployed EvidenceDesk backend.

The public workflow runs E4, E5 and E6 in three separate standard Ubuntu jobs.
Each job rebuilds the native environment from the unchanged `uv.lock`, runs the
published baseline once, and then runs one generator configuration on the same
inputs. These are exposed development sets. Their results are not an independent
evaluation and must not be combined with measurements from the local machine.

The baseline source is `ffeba12c4d4d64a1e1f003f211dfa1781c64515d`.
The candidate's native retrieval, validation and scoring source is
`120e233bb8ff09cabc1bd3c5004a8545039459aa`. The generator is the additional code
in this experiment commit. Both source worktrees are detached and checked before
and after the run. The six synthetic input files retain their original bytes
and CC0 licence.

## Model and execution

`ARTIFACT_MANIFEST.json` pins the official Qwen3-4B Q4_K_M model, llama.cpp
runtime, decoding settings and resource limits. The preparation script checks
the downloads, archive members, extracted files and native MiniLM assets before
the server starts. The model uses CPU only, a loopback endpoint, a 6 GiB memory
limit, no swap and at most four CPUs. No external inference API is used.

The reader receives the question and the unchanged retrieval candidates. It
selects exact source spans and quotations, preserving document and page identity.
Native answer validation remains in force. Invalid model outputs are recorded
separately even when the returned response safely abstains. Exact quotations and
valid JSON do not by themselves establish that an answer is correct.

The fourth experimental revision uses one generated decision field, `status`,
after evidence and spans. It removes the redundant generated `reason` field.
Revision three selected useful source text in its component smoke but emitted
contradictory status/reason combinations. Every job failed that smoke (2/4), so
no candidate dataset score was produced. Those failures remain failures. The
old provider and tests are preserved in `history/provider-v3`; see
`PROVIDER_V4_POLICY.md` for the new contract. Native validation and release
thresholds remain unchanged.

## Results and release decision

The workflow preserves requests, responses, native scores, errors and integrity
receipts. A completed CI run is not a quality pass. All emitted answers still
need an independent check against their sources, including subject, condition,
time, units, source authority and genuine conflicts.

The release criteria in `RELEASE_CRITERIA.md` require a separately authored,
sealed evaluation after development qualification. They include at least 70%
answerable accuracy, a 25 percentage point improvement over the same-input
published baseline, non-regression of citation and abstention metrics, zero
runtime/schema/model-validation errors, and no unsupported affirmative answers.
No workflow in this folder merges or deploys a candidate.

The uploaded diagnostics contain only the fictional evaluation material and
execution evidence. Model binaries, virtual environments, caches, credentials
and user documents are excluded.
