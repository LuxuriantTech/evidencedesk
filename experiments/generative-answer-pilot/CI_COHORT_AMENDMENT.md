# CI cohort amendment before first inference

Declared on 23 October 2026 before any Qwen3 server, smoke inference or
development evaluation has run.

The original `GEN_F1_RELEASE_CRITERIA_PREDECLARED.md` remains unchanged at
SHA-256 `2c7fb031dbf4a1ecc7a49f4f4d72c16ccaacb12ec884a96bbc6301dcfe721484`.
Its initial execution plan was a local WSL experiment. Available Windows memory
was below the 8 GiB loading guard, so that experiment was not started.

For the development measurements, a standard public GitHub Actions Ubuntu
24.04 runner may instead execute this repository's CPU model tests. This amends
the original sentence excluding cloud inference: no paid inference API or
external model endpoint is permitted; repository CI may run the pinned local
model on its own runner. Only the six reviewed, fictional CC0 E4/E5/E6 input
files and experimental source code are public. No user documents, credentials,
session files or private paths are transferred.

Each dataset gets its own job, with one baseline and one candidate execution
in the same rebuilt native environment. These form a new CI cohort. They are
not asserted to reproduce the local environment and cannot be compared across
cohorts to claim a controlled gain. The candidate combines the frozen native
source at `120e233` with the experiment commit's provider and runner files.

The model, runtime, prompt policy, decoding settings, retrieval, sources,
golds, scorer and release thresholds are unchanged. Before the first inference,
the prompt is aligned with the validator: an abstention has no evidence or
answer spans, and an answered response takes every span from one quotation in
one chunk. This resolves two contradictory instructions without using model
outputs. The final provider SHA-256 is
`e77beaedc9f18b58a4c8287161fb516603c7bb39679668d05a1f3cc8a76447f9`.

The service retains the 6 GiB memory cap, no swap, four-CPU quota and nice 10.
It has a 25-minute lifetime within a 35-minute job. The four already declared
invented smoke cases run before the exposed development questions. Failures
retain their raw outputs; no silent retry, policy selection or replacement of
failed evidence is allowed.

All original development, independent-audit and release gates still apply.
A completed CI run is not permission to merge or deploy, and no unseen
GEN-F1 corpus has yet been authored.
