# Optional local 8B pilot

This branch prepares a separate answer backend for EvidenceDesk. It is an
experiment, not a measured or deployed replacement for the published backend.
The regular configuration keeps `grounded-local-v3`. The new mode is explicitly
opt-in and is rejected when public demo mode is enabled.

The pilot uses the official Qwen3-8B Q4_K_M model and a pinned llama.cpp runtime.
The model selects exact source passages; the API reconstructs the answer from
those passages and validates their provenance. This prevents invented answer
text from being displayed, but does not prove that a selected passage answers
the question correctly. Semantic quality needs separate evaluation.

The pinned launcher sets `--cache-ram 0` from the manifest. This disables the
server's RAM prompt-cache pool and idle-slot snapshots, while active-slot
longest-common-prefix reuse remains a separate mechanism. E5's 10 GiB cgroup
OOM is confirmed, but its cause is not attributed to the cache. Treat this
setting as an unvalidated hypothesis for a controlled resource run, not as a
proven memory fix. The memory limit, context, batch, and other launch settings
are unchanged.

The adapter accepts a loopback endpoint only, verifies the server's declared
model/context/slot configuration, and admits one request per API provider
instance. It returns generic unavailable/busy responses. Raw requests and
model diagnostics are discarded after each application request. These measures
are not cryptographic attestation or secure erasure of the model's memory.

See `scripts/local_generation/README.md` for explicit asset preparation and the
optional Compose overlay. Preparation verifies the model and runtime hashes;
the launcher checks all extracted runtime files before execution. The model is
not downloaded by normal application startup or by the integration CI job.

The integration CI rebuilds the locked Python environment, including Mako
1.4.3, tests the API and asset handling, builds the sidecar image, and checks
the runtime binary with `--version`. It does not load Qwen, start an inference
server, or validate API-to-model behavior. Model-quality experiments and their
failed historical results remain separate. No quality score or release approval
can be inferred from this branch's unit tests or successful image build.

Before adoption, the exact candidate must pass actual-answer review, the
declared development criteria, an independently authored sealed evaluation,
and full integration checks. Earlier development measurements on a different
source tree are not qualification of this integration branch.
