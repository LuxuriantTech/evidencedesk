# R2 pre-inference note

Run `40020655855` loaded the pinned Qwen model and reached server readiness, but
the server's default model ID did not match the pinned provider ID. No smoke
completion, candidate answer, or candidate score was produced. Preserve that
run's raw artifacts and failed verdict as-is. The three native baselines ran
before the server and their raw scores are retained. This was the first Qwen
model load, with no generative candidate inference.

The pinned llama.cpp server reports a `created` timestamp for each request at
`/v1/models/data/*/created`. Comparing raw endpoint hashes as an identity check
will therefore differ across consecutive requests, even when the model is the
same. The shared comparator retains the raw endpoint bodies and raw hashes,
then computes a separate canonical identity hash after removing only that JSON
field and its corresponding raw hashes at those exact paths. Model ID,
configuration, size, all other endpoint fields, and their fingerprints remain
part of the comparison. The run receipt keeps both raw snapshots and both
static identity hashes.

The operational server command now supplies the pinned model ID as `--alias`,
and the artifact manifest and runtime help preflight pin and require that flag.
No model, prompt, gold, environment, or decoding setting was changed.
