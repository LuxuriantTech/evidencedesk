# Local 8B synthetic HTTP smoke: V2 preparation

This is a revised, isolated copy of the first HTTP-smoke draft. The original remains unchanged. Nothing here has been staged into the product checkout, committed, pushed, or executed against Docker, a database, or a model.

## Candidate and exact prerequisites

- Intended base: `integration-main-candidate` HEAD `1ee666834caf7fb8f31b4fe0a07813a998335b1b`. The coordinator later reported no-inference CI run `40038531441` green on this base; I did not independently query GitHub. Because the future staging adds files and creates a new SHA, its no-inference CI must pass again for that exact new SHA before the HTTP workflow proceeds.
- Before the future HTTP workflow starts a model, it waits up to ten minutes for `integration-local8b-ci.yml` to complete successfully for the **same `GITHUB_SHA`**. Any failed conclusion, missing run, or timeout fails closed. This is the no-inference CI prerequisite, not a quality pass.
- The workflow pins SHA-256 for `local_generation_core.py`, `local_generation.py`, `scripts/local_generation/manifest.json`, and `uv.lock` from the reviewed base. It also checks that exact base is an ancestor and that checkout HEAD equals `GITHUB_SHA`.
- The same-SHA CI query uses the read-only `actions` permission and exposes `github.token` only to the wait step as `GH_TOKEN`.
- The current product manifest uses model alias `Qwen3-8B-Q4_K_M.gguf`, context 6144, input budget 5632, output limit 256, cache RAM 0, and one slot. V2 smoke calls use a 240-second HTTP timeout for `/ask` while short API checks retain 20 seconds. No runtime/model setting is changed by this harness.

## Trigger and ordering

The workflow draft uses a narrowly filtered `push` on the already experimental branch `codex/evidencedesk-local8b-integration-20261023`, with paths limited to the staged smoke directory and its workflow file. This lets the existing no-inference workflow and the new smoke workflow both be triggered by the same experimental commit. It does not depend on a `workflow_dispatch` workflow being present on the repository default branch. The future workflow must remain outside the product until reviewed and staged; this draft does not trigger anything.

After the same-SHA no-inference CI succeeds, the synthetic HTTP smoke can technically run alongside the separate development-quality run: they use isolated ephemeral GitHub-hosted runners, isolated Compose projects, generated credentials and fake documents, and the HTTP test does not read a corpus, golds, scorer, or quality receipts. This is an infrastructure observation only. It cannot qualify the model, change any quality threshold, satisfy the development gate, or authorize merge/fresh evaluation. Keeping dev-first ordering remains a process choice, not a technical dependency.

## What the HTTP smoke checks

- Unauthenticated calls return 401; reader uploads return 403; analyst upload and worker ingestion complete.
- Two generated text files state different dates for the same fake asset in different dossiers. Each `/ask` response must contain the matching date and only cite the document in the requested dossier. An empty dossier must abstain without citations.
- Stopping only this run's model sidecar must yield sanitized `503 local_model_unavailable`.
- The smoke stores only reduced status/check receipts. The synthetic question and documents are sent only to this isolated local stack; no raw prompt or answer is uploaded as an artifact.

This tests the real API/provider/worker route and dossier filter for these synthetic examples. It is not a general semantic-quality result or proof of a multi-tenant ACL. The API's ask endpoint requires `Action.ASK_QUESTION`; retrieval itself filters completed, undeleted documents by `Document.dossier_id`. Dossiers are inserted only in the disposable database because no dossier-create route is available in this app state. `Dossier` requires `id` and `name`; `description` and `is_synthetic` have defaults, so the proposed explicit INSERT fields are valid against the current model.

## Transport corrections in V2

- `request_json` now accepts a bounded per-request timeout, with a 20-second default and 240 seconds for model-backed asks. The provider allows 180 seconds; the extra allowance covers the HTTP/API path. The complete smoke step remains bounded by its workflow timeout.
- It uses an opener with `ProxyHandler({})` and a no-redirect handler. Thus environment proxy settings cannot route bearer credentials elsewhere, and a redirect is surfaced as an HTTP failure rather than followed off-loopback. The server-identity probe in the final V2 workflow uses the same explicit no-proxy/no-redirect pattern and sends no credentials.
- The workflow exports `EVIDENCEDESK_8B_BUNDLE="$HTTP_BUNDLE_DIR"` inside each shell step instead of reading a dynamically appended `GITHUB_ENV` value through the `${{ env.* }}` expression context.

Mocked tests cover the timeout passed to the opener, the short default, disabled proxy and redirect handlers, and rejection of out-of-range timeouts. They do not open sockets or run services.

## Bounded local staging script

`scripts/local_generation/http_qualification/` is the canonical V2 source tree, laid out exactly as it would be staged into the product. `stage_experiment.py` is a reviewable script for a future, explicit local staging step. It requires the exact clean base SHA and pinned core/manifest/lock hashes, creates a fresh clone under the cycle directory on the experimental branch (reusing that local branch only if it already points at the exact base), copies only the allowlisted harness files, one documentation file, and one workflow file, and emits a SHA-256 inventory outside the clone. It refuses an existing destination or manifest, dirty/wrong-head source, symlinked inputs, or any path escaping the cycle directory. It does not commit, push, launch CI, run tests, build images, or start a model. A failed partial clone is left for inspection instead of being recursively removed. It has not been executed.

The eventual experimental commit SHA will be new and is intentionally not invented here. The workflow verifies the exact checked-out SHA at runtime; no merge to the default branch is proposed.

## Scope and current state

This folder is a preparation artifact only. The original draft was preserved. No workflow was triggered, and no model-quality/fresh/merge gate has been changed or passed by this HTTP preparation.
