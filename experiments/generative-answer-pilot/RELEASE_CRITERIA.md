# GEN-F1: local constrained-generation experiment

Declared on 23 October 2026 before any Qwen3 model loading, smoke inference, development calibration or GEN-F1 authoring. This is a separate experiment. The failed XLM-R QA runs, the legacy rule runs, their native FAIL verdicts and the unexecuted E7/QA-F1 protocols remain unchanged. No GEN-F1 corpus exists.

## Fixed initial development configuration

- Model: official `Qwen/Qwen3-4B-GGUF`, Apache-2.0, repository revision `bc640142c66e1fdd12af0bd68f40445458f3869b`, `Qwen3-4B-Q4_K_M.gguf`, 2497280256 bytes, SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
- Runtime: official llama.cpp `b11798`, source commit `9febe63d5a379f8568bae7116a5b2ba844a4a4a6`, Ubuntu CPU archive SHA-256 `9342761b01a8d82e0567c9673d3dd8f6264964e0ccb27a008193853c757cf563`. Verify the archive, safe paths, extracted binaries and actual version before loading a model.
- One initial generation policy: temperature 0, seed 20261023, maximum 512 output tokens, thinking disabled. Context limit 6144 tokens, CPU threads 4, batch threads 4 and batch size 128. Before any inference, verify which switches the pinned binary actually supports. A necessary command-line compatibility correction must be recorded before the first model load; it is not an outcome-based tuning step.
- Run only in a task-owned resource scope with at most 6 GiB RAM, no swap, CPU quota 400%, nice 10. Do not change global WSL, browser or system settings. Recheck host memory before model loading; preserve any resource failure and exact resource use. No paid API or cloud inference.
- Before the first load, the pinned binary help confirmed the explicit operational switches: one server slot (`--parallel 1`), CPU only (`--n-gpu-layers 0`), Jinja template enabled, context shifting disabled, physical batch 128 and prediction limit 512. Count the complete chat prompt with the pinned server's `/v1/chat/completions/input_tokens` endpoint before generation. Reject more than 5632 prompt tokens, with no truncation. Freeze and record the complete actual server command in the controller and snapshot.
- Freeze the prompt, JSON schema, decoding policy, provider, runner, binary/model hashes and environments before the first smoke. The smoke uses invented examples outside all evaluation corpora. Every later code/prompt/configuration change needs a new snapshot and exclusive outputs.

## Evidence and failure handling

The generator receives only the question and unchanged native retrieval candidates. Document text is untrusted data, never instructions. No gold, answer aliases, case IDs or outside corpus facts may enter generation. Use the same frozen MiniLM, native retrieval configuration, source commit and native scorer as the existing exposed development runs. Do not rewrite any source, gold, threshold, citation matcher or score.

All returned answer spans and quotes must match the supplied source text exactly. Quotes retain their original chunk, document and page provenance. Mechanical substring verification is necessary but does not prove that the selected fact answers the question. Object, time, comparator, source authority, uncertainty and conflicts must be inspected in actual answer audits. Fixed explanatory phrases are allowed; generated factual prose outside verified spans is not.

Preserve complete prompts, raw responses, accepted/rejected spans and reasons in local-only evidence. A malformed or unsupported model output may abstain safely, but must still count as a distinct model validation failure. Such an abstention must not hide a failed output or qualify as a successful model decision. Network/runtime errors remain errors. No silent retries or postprocessing using expected answers. Do not mistake JSON Schema validity for factual validity or a support score for calibrated probability.

First run only on already exposed E4/E5/E6-v2. Keep all outcomes, failures, native verdicts, runtime/model validation errors, memory and latency. These are development measurements, not independent results. A rule-based candidate can be reported separately; its results must not be mixed with generation results.

## Independent comparison and publication

After development qualification, freeze one reviewed candidate commit and policy. A new author who has not seen runtime code, previous cases or results creates a synthetic supplier corpus plus an unrelated transfer domain, with multi-page prose, structured facts, explicit links, absence, source-specific questions and genuine conflicts. Include at least 18 answerable, five missing and five ambiguous questions. A separate blind reviewer verifies every answer and exact quote against all pages before sealing. Run the unchanged native validator.

Execute the published baseline `ffeba12c4d4d64a1e1f003f211dfa1781c64515d` and the exact frozen candidate once each on identical sealed inputs. Report model/environment changes explicitly; this is not a same-model comparison. Preserve START/FINISH and input/source/model/runtime/environment hashes before and after each role.

Keep the original improvement gates: at least 70% correct answerable questions and at least 25 percentage points over the published baseline, with no regression in citation precision, citation recall or abstention accuracy. Require zero runtime, schema and model validation errors. Keep the native verdict separate. No post-result gold edits, threshold relaxation, replacement of failed raw outputs or repeated attempt called fresh.

An independent audit must verify every emitted answer and citation. Any demonstrable unsupported affirmative blocks release regardless of aggregates. A failed independent attempt requires a general correction and a newly authored unseen evaluation. Existing unit, integration, full-stack/PDF, frontend, lint, typing and dependency checks must pass on the exact candidate, with meaningful actual-model contract coverage. Mock tests alone do not qualify it.

No merge or deployment before these gates pass. Do not publish model binaries, environments, private paths or session files. The static portfolio reader must remain clearly distinct from a hosted backend. A successful internal synthetic evaluation is described with its actual counts and limits, without claiming broad production readiness.
