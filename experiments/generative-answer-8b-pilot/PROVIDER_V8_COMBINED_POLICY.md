# V8 combined provider and runtime policy

Status: staged-candidate policy only; not run, measured, or qualified.

V8 combines two changes in the 8B provider: the generic `SYSTEM_PROMPT` draft
and strict primary-anchor selection in `_answer_excerpt`. It also lengthens
the bounded service window to 90 minutes and its workflow job timeout to 105
minutes so the full case set can finish. These are three combined changes;
any later result cannot be attributed to the prompt or selector individually.

The prompt asks for one answer source anchor and keeps corroborating evidence
in separate evidence quotes. The selector accepts a primary citation only
from the chunk named by every answer span and only when one quote from that
chunk contains every span. If the spans do not share one chunk and quote, the
existing fail-closed abstention path remains in force. It does not deduplicate,
reassign, or remap spans.

The window change follows the V5 E4 job reaching its 45-minute service limit
with 18 of 28 outputs. That timeout remains a failure; more time is not
evidence of a quality improvement. Earlier V5, V6, and V7 outcomes remain
unchanged and must not be requalified. The V7 45-minute service and 55-minute
job limits remain in their historical snapshot and release criteria.

Only these resource values change: `service_runtime_max_seconds` 2700 to 5400
and `workflow_job_timeout_minutes` 55 to 105. The following values remain
unchanged and are checked by the staged tests and manifest: memory max 10 GiB,
swap 0, CPU quota 400%, nice 10, cache RAM 0 MiB, context 6144 tokens,
maximum input 5632 tokens, and per-request timeout 180 seconds. Model, runtime,
generation settings, retrieval order, scorer, golds, datasets, and all quality
gates remain unchanged. The historical `RELEASE_CRITERIA.md` is preserved
byte-for-byte.

There is no quality claim in this policy. A future CI result is a new exposed
development cohort on previously exposed data, not independent validation,
product qualification, or a release approval. Root must review the exact
staged bytes and complete the mocked tests before deciding whether to run it.
