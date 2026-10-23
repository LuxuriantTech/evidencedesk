# ID selector prompt V2 preparation

V2 changes only `SYSTEM_PROMPT` relative to V1. The explicit decision order is: first, two distinct eligible chunks within the source and scope requested by the question, containing incompatible values for the same relation, object, time, and conditions, with no explicit correction or priority, MUST produce `ambiguous` with those two IDs; do not abstain merely because of that conflict. Second, a conflict contained within one chunk is outside this two-ID protocol and must abstain. Otherwise, answer when the requested fact is explicitly supported; abstain when it is not.

The provider and original manifest for V1 are preserved byte-exact under `history/v1/`. V2 uses the same response schema, adapter, guards, limits, tests, and 256-token generation variant as V1. The source baseline remains SHA-256 `4a782199bf3c80273dc4c8975e1020253151f6c18ee7ed4f61b69c0cbdef7570`.

Ten existing tests pass under the supplied WSL environment using real frozen API classes and validator, with transport mocked. The AST comparison of V1 and V2 reports exactly one changed top-level assignment: `SYSTEM_PROMPT`. No test or other provider code changed for V2.

The historical WSL environment is Python 3.12.15 with Mako 1.4.1. This does not establish compatibility with the separate GitHub CI runtime pinned to Python 3.12.14. No install or environment synchronization occurred. The pinned llama.cpp schema check remains pending GitHub CI; no inference, model, server, Docker, or database was run.

This remains an experimental alternative, not adopted or benchmark-evaluated. Full passages remain potentially verbose. Conflicts limited to one chunk remain unsupported by the two-ID ambiguity protocol and are directed to abstention.
