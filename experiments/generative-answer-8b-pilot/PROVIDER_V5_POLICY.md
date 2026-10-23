# Decision contract V5

This is a new exposed-development policy, not a correction of old scores.
V4 passed 3/4 component cases in each job but failed its empty-array abstention
contract. No V4 candidate dataset benchmark ran. All previous failures and
snapshots remain preserved.

Three mutually exclusive JSON-schema branches constrain the single status:
answered requires at least one exact proof/span; ambiguous requires at least
two of each; abstained requires empty arrays. The pinned llama.cpp parser
supports the union, const values and array bounds. Schema restrictions do not
prove semantic correctness; exact-quote/native validation and error accounting
remain unchanged. The prompt requests status first, then evidence and spans.

A request for a missing value requires abstention. An explicit question about
existence, availability or whether information was recorded can be answered by
an exact affirmative or negative statement of that status. No evaluation facts,
IDs, expected values or aliases are embedded in the provider.

Sources: pinned llama.cpp common/json-schema.cpp and
common/json-schema-to-grammar.cpp at 9febe63d5a379f8568bae7116a5b2ba844a4a4a6.
Actual server smoke and complete answer audits remain required. Mock tests and
valid JSON cannot qualify factual answers or authorize product deployment.
