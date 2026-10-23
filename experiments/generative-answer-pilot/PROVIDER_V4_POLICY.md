# Provider response policy V4

V4 removes the model-generated `reason` field. The model supplies exact
`evidence`, exact `answer_spans`, and one `status`; status alone selects the
decision branch. This avoids a redundant label contradicting the selected
status. It does not relax source matching, exact-span checks, or native answer
validation. The previous provider and tests are preserved byte-for-byte under
`history/provider-v3/`.

- `answered`: evidence and spans are required. The final answer still has to be
  assembled from one exact quote in one chunk. Additional corroborating
  evidence remains allowed; spans that cannot share a supporting quote fail
  closed.
- `ambiguous`: at least two distinct exact values are required, each supported
  by a distinct exact quote. The native ambiguity reason is assigned locally
  from this status, never requested from the model.
- `abstained`: both arrays must be empty.

Evidence is selected before status. A source-specific question is answered
from its named source; ambiguity is reserved for incompatible values for the
same requested fact, object, conditions, and time without an explicit
resolution. This same-fact/scope distinction is a model instruction, not a
deterministic semantic parser; exact-span validation can prove that a quote
exists, but cannot itself establish that two claims concern the same fact. JSON
key order is advisory only. The model may only quote supplied retrieved text.
Duplicate JSON keys at any nesting level, backend, schema, span, and
native-validation errors remain blocking and countable.

The prior R3 attempt stopped before candidate benchmarking because the prior
response contract rejected internally inconsistent status/reason output. This
policy change is a development-only contract revision, not evidence that R3 or
V4 passes any quality gate.
