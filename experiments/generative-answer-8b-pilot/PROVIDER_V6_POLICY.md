# Provider V6: stable evidence prefix experiment

This is a new exposed-development variant, not a product qualification. V5
results and snapshots are retained unchanged in Git and in their original
CI artifacts. V5 8B already passed all four component smokes in each V1 job;
that run failed an artifact-manifest preflight before candidate dataset calls.
The repaired V5 run 40025810416 was still in progress when V6 was prepared.
Its eventual results must be audited independently and must not be attributed
to V6. No V6 model inference has occurred before this snapshot.

V5 put the question first, followed by chunks in native relevance order. V6
puts all the same chunk records first, sorted by chunk_id, then a retrieval_order
list preserving the exact native ranking, then the question. All six chunk
fields remain byte-for-byte values from the retriever input. A generic prompt
sentence explains this representation; ranking is a relevance hint, never
factual authority. No case ID, expected answer, alias or evaluation fact is
added. Prompt order can change model behavior and must be evaluated anew.

The purpose is to test whether stable prefixes improve llama.cpp KV reuse.
No speed or quality gain is assumed. Comparisons across separate CI hosts are
descriptive, not controlled causal timing claims. The model, schema, validation,
native scorer, decoding, context, resource limits and exposed E4/E5/E6 inputs
are unchanged. Three new mocked tests exercise preservation and prefix order;
they do not measure model behavior. Native runtime and semantic failures stay
failures. All prior release gates still apply, including fresh blind evaluation
only after all development and actual-answer audit gates qualify.

The workflow uses a distinct V6 concurrency group so the ongoing V5 run is not
cancelled or blocked. Both use separate standard public CPU runners, bounded
as previously declared. There is no model execution on the user's computer.
