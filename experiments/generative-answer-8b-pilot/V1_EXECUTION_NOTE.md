# First 8B execution and manifest preflight repair

The first completed job (E4) of run 40024842158 at commit 9732359 passed all
four invented component smoke cases. Its generation child then stopped before
creating a candidate evaluation receipt or issuing any dataset request: an
inline artifact-manifest digest still referred to the preparation copy, while
the parent correctly checked the final published manifest. This is an execution
failure, not a candidate quality score. The other jobs were still pending when
the operational repair was prepared; their final outcomes must be read from
the original receipts, not inferred from this note.

The new revision shares one reviewed artifact loader and digest between parent
and generation child. A regression test checks both entrypoints and rejects a
byte-modified manifest even if its JSON remains parseable. Model, prompt,
schema, decoding, resources, source inputs, native validation and scorer remain
unchanged. Earlier raw responses, failed job and V1 snapshot remain intact.

Passing four component cases is not dataset qualification or permission to
merge or deploy the product. A new baseline/smoke/candidate run is required.
