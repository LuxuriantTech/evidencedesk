# Grounded answer quality development notes

This change improves `grounded-local-v3` using the existing `development_v3`
selection partition. That partition was visible during development; its results
are regression evidence, not an independent evaluation or a holdout result.
Frozen v7 cases, metrics, manifests, and historical negative reports were not
edited or used to select the changes.

The engine now expands a small set of amount-relation equivalents (reserve/fund,
maintenance/entretien, coverage/couverture) when selecting evidence clauses.
Conflicting supported values of the same answer type in one document remain
ambiguous even when their retrieval scores differ. Currency symbols followed by
punctuation are retained by extraction. Field extraction recognizes common
French beneficiary and responsibility wording, and avoids reading `archive` as
an obligation verb when it occurs as a noun or in an organization name.

On the 24-case `datasets/development_v3/evaluation_selection.json` partition,
the recorded ONNX baseline was 16/16 expected citations matched, 16/16 retrieval
hits at 5, 15/16 answerable cases correct, 7/8 abstentions correct, and
extraction TP/predicted/gold of 40/43/45 (precision .930, recall .889, F1 .909).
After these changes, the corresponding counts were 16/16, 16/16, 15/16, 7/8,
and 44/45/45 (precision, recall, and F1 .978). Citation precision was 16/17;
the one extra returned citation and the answerable mismatch are retained in the
result rather than hidden. Errors and schema errors were zero in both runs.
The Fenwick conflict now returns ambiguity, so it is no longer scored as a
correct answerable response. One remaining extraction mismatch is the written
amount `forty-six thousand pounds`, which is returned verbatim but fails the
current evaluator's normalization.

To reproduce with the pinned local model, install the locked development
environment and run the evaluation runner over the selection manifest with
`LocalSemanticEmbeddingProvider`, `RetrievalMethod.HYBRID`,
`DecisionConfig(0.48, 0.38, 0.08)`, and `extract_supplier_fields_v3`. The model
files are verified by the repository manifest and SHA-256 before inference.
This comparison exercises the selected model locally; it does not imply that the
unseen independent evaluation or historical v7 holdout has passed.
