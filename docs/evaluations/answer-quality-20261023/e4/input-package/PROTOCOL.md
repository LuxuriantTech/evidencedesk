# E4 frozen evaluation protocol

Status: FROZEN after independent review of the synthetic corpus and golds, before either role is executed. No model, adapter, baseline, or candidate inference has been run on E4 at freeze time.

## Dataset and execution layout

The evaluation inputs are under inputs/. The pinned common adapter is inputs/adapters/run_baseline.py; its parent directory is the adapter input root. The suite and corpus manifest are inputs/datasets/evaluation_cases.json and inputs/datasets/corpus_manifest.json. Every corpus source_path resolves beneath inputs/ at inputs/datasets/sources/. These source payloads are required inputs, included in the frozen manifest, and must be present byte-for-byte before registration or execution.

The corpus is synthetic and concerns fictional wetland field operations. It contains ten documents and 28 questions: 18 answerable, 5 unanswerable, and 5 ambiguous. An independent reviewer checked all 28 citation excerpts against their source document/page, confirmed answerable golds are directly supported, and found no joins based solely on co-occurrence. Cross-document joins require explicit links between typed identifiers. No semantic corrections were requested. The source/citation objects were transformed into the adapter's E3-compatible representation without changing case IDs, answers, citation text, or page numbers; the earlier pre-layout draft is preserved under history/pre-layout/.

## Pinned comparison

The baseline commit is 0ae029757bfae524034d07335d0c518423b4afa9. The candidate commit is ffeba12c4d4d64a1e1f003f211dfa1781c64515d. Use only the pinned adapter, scorer, thresholds, configuration, and MiniLM ONNX model bundle recorded in FROZEN_INPUT_MANIFEST.json. Both runs must use Linux CPython 3.12.14 with the same pinned environment inventory. The already-reviewed execute_frozen_eval_v2.py wrapper must be used, with distinct exclusive output roots for baseline and candidate. No existing run directory may be reused or overwritten.

The native scorer is authoritative. Do not change aliases, normalizers, thresholds, citation handling, or verdict rules after this freeze. Report the scorer's exact results, including failures, abstentions, citations, and per-case output. Do not substitute a manually recalculated score.

## Recorded release criteria

For a release recommendation, compare answerable accuracy, citation precision, citation recall, and abstention performance using the frozen scorer output. All four must be non-regressive, and at least one must improve. Schema errors must be zero and the full project CI stack must pass. These criteria are recorded for interpretation only; they do not alter the scorer's native verdict or convert a native FAIL into a pass. Report any divergence between the native verdict and the recorded release criteria plainly.

## Scope and limits

This is a synthetic, locally constructed evaluation with an independent internal review, not a third-party benchmark or certification. E1, E2, and E3 are development datasets and are not evidence of held-out E4 performance. Preserve both complete runs and their receipts/raw outputs. This protocol does not authorize publishing, pushing, or merging.
