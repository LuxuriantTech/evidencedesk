# EvidenceDesk evaluation history

This directory is staged for review and can be copied as a unit to `docs/evaluations/answer-quality-20261023/history`. It has not been published or copied into the repository.

- [E1 index](E1/README.md) covers the first 20-case synthetic comparison, its frozen protocol and inputs, unchanged adapter, raw baseline/candidate results, per-case comparison, and reports. Both versions score 7/13 and both are `FAIL`.
- [E2 bundle README](E2/README.md) leads to the existing public-ready 19-case synthetic bundle, copied byte-for-byte with its own manifest. The candidate regresses from 6/13 to 4/13; both verdicts are `FAIL`.
- [E3 index](E3/README.md) covers a 29-case synthetic development evaluation. The first baseline raw/stdout were lost by overwrite; only the recovery baseline is retained. The first and only candidate raw is preserved byte-for-byte, with machine paths redacted from staged stdout. The candidate scores 7/17 against the recovery baseline's 10/17; both verdicts are `FAIL`.
- [Results](RESULTS.md) summarize the three outcomes and limits; [exclusions and redactions](EXCLUSIONS_AND_REDACTIONS.md) explain staged differences. `ARTIFACT_MANIFEST.json` maps each staged file to its source and records source/copy SHA-256.

All corpora are synthetic. The model weights and machine caches are not redistributed. E2's lockfile, model manifest, configuration, grader and data are included. E3 includes the same shared grader, lockfile, model manifest and configuration from E2, whose required hashes match the E3 frozen rules. E3's original source-side manifest marked that transfer private; this local staging does not publish it or override that historical provenance.

The E3 `NOT_RAW` terminal reconstruction is explicitly excluded from results. The empty stdout file is an interrupted retry marker, not a run, and is also excluded. No E4 suite or result exists in the inspected source directory. No runtime, scorer, gold answer or source result was edited.
