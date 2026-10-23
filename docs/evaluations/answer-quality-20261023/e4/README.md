# E4 evaluation evidence bundle

**Native result: FAIL for both baseline and candidate. Answerable accuracy is unchanged at 2/18 (11.1%).** This is a small internal experiment on a synthetic development corpus, not a benchmark, certification, or production qualification.

The candidate returned fewer citations (10 to 5): precision rose from 20% to 40%, while recall stayed at 2/18 (11.1%). Correct abstentions rose from 3/10 to 5/10. Retrieval recall at 5 remained 18/18; runtime and schema errors were zero. The two correctly answered cases were the same (`e4-a09`, `e4-a16`); no correct fact answer was gained or lost. Every changed answer became an abstention. The system became more cautious, but did not demonstrate higher answer accuracy.

## Contents

- [Result review](E4_RESULT_REVIEW.md) discusses remaining errors and limits, including the wrong-time answer on `e4-a10`, missed ambiguity on `e4-c04`, and extraction not being evaluated.
- [Frozen protocol](PROTOCOL.md), [layout correction note](LAYOUT_CORRECTION_NOTE.md), frozen comparison rules, freeze receipt, and final input package document the setup. The pre-layout attempt stopped before inference; it is not an evaluation result.
- [Raw baseline and candidate outputs](comparison/) and their original start/finish receipts are preserved. Run-root paths in provenance JSON and stdout are replaced with role labels; hashes of source and staged copies appear in [the artifact manifest](ARTIFACT_MANIFEST.json). Raw result JSON and source inputs are byte-identical.
- [Pinned wrapper, scorer, configuration, lockfile, and model manifest](pinned-toolchain/) provide code/configuration pins. The model weight files are not included; their seven original hashes are recorded in the staged frozen input manifest. The original start receipts contain all 96 runtime distribution pairs. A canonical JSON derivative is included at `pinned-toolchain/environment-inventory.json`; its SHA-256 matches the environment inventory hash pinned before execution. It is explicitly a reconstruction from the receipt, not a separately preserved original inventory file.

## Limits and interpretation

The corpus has 28 fictional wetland field-operation cases: 18 answerable, 5 unanswerable, and 5 ambiguous. This one development run found correct retrieval for all cases at 5, but answerable accuracy remains 2/18 and both native scorer verdicts are `FAIL`. In particular, the candidate still gives a confident wrong time for `e4-a10` and misses the unresolved Q-3/Q-4 ambiguity in `e4-c04`. Extraction quality was not tested: the native output contains no evaluated extraction targets or predictions.

The 40% citation precision results from halving returned citations while retaining the same two correct citations; citation recall did not improve. More abstentions prevent some unsupported or wrong answers but also leave supported questions unanswered. These results support only the narrow statement that this candidate version is more selective on this fixed synthetic set. They do not establish greater factual accuracy, general reliability, or readiness for use.

The final E4 layout was corrected and frozen before the two recorded executions. The expected citation count is 18. An earlier label of 28 counted cases, not expected citations; the frozen manifest was not edited after registration. Gold answers, scorer rules, and thresholds were not changed after the freeze. This bundle is prepared for review and reproducibility; the protocol itself grants no authority to publish or merge changes.

See [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json) for source-to-copy hashes, redactions, and explicit exclusions.
