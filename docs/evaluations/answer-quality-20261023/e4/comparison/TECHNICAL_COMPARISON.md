# E4 comparison report

## Result

The frozen native scorer returned **FAIL for both baseline and candidate**. The candidate did not improve answerable accuracy: it answered 2 of 18 answerable cases correctly, exactly matching baseline (2/18, 11.1%). Do not describe this as improved answer quality.

| Native metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Native verdict | FAIL | FAIL | unchanged |
| Answerable correct | 2/18 | 2/18 | unchanged |
| Citation precision | 0.200000 | 0.400000 | improved |
| Citation recall | 0.111111 | 0.111111 | unchanged |
| Abstention accuracy | 0.300000 (3/10) | 0.500000 (5/10) | improved |
| Citations returned | 10 | 5 | fewer returned |
| Expected citations matched | 2/18 | 2/18 | unchanged |
| Retrieval hits at 5 | 18/28 | 18/28 | unchanged |
| Retrieval recall at 5 | 1.000000 | 1.000000 | unchanged |
| Errors / schema errors | 0 / 0 | 0 / 0 | zero both |

Candidate precision and abstention improve, while answerable accuracy and citation recall remain unchanged. Citation recall is still 2/18. The smaller number of returned citations also means the precision increase should not be confused with broader evidence coverage. The native FAIL remains decisive; this comparison does not support a release recommendation.

The recorded secondary criteria show no regression in the four listed measures and improvement in precision and abstention, but they do not override the native scorer verdict. CI is a separate gate; these model-evaluation receipts do not attest the full repository CI result.

## Provenance and preservation

This comparison uses the two completed E4 role runs, registration SHA-256 24ec1c048e914b7ccbea9419e505a29b3d51508a1d52fcc20fc8b28bb4931a81, baseline commit 0ae029757bfae524034d07335d0c518423b4afa9, and candidate commit ffeba12c4d4d64a1e1f003f211dfa1781c64515d. Both raw outputs have identical suite and corpus hashes, and their byte-for-byte SHA-256 values are recorded in candidate-run/ARTIFACTS_SHA256SUMS.json. Each original run directory, including start/finish receipts and stdout/stderr, is copied under provenance/ and verified byte-identical to its WSL source. The original WSL run roots were not modified.

The E4 package originally had an incomplete pre-layout freeze; root registration failed before creating a run or executing inference. The corrected adapter-compatible inputs and all source payloads were then frozen and registered. The pre-layout artifacts remain archived under the parent E4 history directory; see LAYOUT_CORRECTION_NOTE.md.

The early manifest label citation_quotes_exact: 28 counted the 28 reviewed cases, not 28 expected citations. The final gold set has 18 answerable citation expectations; the independent review found all expected citation excerpts exact and supported. The frozen manifest was left unchanged after registration; this clarification is recorded here.

## Limits

This is one 28-case synthetic internal evaluation, not a third-party benchmark or certification. Both runs use the same model, corpus, scorer, thresholds, and adapter, but the small sample limits generalization. E4 is now development evidence after execution. Do not call the system generally reliable or say the candidate has better answer accuracy based on these results.
The recorded-evaluation builder contract was checked against this folder using a temporary output file; it read all 28 cases and emitted the native baseline/candidate metric fields. The preview stayed in the system temporary directory and was not published.
