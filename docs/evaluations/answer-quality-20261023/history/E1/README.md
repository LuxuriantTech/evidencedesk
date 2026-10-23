# E1: first frozen comparison

- [Protocol](PROTOCOL.md) and [input hashes](SHA256SUMS.json)
- Frozen [questions](datasets/evaluation_cases.json), [corpus manifest](datasets/corpus_manifest.json), and synthetic source documents in `datasets/sources/`
- Unchanged [run adapter](adapters/run_baseline.py)
- [Baseline report](baseline-run/BASELINE_REPORT.md) and [baseline raw output](baseline-run/raw_results.json)
- [Candidate comparison](candidate-run/COMPARISON_REPORT.md), [per-case comparison](candidate-run/case_comparison.json), [candidate raw output](candidate-run/raw_results.json), and [stdout](candidate-run/runner_stdout.txt)

Both baseline and candidate score 7/13 and both retain the native `FAIL` verdict. The full engine checkout is not copied. The original baseline report points to `baseline-checkout/` for engine files; those historical paths are not present in this packet. The frozen adapter, input hashes and evaluation outputs are included, and the baseline commit is `0ae029757bfae524034d07335d0c518423b4afa9`. Candidate timing is not directly comparable because baseline ran on Windows and candidate on Linux.
