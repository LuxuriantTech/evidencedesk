# Exclusions and redactions

The original source files were not modified. This staged copy removes only machine-specific paths and records source and copy SHA-256 in `ARTIFACT_MANIFEST.json`.

- E1 candidate `ARTIFACTS_SHA256SUMS.json`: the `clone_path_linux` value is replaced with a redaction marker. All other fields are preserved. The E1 `run_candidate.sh` is excluded because it hardcodes a local clone path; the unchanged shared adapter is included.
- E3 baseline and first-candidate stdout: the single user-specific absolute output path is redacted in staged stdout copies. Their raw JSON results remain byte-identical to source and preserve the recorded run metadata.
- The E3 reconstructed `first-baseline-stdout-transcription_NOT_RAW.txt` is excluded from raw evidence. Its source SHA-256 is recorded in the manifest. It is a reconstruction, not the lost original stdout. The recorded original stdout hash was `86177c59b411ab10f570d240ee9d9b14e764b1d1b44be7108ddc17021e938106`; the original stdout bytes are unavailable.
- The zero-byte E3 candidate retry stdout marker is excluded; the first and only candidate inference has its own retained raw and stdout files.
- An E1 noncanonical suffixed `raw_results.json` near-duplicate is excluded; the canonical result referenced by the comparison report is retained.
- Model binaries, caches and detached repository checkouts are not copied. Their model/file hashes and locks are retained where available.
- E3's abandoned pre-run draft is excluded from E1–E3 results because it was not run. No E4 suite/result was present in the inspected evidence source.

This directory is only a local review bundle. The existing E3 manifest says its source transfer was private and no external publication step was authorized there; this bundle does not publish it or supersede that provenance. Review the content and permissions before copying it into a public repository.

