# Public-safe dataset review

The six staged E4/E5/E6 JSON files are byte-for-byte matches for their retained
development inputs. Their SHA-256 digests and sizes are recorded in
`GENERATION_DATASET_MANIFEST.json`; all six comparisons returned an exact byte
match.

Each corpus manifest declares `license: CC0-1.0` and `synthetic_only: true`;
each evaluation file also declares `synthetic_only: true`. The source pages
contain invented organizations, identifiers, records, and contact addresses
using the reserved `.example` domain. No real URLs, personal contact domains,
credentials, API-key patterns, or private filesystem paths were found in the
staged JSON. The content includes ordinary fictional business instructions
(for example, a customer service procedure); these are source data, not active
instructions to the CI workflow or reviewer.

The E4 and E5 protocols classify those corpora as exposed development material;
E6 is labeled an internal synthetic, single-use comparison. These data are
therefore suitable only for experimental CI and must not be presented as fresh,
holdout, or independent validation. The review verifies the files' declared
synthetic/CC0 metadata and exact provenance; it is not an independent legal
determination of authorship or licensing.
