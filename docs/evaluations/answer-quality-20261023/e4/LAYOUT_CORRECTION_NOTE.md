# E4 layout correction

Root pre-registration stopped before creating a run directory and before any inference because the frozen package did not contain the common adapter and corpus source payloads. No model or adapter was executed.

The initial pre-layout freeze artifacts and draft status were moved intact to history/pre-layout/. They are retained as an unusable pre-layout record, not represented as an executed evaluation. The frozen suite and corpus were converted to the adapter's existing schema and placed with the byte-identical pinned adapter and all ten source payloads under inputs/. Case IDs, canonical answers, citation text and page numbers were preserved through the structural conversion. Every source_path resolves under inputs/ and every page string is present in its corresponding source payload.

Use the current FREEZE_RECEIPT.json and FROZEN_INPUT_MANIFEST.json. The archived STATUS_DRAFT.json is historical and is superseded by the current receipt.
