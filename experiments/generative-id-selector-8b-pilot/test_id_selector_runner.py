from dataclasses import dataclass
import unittest
from pathlib import Path
import run_id_selector_case as runner

from run_id_selector_case import (durable_selector_record, request_schema_fingerprint,
                                  verify_selector_diagnostic)


def payload(ids):
    schema = {"oneOf": [{"type": "object", "properties": {
        "status": {"const": "answered"},
        "primary_chunk_id": {"enum": list(ids)},
    }}]}
    return {"response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}


class SelectorRunnerReceiptTests(unittest.TestCase):
    def test_snapshot_tracks_existing_selector_source_files(self):
        files = runner.pilot_source_files()
        self.assertEqual(len(files), 5)
        self.assertEqual(len(set(files)), 5)
        self.assertIn(Path(runner.__file__).resolve(), files)
        for path in files:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())

    def test_dynamic_schema_fingerprint_is_per_request(self):
        first = payload(["chunk-a", "chunk-b"])
        second = payload(["chunk-b", "chunk-c"])
        self.assertEqual(request_schema_fingerprint(first), request_schema_fingerprint(first))
        self.assertNotEqual(request_schema_fingerprint(first), request_schema_fingerprint(second))

    def test_declared_fingerprint_must_match_exact_payload_schema(self):
        request = payload(["chunk-a", "chunk-b"])
        schema_hash = request_schema_fingerprint(request)
        diagnostic = {
            "protocol_version": "id-selector-v1",
            "schema_fingerprint_algorithm": "sha256-canonical-json-v1",
            "request_payload": request,
            "schema_fingerprint": schema_hash,
            "model_selection_receipt": {"protocol_version": "id-selector-v1",
                "schema_fingerprint_algorithm": "sha256-canonical-json-v1",
                "schema_fingerprint": schema_hash,
                "status": "answered", "model_selected_chunk_ids": ["chunk-a"]},
            "model_selected_chunk_ids": ["chunk-a"],
            "constructed_citation_chunk_ids": ["chunk-a"],
        }
        self.assertEqual(verify_selector_diagnostic(diagnostic)["status"], "verified")
        diagnostic["schema_fingerprint"] = request_schema_fingerprint(payload(["chunk-z"]))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_selector_diagnostic(diagnostic)

    def test_model_selection_and_derived_native_answer_are_separate(self):
        @dataclass
        class NativeAnswer:
            answer: str
            citations: tuple
        selection = {"status": "answered", "model_selected_chunk_ids": ["chunk-a"]}
        diagnostic = {"model_selection_receipt": selection}
        record = durable_selector_record(1, "question", NativeAnswer("source text", ("citation",)),
                                         diagnostic, {"status": "verified"})
        self.assertEqual(record["model_selection"], selection)
        self.assertEqual(record["response"]["answer"], "source text")
        self.assertIn("reconstructed", record["response_semantics"])
        self.assertEqual(record["native_derived_answer"]["answer"], "source text")
        self.assertNotEqual(record["model_selection"], record["native_derived_answer"])


if __name__ == "__main__":
    unittest.main()
