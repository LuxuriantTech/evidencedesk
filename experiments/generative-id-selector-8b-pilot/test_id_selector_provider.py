"""Transport-mocked tests against the actual frozen EvidenceDesk API types."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from evidencedesk_api.answering import GroundedAnswer, validate_grounded_answer
from evidencedesk_api.retrieval import EvidenceChunk, RankedChunk

import generative_provider as provider_module


QUESTION = "When does Riverside station open?"


def ranked_chunks(ids=("chunk-a", "chunk-b", "chunk-c")) -> list[RankedChunk]:
    records = [
        ("doc-a", "Field Notes", 2, "Summary", "Riverside station opens at 08:30; entrance is on Oak Lane."),
        ("doc-b", "Travel Memo", 4, "Schedule", "Riverside station opens at 09:00 on public holidays."),
        ("doc-c", "Workshop Log", 1, None, "The blue cabinet contains spare paper and pencils."),
    ]
    result = []
    for index, (chunk_id, (document_id, name, page, section, text)) in enumerate(zip(ids, records)):
        chunk = EvidenceChunk(id=chunk_id, document_id=document_id, document_name=name,
            page=page, section=section, text=text, embedding=[])
        result.append(RankedChunk(chunk=chunk, score=1.0 - index * 0.1,
            lexical_score=0.5, dense_score=0.5, entity_score=0.0))
    return result


def completion(decision: dict | None = None, *, finish_reason="stop", content=None) -> bytes:
    if content is None:
        content = json.dumps(decision)
    return json.dumps({"choices": [{"message": {"content": content},
                                      "finish_reason": finish_reason}]}).encode()


class IdSelectorProviderTests(unittest.TestCase):
    def setUp(self):
        self.ranked = ranked_chunks()
        self.provider = provider_module.GenerativeAnswerProvider()

    def post(self, decision, *, input_tokens=42, finish_reason="stop", fail_at=None):
        body = completion(decision, finish_reason=finish_reason)

        def fake_post(url, payload, timeout):
            if fail_at == "preflight":
                raise provider_module.BackendRequestError("mock_preflight_transport")
            if url.endswith("/input_tokens"):
                return 200, {}, json.dumps({"input_tokens": input_tokens}).encode()
            request_body = json.loads(payload["messages"][1]["content"])
            self.assertEqual(request_body["retrieval_order"], [item.chunk.id for item in self.ranked])
            expected_chunks = [{
                "chunk_id": item.chunk.id, "document_id": item.chunk.document_id,
                "document_name": item.chunk.document_name, "page": item.chunk.page,
                "section": item.chunk.section, "text": item.chunk.text,
            } for item in sorted(self.ranked, key=lambda item: item.chunk.id)]
            self.assertEqual(request_body["retrieved_chunks"], expected_chunks)
            if fail_at == "generation":
                raise provider_module.BackendRequestError("mock_generation_transport")
            return 200, {}, body
        return fake_post

    def answer(self, decision, **kwargs):
        with patch.object(provider_module, "_http_post", self.post(decision, **kwargs)):
            return self.provider.answer(QUESTION, self.ranked)

    def assert_native(self, result):
        self.assertIs(type(result), GroundedAnswer)
        validate_grounded_answer(result, self.ranked)

    def test_answer_is_full_exact_primary_and_citations_only_selected(self):
        result = self.answer({"status": "answered", "primary_chunk_id": "chunk-a",
                              "corroborating_chunk_ids": ["chunk-c"]})
        self.assert_native(result)
        self.assertEqual(result.answer, self.ranked[0].chunk.text)
        self.assertEqual(result.supporting_excerpt, self.ranked[0].chunk.text)
        self.assertEqual((result.supporting_document, result.supporting_page), ("doc-a", 2))
        self.assertEqual([c.chunk_id for c in result.citations], ["chunk-a", "chunk-c"])
        self.assertEqual([c.excerpt for c in result.citations], [self.ranked[0].chunk.text, self.ranked[2].chunk.text])
        self.assertNotIn("09:00", result.answer)
        diag = self.provider.diagnostics[-1]
        self.assertEqual(diag["model_selected_chunk_ids"], ["chunk-a", "chunk-c"])
        self.assertEqual(diag["constructed_citation_chunk_ids"], ["chunk-a", "chunk-c"])
        self.assertEqual(diag["model_selection_receipt"]["primary_chunk_id"], "chunk-a")
        self.assertEqual(diag["model_selection_receipt"]["corroborating_chunk_ids"], ["chunk-c"])
        self.assertEqual(diag["model_selection_receipt"]["schema_fingerprint"], diag["schema_fingerprint"])
        self.assertEqual(diag["confidence_is_calibrated"], False)

    def test_ambiguous_output_uses_two_exact_distinct_chunks(self):
        result = self.answer({"status": "ambiguous", "competing_chunk_ids": ["chunk-a", "chunk-b"]})
        self.assert_native(result)
        self.assertEqual(result.answer, provider_module.AMBIGUITY_PREFIX + "; ".join(
            item.chunk.text for item in self.ranked[:2]))
        self.assertEqual([c.chunk_id for c in result.citations], ["chunk-a", "chunk-b"])

    def test_abstention_is_empty_and_native_valid(self):
        result = self.answer({"status": "abstained"})
        self.assert_native(result)
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.citations, ())
        self.assertEqual(result.supporting_chunk_ids, ())

    def test_unknown_and_duplicate_selections_fail_closed(self):
        bad = [
            {"status": "answered", "primary_chunk_id": "outside", "corroborating_chunk_ids": []},
            {"status": "answered", "primary_chunk_id": "chunk-a", "corroborating_chunk_ids": ["chunk-a"]},
            {"status": "ambiguous", "competing_chunk_ids": ["chunk-a", "chunk-a"]},
        ]
        for decision in bad:
            result = self.answer(decision)
            self.assert_native(result)
            self.assertEqual(result.status, "abstained")
            self.assertEqual(result.citations, ())

    def test_dynamic_schema_is_closed_and_fingerprint_matches_exact_payload(self):
        result = self.answer({"status": "abstained"})
        self.assert_native(result)
        diagnostic = self.provider.diagnostics[-1]
        schema = diagnostic["request_payload"]["response_format"]["json_schema"]["schema"]
        self.assertEqual(diagnostic["schema_fingerprint"], provider_module._canonical_hash(schema))
        self.assertEqual(diagnostic["schema_fingerprint_algorithm"], "sha256-canonical-json-v1")
        self.assertEqual(diagnostic["protocol_version"], "id-selector-v1")
        ids = {item.chunk.id for item in self.ranked}
        for branch in schema["oneOf"]:
            self.assertFalse(branch["additionalProperties"])
            for key, definition in branch["properties"].items():
                if key.endswith("_chunk_id"):
                    self.assertEqual(set(definition["enum"]), ids)
                if key.endswith("_chunk_ids"):
                    self.assertEqual(set(definition["items"]["enum"]), ids)

        second = provider_module._selection_schema(provider_module._context(
            QUESTION, ranked_chunks(("another-a", "another-b"))))
        self.assertNotEqual(diagnostic["schema_fingerprint"], provider_module._canonical_hash(second))
        self.assertEqual(provider_module.response_schema_for(QUESTION, self.ranked), schema)

    def test_single_chunk_cannot_represent_internal_conflict(self):
        one = ranked_chunks(("solo",))
        branches = provider_module._selection_schema(provider_module._context(QUESTION, one))["oneOf"]
        self.assertEqual([b["properties"]["status"]["enum"][0] for b in branches], ["answered", "abstained"])
        self.assertNotIn("competing_chunk_ids", json.dumps(branches))
        # A conflict contained within one source chunk cannot be encoded by this protocol;
        # the prompt directs the model to abstain instead of mislabeling that case.

    def test_exact_input_preflight_boundary_5632_and_5633(self):
        accepted = self.answer({"status": "abstained"}, input_tokens=5632)
        self.assert_native(accepted)
        self.assertEqual(self.provider.diagnostics[-1]["context_preflight"], "passed_exact_server_token_count")

        fresh = provider_module.GenerativeAnswerProvider()
        with patch.object(provider_module, "_http_post", self.post({"status": "abstained"}, input_tokens=5633)):
            with self.assertRaises(provider_module.ProviderInputError):
                fresh.answer(QUESTION, self.ranked)
        self.assertEqual(fresh.diagnostics[-1]["context_preflight"], "rejected_over_context_budget")
        self.assertEqual(fresh.diagnostics[-1]["input_tokens"], 5633)

    def test_transport_errors_raise_and_keep_diagnostics(self):
        with patch.object(provider_module, "_http_post", self.post({"status": "abstained"}, fail_at="preflight")):
            with self.assertRaises(provider_module.BackendRequestError):
                self.provider.answer(QUESTION, self.ranked)
        diagnostic = self.provider.diagnostics[-1]
        self.assertEqual(diagnostic["preflight_error_type"], "BackendRequestError")
        self.assertEqual(diagnostic["preflight_error"], "mock_preflight_transport")

        self.provider = provider_module.GenerativeAnswerProvider()
        with patch.object(provider_module, "_http_post", self.post({"status": "abstained"}, fail_at="generation")):
            with self.assertRaises(provider_module.BackendRequestError):
                self.provider.answer(QUESTION, self.ranked)
        diagnostic = self.provider.diagnostics[-1]
        self.assertEqual(diagnostic["backend_error_type"], "BackendRequestError")
        self.assertEqual(diagnostic["backend_error"], "mock_generation_transport")

    def test_generation_envelope_bad_finish_reason_fails_closed(self):
        result = self.answer({"status": "answered", "primary_chunk_id": "chunk-a",
                              "corroborating_chunk_ids": []}, finish_reason="length")
        self.assert_native(result)
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["decision"], "fail_closed_abstention")

    def test_original_local_endpoint_and_context_guards_remain(self):
        with self.assertRaises(ValueError):
            provider_module.GenerativeAnswerProvider(base_url="https://example.org")
        with self.assertRaises(provider_module.ProviderInputError):
            provider_module._context("  ", self.ranked)
        duplicate = ranked_chunks(("same", "same"))
        with self.assertRaises(provider_module.ProviderInputError):
            provider_module._context(QUESTION, duplicate)
        too_many = self.ranked * (provider_module.MAX_CHUNKS // len(self.ranked) + 1)
        with self.assertRaises(provider_module.ProviderInputError):
            provider_module._context(QUESTION, too_many)


if __name__ == "__main__":
    unittest.main()
