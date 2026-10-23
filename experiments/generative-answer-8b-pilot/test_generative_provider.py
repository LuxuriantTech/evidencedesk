from __future__ import annotations

import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

API_DIR = Path(__file__).resolve().parents[1] / "rule-candidate-scope" / "apps" / "api"
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidencedesk_api.answering import GroundedAnswer, validate_grounded_answer  # noqa: E402
from generative_provider import (  # noqa: E402
    ABSTENTION_TEXT,
    AMBIGUITY_PREFIX,
    CONFIG_FINGERPRINT,
    JSON_SCHEMA,
    MODE,
    GENERATION_CONFIG,
    RESPONSE_SCHEMA,
    ModelOutputValidationError,
    ProviderInputError,
    BackendRequestError,
    GenerativeAnswerProvider,
    _context,
    _user_payload,
    _validate_model_json,
)


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    page: int
    section: str | None
    text: str


@dataclass
class Ranked:
    chunk: Chunk


def c(chunk_id: str, text: str, document_id: str | None = None, page: int = 1) -> Ranked:
    return Ranked(Chunk(chunk_id, document_id or f"doc-{chunk_id}", f"{chunk_id}.pdf", page, None, text))


def response(content: str, *, finish_reason: str = "stop", reasoning_content: str = "") -> tuple[int, dict[str, str], bytes]:
    body = json.dumps({"choices": [{"finish_reason": finish_reason, "message": {
        "content": content, "reasoning_content": reasoning_content,
    }}]}).encode()
    return 200, {"content-type": "application/json"}, body


def fake_local_post(content: str, input_tokens: int = 128):
    def call(url, payload, timeout, *, finish_reason="stop", reasoning_content=""):
        if url.endswith("/input_tokens"):
            body = json.dumps({"input_tokens": input_tokens}).encode()
            return 200, {"content-type": "application/json"}, body
        return response(content, finish_reason=finish_reason, reasoning_content=reasoning_content)
    return call


def decision(status: str = "answered", evidence=None, spans=None) -> str:
    return json.dumps({
        "evidence": evidence if evidence is not None else [{"chunk_id": "c1", "quote": "Serial is SN-73."}],
        "answer_spans": spans if spans is not None else [{"chunk_id": "c1", "text": "SN-73"}],
        "status": status,
    })


class GenerativeProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = GenerativeAnswerProvider()

    def call_with_content(self, content: str, ranked=None):
        with patch("generative_provider._http_post", side_effect=fake_local_post(content)) as post:
            result = self.provider.answer("What is the serial?", ranked or [c("c1", "Serial is SN-73.")])
        return result, post

    def test_payload_preserves_each_chunk_field_and_records_native_rank(self):
        ranked = [
            Ranked(Chunk("z-last", "doc-z", "Zeta.pdf", 7, "Final", "Exact Z text.")),
            Ranked(Chunk("a-first", "doc-a", "Alpha.pdf", 2, "Intro", "Exact A text.")),
        ]
        payload = json.loads(_user_payload("question one", _context("question one", ranked)))
        self.assertEqual(list(payload), ["retrieved_chunks", "retrieval_order", "question"])
        self.assertEqual(payload["retrieval_order"], ["z-last", "a-first"])
        self.assertEqual(payload["question"], "question one")
        self.assertEqual(payload["retrieved_chunks"], [
            {"chunk_id": "a-first", "document_id": "doc-a", "document_name": "Alpha.pdf",
             "page": 2, "section": "Intro", "text": "Exact A text."},
            {"chunk_id": "z-last", "document_id": "doc-z", "document_name": "Zeta.pdf",
             "page": 7, "section": "Final", "text": "Exact Z text."},
        ])

    def test_same_chunk_pool_has_identical_payload_prefix_across_questions(self):
        ranked = [c("z-last", "Exact Z text."), c("a-first", "Exact A text.")]
        first = _user_payload("question one", _context("question one", ranked))
        second = _user_payload("a different question", _context("a different question", ranked))
        marker = ',"question":'
        self.assertIn(marker, first)
        self.assertIn(marker, second)
        self.assertEqual(first.split(marker, 1)[0], second.split(marker, 1)[0])

    def test_rank_permutation_keeps_chunk_blocks_but_updates_retrieval_order(self):
        first_rank = [c("z-last", "Exact Z text."), c("a-first", "Exact A text.")]
        second_rank = list(reversed(first_rank))
        first = json.loads(_user_payload("same question", _context("same question", first_rank)))
        second = json.loads(_user_payload("same question", _context("same question", second_rank)))
        self.assertEqual(first["retrieved_chunks"], second["retrieved_chunks"])
        self.assertEqual(first["retrieval_order"], ["z-last", "a-first"])
        self.assertEqual(second["retrieval_order"], ["a-first", "z-last"])

    def test_answer_is_exact_validated_span_and_native_type(self):
        result, post = self.call_with_content(decision())
        self.assertIsInstance(result, GroundedAnswer)
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer, "SN-73")
        self.assertEqual(result.citations[0].excerpt, "Serial is SN-73.")
        payload = post.call_args_list[1].args[1]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 20261023)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], JSON_SCHEMA)
        self.assertEqual(RESPONSE_SCHEMA, JSON_SCHEMA)
        self.assertEqual(GENERATION_CONFIG, {"seed": 20261023, "temperature": 0.0, "max_tokens": 512, "enable_thinking": False})
        self.assertEqual(self.provider.diagnostics[-1]["request_payload"], payload)
        self.assertIn("raw_response_body", self.provider.diagnostics[-1])
        self.assertEqual(self.provider.diagnostics[-1]["config_fingerprint"], CONFIG_FINGERPRINT)
        self.assertEqual(self.provider.diagnostics[-1]["input_tokens"], 128)
        self.assertEqual(self.provider.diagnostics[-1]["context_preflight"], "passed_exact_server_token_count")
        self.assertEqual(post.call_args_list[0].args[1], post.call_args_list[1].args[1])
        self.assertEqual(result.mode if hasattr(result, "mode") else self.provider.mode, MODE)
        self.assertEqual(
            [branch["properties"]["status"]["const"] for branch in JSON_SCHEMA["oneOf"]],
            ["answered", "ambiguous", "abstained"],
        )
        self.assertEqual(
            list(JSON_SCHEMA["oneOf"][0]["properties"]),
            ["status", "evidence", "answer_spans"],
        )

    def test_ambiguous_values_require_distinct_proof_quotes(self):
        ranked = [c("c1", "The fee was EUR 40, later EUR 55."), c("c2", "Account is active.")]
        content = decision(
            "ambiguous",
            [{"chunk_id": "c1", "quote": "The fee was EUR 40, later EUR 55."},
             {"chunk_id": "c2", "quote": "Account is active."}],
            [{"chunk_id": "c1", "text": "EUR 40"}, {"chunk_id": "c1", "text": "EUR 55"}],
        )
        result, _ = self.call_with_content(content, ranked)
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"],
                         "model_output_ambiguity_values_share_evidence")

    def test_ambiguous_requires_two_distinct_values_with_separate_evidence(self):
        content = decision(
            "ambiguous",
            [
                {"chunk_id": "c1", "quote": "The fee is EUR 40."},
                {"chunk_id": "c2", "quote": "The fee is EUR 55."},
            ],
            [
                {"chunk_id": "c1", "text": "EUR 40"},
                {"chunk_id": "c2", "text": "EUR 55"},
            ],
        )
        result, _ = self.call_with_content(content, [c("c1", "The fee is EUR 40."), c("c2", "The fee is EUR 55.")])
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.answer, AMBIGUITY_PREFIX + "EUR 40; EUR 55")
        self.assertEqual(len(result.citations), 2)
        self.assertFalse(result.answerable)
        validate_grounded_answer(result, [c("c1", "The fee is EUR 40."), c("c2", "The fee is EUR 55.")])

    def test_ambiguous_can_use_distinct_quotes_from_same_chunk(self):
        chunk = c("c1", "Draft fee: EUR 40. Signed fee: EUR 45.")
        content = decision(
            "ambiguous",
            [
                {"chunk_id": "c1", "quote": "Draft fee: EUR 40."},
                {"chunk_id": "c1", "quote": "Signed fee: EUR 45."},
            ],
            [
                {"chunk_id": "c1", "text": "EUR 40"},
                {"chunk_id": "c1", "text": "EUR 45"},
            ],
        )
        result, _ = self.call_with_content(content, [chunk])
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(len(result.citations), 2)
        validate_grounded_answer(result, [chunk])

    def test_ambiguous_same_value_is_rejected_and_diagnosed(self):
        content = decision(
            "ambiguous",
            [{"chunk_id": "c1", "quote": "Fee EUR 40."}, {"chunk_id": "c2", "quote": "Fee EUR 40."}],
            [{"chunk_id": "c1", "text": "EUR 40"}, {"chunk_id": "c2", "text": "EUR 40"}],
        )
        result, _ = self.call_with_content(content, [c("c1", "Fee EUR 40."), c("c2", "Fee EUR 40.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_output_ambiguity_contract")

    def test_abstention_has_no_spans_or_citations(self):
        content = decision("abstained", [], [])
        result, _ = self.call_with_content(content, [c("c1", "No serial number is listed.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.answer, ABSTENTION_TEXT)
        self.assertEqual(result.citations, ())
        validate_grounded_answer(result, [c("c1", "No serial number is listed.")])

    def test_answered_multiple_spans_in_same_quote_builds_minimal_contiguous_answer(self):
        chunk = c("c1", "Contact: Lea Martin. Start date: 1 May 2028. Phone: 555-0100.")
        content = decision(
            "answered",
            [{"chunk_id": "c1", "quote": "Contact: Lea Martin. Start date: 1 May 2028. Phone: 555-0100."}],
            [{"chunk_id": "c1", "text": "Lea Martin"}, {"chunk_id": "c1", "text": "1 May 2028"}],
        )
        result, _ = self.call_with_content(content, [chunk])
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer, "Lea Martin. Start date: 1 May 2028")
        validate_grounded_answer(result, [chunk])

    def test_answered_allows_corroborating_quotes_when_spans_share_one_quote(self):
        chunk = c("c1", "The account code is READY. Account code: READY.")
        content = decision(
            "answered",
            [{"chunk_id": "c1", "quote": "The account code is READY."},
             {"chunk_id": "c1", "quote": "Account code: READY."}],
            [{"chunk_id": "c1", "text": "READY"}],
        )
        result, _ = self.call_with_content(content, [chunk])
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer, "READY")
        self.assertEqual(len(result.citations), 2)
        validate_grounded_answer(result, [chunk])

    def test_answered_spans_in_separate_chunks_fail_closed(self):
        ranked = [c("c1", "Contact: Lea Martin."), c("c2", "Start date: 1 May 2028.")]
        content = decision(
            "answered",
            [{"chunk_id": "c1", "quote": "Contact: Lea Martin."}, {"chunk_id": "c2", "quote": "Start date: 1 May 2028."}],
            [{"chunk_id": "c1", "text": "Lea Martin"}, {"chunk_id": "c2", "text": "1 May 2028"}],
        )
        result, _ = self.call_with_content(content, ranked)
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_output_answer_spans_no_shared_quote")
        validate_grounded_answer(result, ranked)

    def test_fail_closed_output_errors_are_countable_not_silent(self):
        for content, expected in [
            ("not json", "model_output_invalid_json"),
            (json.dumps({"evidence": [], "answer_spans": [], "status": "abstained", "reason": "no_evidence"}), "model_output_unknown_or_missing_keys"),
            (decision(evidence=[{"chunk_id": "elsewhere", "quote": "Serial is SN-73."}]), "model_output_unknown_chunk_id"),
            (decision(evidence=[{"chunk_id": "c1", "quote": "Serial is SN-74."}]), "model_output_quote_not_exact_substring"),
            (decision(spans=[{"chunk_id": "c2", "text": "SN-73"}]), "model_output_unknown_span_chunk_id"),
        ]:
            with self.subTest(expected=expected):
                result, _ = self.call_with_content(content)
                self.assertEqual(result.status, "abstained")
                self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], expected)
                validate_grounded_answer(result, [c("c1", "Serial is SN-73.")])

    def test_duplicate_json_keys_are_rejected_at_every_object_level(self):
        responses = (
            '{"evidence":[{"chunk_id":"c1","quote":"Serial is SN-73."}],'
            '"answer_spans":[{"chunk_id":"c1","text":"SN-73"}],'
            '"status":"answered","status":"abstained"}',
            '{"evidence":[{"chunk_id":"c1","chunk_id":"c1","quote":"Serial is SN-73."}],'
            '"answer_spans":[{"chunk_id":"c1","text":"SN-73"}],"status":"answered"}',
        )
        for content in responses:
            with self.subTest(content=content):
                result, _ = self.call_with_content(content)
                self.assertEqual(result.status, "abstained")
                self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"],
                                 "model_output_duplicate_json_key")

    def test_parseable_response_with_length_finish_reason_fails_closed(self):
        content = decision()
        with patch("generative_provider._http_post", side_effect=[
            (200, {}, json.dumps({"input_tokens": 32}).encode()),
            response(content, finish_reason="length"),
        ]):
            result = self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_finish_reason_not_stop:'length'")
        validate_grounded_answer(result, [c("c1", "Serial is SN-73.")])

    def test_nonempty_reasoning_content_fails_closed(self):
        with patch("generative_provider._http_post", side_effect=[
            (200, {}, json.dumps({"input_tokens": 32}).encode()),
            response(decision(), reasoning_content="private reasoning"),
        ]):
            result = self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_reasoning_content_not_empty")
        validate_grounded_answer(result, [c("c1", "Serial is SN-73.")])

    def test_think_tags_in_content_fail_closed(self):
        with patch("generative_provider._http_post", side_effect=[
            (200, {}, json.dumps({"input_tokens": 32}).encode()),
            response("<think>hidden</think>" + decision()),
        ]):
            result = self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_think_tags_present")
        validate_grounded_answer(result, [c("c1", "Serial is SN-73.")])

    def test_duplicate_input_chunk_ids_rejected_before_backend(self):
        with patch("generative_provider._http_post") as post:
            with self.assertRaisesRegex(ProviderInputError, "duplicate_chunk_id"):
                self.provider.answer("q", [c("c1", "A."), c("c1", "B.")])
            post.assert_not_called()

    def test_remote_urls_and_retuned_configs_rejected(self):
        with self.assertRaises(ValueError):
            GenerativeAnswerProvider("https://example.com")
        with self.assertRaises(ValueError):
            GenerativeAnswerProvider("http://192.168.1.20:8080")
        with self.assertRaises(ValueError):
            GenerativeAnswerProvider(max_tokens=256)
        with self.assertRaises(ValueError):
            GenerativeAnswerProvider(enable_thinking=True)

    def test_backend_errors_propagate_and_are_recorded(self):
        with patch("generative_provider._http_post", side_effect=BackendRequestError("server down")):
            with self.assertRaises(BackendRequestError):
                self.provider.answer("q", [c("c1", "A.")])
        self.assertEqual(self.provider.diagnostics[-1]["preflight_error"], "server down")

    def test_exact_preflight_refuses_over_context_before_completion(self):
        content = decision()
        calls = []

        def post(url, payload, timeout):
            calls.append((url, payload))
            return 200, {}, json.dumps({"input_tokens": 5633}).encode()

        with patch("generative_provider._http_post", side_effect=post):
            with self.assertRaisesRegex(ProviderInputError, "prompt_exceeds_context_budget"):
                self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0].endswith("/v1/chat/completions/input_tokens"))
        self.assertEqual(self.provider.diagnostics[-1]["input_tokens"], 5633)
        self.assertEqual(self.provider.diagnostics[-1]["context_preflight"], "rejected_over_context_budget")

    def test_bad_preflight_count_is_backend_error_and_never_generates(self):
        calls = []

        def post(url, payload, timeout):
            calls.append(url)
            return 200, {}, json.dumps({"input_tokens": True}).encode()

        with patch("generative_provider._http_post", side_effect=post):
            with self.assertRaisesRegex(BackendRequestError, "invalid_input_tokens"):
                self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(len(calls), 1)
        self.assertIn("preflight_error_type", self.provider.diagnostics[-1])

    def test_rejects_duplicate_or_oversized_model_output(self):
        chunk = c("c1", "Serial SN-73.").chunk
        chunks = [type("Wrapped", (), {"chunk": chunk})()]
        duplicate = {
            "evidence": [{"chunk_id": "c1", "quote": "Serial SN-73."}] * 2,
            "answer_spans": [{"chunk_id": "c1", "text": "SN-73"}],
            "status": "answered",
        }
        with self.assertRaisesRegex(ModelOutputValidationError, "duplicate_evidence"):
            _validate_model_json(json.dumps(duplicate), [type("Simple", (), {"chunk_id": "c1", "text": chunk.text})()])

    def test_rejects_context_limits(self):
        with self.assertRaisesRegex(ProviderInputError, "oversize_question"):
            self.provider.answer("x" * 8001, [])
        with self.assertRaisesRegex(ProviderInputError, "too_many_chunks"):
            self.provider.answer("q", [c(f"c{i}", "t") for i in range(21)])




    def test_primary_anchor_wins_over_shorter_duplicate_value_in_another_source(self):
        ranked = [
            c("maintenance-log", "Maintenance log assigns device VX-22 to route R4."),
            c("calibration-card", "Calibration code R4."),
        ]
        content = json.dumps({
            "status": "answered",
            "answer_spans": [{"chunk_id": "maintenance-log", "text": "R4"}],
            "evidence": [
                {"chunk_id": "calibration-card", "quote": "Calibration code R4."},
                {"chunk_id": "maintenance-log", "quote": "Maintenance log assigns device VX-22 to route R4."},
            ],
        })
        result, _ = self.call_with_content(content, ranked)
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer, "R4")
        self.assertEqual(result.supporting_document, "doc-maintenance-log")
        self.assertEqual(result.supporting_excerpt, "Maintenance log assigns device VX-22 to route R4.")

    def test_answer_spans_from_different_chunks_fail_closed(self):
        ranked = [
            c("maintenance-log", "Unit VX-22 uses route R4."),
            c("work-order", "Work order for unit QZ-9 references route R4."),
        ]
        content = json.dumps({
            "status": "answered",
            "answer_spans": [
                {"chunk_id": "maintenance-log", "text": "R4"},
                {"chunk_id": "work-order", "text": "QZ-9"},
            ],
            "evidence": [
                {"chunk_id": "maintenance-log", "quote": "Unit VX-22 uses route R4."},
                {"chunk_id": "work-order", "quote": "Work order for unit QZ-9 references route R4."},
            ],
        })
        result, _ = self.call_with_content(content, ranked)
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_output_answer_spans_no_shared_quote")
        validate_grounded_answer(result, ranked)

    def test_spans_in_one_chunk_but_different_quotes_fail_closed(self):
        ranked = [c("maintenance-log", "Unit VX-22 uses route R4. Unit QZ-9 uses route B7.")]
        content = json.dumps({
            "status": "answered",
            "answer_spans": [
                {"chunk_id": "maintenance-log", "text": "R4"},
                {"chunk_id": "maintenance-log", "text": "QZ-9"},
            ],
            "evidence": [
                {"chunk_id": "maintenance-log", "quote": "Unit VX-22 uses route R4."},
                {"chunk_id": "maintenance-log", "quote": "Unit QZ-9 uses route B7."},
            ],
        })
        result, _ = self.call_with_content(content, ranked)
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_output_answer_spans_no_shared_quote")

    def test_multiple_spans_may_use_one_shared_primary_quote(self):
        quote = "Maintenance log assigns device VX-22 to route R4."
        ranked = [c("maintenance-log", quote), c("other-record", "VX-22 and R4.")]
        content = json.dumps({
            "status": "answered",
            "answer_spans": [
                {"chunk_id": "maintenance-log", "text": "VX-22"},
                {"chunk_id": "maintenance-log", "text": "R4"},
            ],
            "evidence": [
                {"chunk_id": "other-record", "quote": "VX-22 and R4."},
                {"chunk_id": "maintenance-log", "quote": quote},
            ],
        })
        result, _ = self.call_with_content(content, ranked)
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.supporting_document, "doc-maintenance-log")
        self.assertEqual(result.supporting_excerpt, quote)
        validate_grounded_answer(result, ranked)

if __name__ == "__main__":
    unittest.main()
