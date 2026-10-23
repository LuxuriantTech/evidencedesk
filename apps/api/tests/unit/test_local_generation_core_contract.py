# ruff: noqa: E501
from __future__ import annotations

import json
import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

from evidencedesk_api.answering import GroundedAnswer, validate_grounded_answer
from evidencedesk_api.local_generation_core import (
    ABSTENTION_TEXT,
    AMBIGUITY_PREFIX,
    CONFIG_FINGERPRINT,
    GENERATION_CONFIG,
    JSON_SCHEMA,
    MODE,
    RESPONSE_SCHEMA,
    BackendRequestError,
    GenerativeAnswerProvider,
    ModelOutputValidationError,
    ProviderInputError,
    _Chunk,
    _validate_model_json,
)
from evidencedesk_api.retrieval import EvidenceChunk, RankedChunk


def c(chunk_id: str, text: str, document_id: str | None = None, page: int = 1) -> RankedChunk:
    chunk = EvidenceChunk(
        id=chunk_id,
        document_id=document_id or f"doc-{chunk_id}",
        document_name=f"{chunk_id}.pdf",
        page=page,
        section=None,
        text=text,
        embedding=[],
    )
    return RankedChunk(
        chunk=chunk,
        score=1.0,
        lexical_score=1.0,
        dense_score=1.0,
        entity_score=0.0,
    )


def response(
    content: str, *, finish_reason: str = "stop", reasoning_content: str = ""
) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps({"model": "Qwen3-8B-Q4_K_M.gguf", "choices": [{"finish_reason": finish_reason, "message": {
        "content": content, "reasoning_content": reasoning_content,
    }}]}).encode()
    return 200, {"content-type": "application/json"}, body


def fake_local_post(
    content: str, input_tokens: int = 128
) -> Callable[..., tuple[int, dict[str, str], bytes]]:
    def call(
        url: str,
        payload: dict[str, Any],
        timeout: float,
        *,
        finish_reason: str = "stop",
        reasoning_content: str = "",
    ) -> tuple[int, dict[str, str], bytes]:
        if url.endswith("/input_tokens"):
            body = json.dumps({"input_tokens": input_tokens}).encode()
            return 200, {"content-type": "application/json"}, body
        return response(content, finish_reason=finish_reason, reasoning_content=reasoning_content)
    return call


def decision(
    status: str = "answered",
    evidence: list[dict[str, str]] | None = None,
    spans: list[dict[str, str]] | None = None,
) -> str:
    return json.dumps({
        "evidence": evidence if evidence is not None else [{"chunk_id": "c1", "quote": "Serial is SN-73."}],
        "answer_spans": spans if spans is not None else [{"chunk_id": "c1", "text": "SN-73"}],
        "status": status,
    })


class GenerativeProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = GenerativeAnswerProvider()

    def call_with_content(
        self, content: str, ranked: list[RankedChunk] | None = None
    ) -> tuple[GroundedAnswer, MagicMock]:
        with patch("evidencedesk_api.local_generation_core._http_post", side_effect=fake_local_post(content)) as post:
            result = self.provider.answer("What is the serial?", ranked or [c("c1", "Serial is SN-73.")])
        return result, post

    def test_answer_is_exact_validated_span_and_native_type(self) -> None:
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

    def test_ambiguous_values_require_distinct_proof_quotes(self) -> None:
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

    def test_ambiguous_requires_two_distinct_values_with_separate_evidence(self) -> None:
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

    def test_ambiguous_can_use_distinct_quotes_from_same_chunk(self) -> None:
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

    def test_ambiguous_same_value_is_rejected_and_diagnosed(self) -> None:
        content = decision(
            "ambiguous",
            [{"chunk_id": "c1", "quote": "Fee EUR 40."}, {"chunk_id": "c2", "quote": "Fee EUR 40."}],
            [{"chunk_id": "c1", "text": "EUR 40"}, {"chunk_id": "c2", "text": "EUR 40"}],
        )
        result, _ = self.call_with_content(content, [c("c1", "Fee EUR 40."), c("c2", "Fee EUR 40.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_output_ambiguity_contract")

    def test_abstention_has_no_spans_or_citations(self) -> None:
        content = decision("abstained", [], [])
        result, _ = self.call_with_content(content, [c("c1", "No serial number is listed.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.answer, ABSTENTION_TEXT)
        self.assertEqual(result.citations, ())
        validate_grounded_answer(result, [c("c1", "No serial number is listed.")])

    def test_answered_multiple_spans_in_same_quote_builds_minimal_contiguous_answer(self) -> None:
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

    def test_answered_allows_corroborating_quotes_when_spans_share_one_quote(self) -> None:
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

    def test_answered_spans_in_separate_chunks_fail_closed(self) -> None:
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

    def test_fail_closed_output_errors_are_countable_not_silent(self) -> None:
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

    def test_duplicate_json_keys_are_rejected_at_every_object_level(self) -> None:
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

    def test_parseable_response_with_length_finish_reason_fails_closed(self) -> None:
        content = decision()
        with patch("evidencedesk_api.local_generation_core._http_post", side_effect=[
            (200, {}, json.dumps({"input_tokens": 32}).encode()),
            response(content, finish_reason="length"),
        ]):
            result = self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_finish_reason_not_stop:'length'")
        validate_grounded_answer(result, [c("c1", "Serial is SN-73.")])

    def test_nonempty_reasoning_content_fails_closed(self) -> None:
        with patch("evidencedesk_api.local_generation_core._http_post", side_effect=[
            (200, {}, json.dumps({"input_tokens": 32}).encode()),
            response(decision(), reasoning_content="private reasoning"),
        ]):
            result = self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_reasoning_content_not_empty")
        validate_grounded_answer(result, [c("c1", "Serial is SN-73.")])

    def test_think_tags_in_content_fail_closed(self) -> None:
        with patch("evidencedesk_api.local_generation_core._http_post", side_effect=[
            (200, {}, json.dumps({"input_tokens": 32}).encode()),
            response("<think>hidden</think>" + decision()),
        ]):
            result = self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(self.provider.diagnostics[-1]["model_validation_reason"], "model_think_tags_present")
        validate_grounded_answer(result, [c("c1", "Serial is SN-73.")])

    def test_duplicate_input_chunk_ids_rejected_before_backend(self) -> None:
        with patch("evidencedesk_api.local_generation_core._http_post") as post:
            with self.assertRaisesRegex(ProviderInputError, "duplicate_chunk_id"):
                self.provider.answer("q", [c("c1", "A."), c("c1", "B.")])
            post.assert_not_called()

    def test_remote_urls_and_retuned_configs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GenerativeAnswerProvider("https://example.com")
        with self.assertRaises(ValueError):
            GenerativeAnswerProvider("http://192.168.1.20:8080")
        with self.assertRaises(ValueError):
            GenerativeAnswerProvider(max_tokens=256)
        with self.assertRaises(ValueError):
            GenerativeAnswerProvider(enable_thinking=True)

    def test_backend_errors_propagate_and_are_recorded(self) -> None:
        with patch("evidencedesk_api.local_generation_core._http_post", side_effect=BackendRequestError("server down")):
            with self.assertRaises(BackendRequestError):
                self.provider.answer("q", [c("c1", "A.")])
        self.assertEqual(self.provider.diagnostics[-1]["preflight_error"], "server down")

    def test_exact_preflight_refuses_over_context_before_completion(self) -> None:
        calls: list[Any] = []

        def post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, str], bytes]:
            calls.append((url, payload))
            return 200, {}, json.dumps({"input_tokens": 5633}).encode()

        with patch("evidencedesk_api.local_generation_core._http_post", side_effect=post):
            with self.assertRaisesRegex(ProviderInputError, "prompt_exceeds_context_budget"):
                self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0].endswith("/v1/chat/completions/input_tokens"))
        self.assertEqual(self.provider.diagnostics[-1]["input_tokens"], 5633)
        self.assertEqual(self.provider.diagnostics[-1]["context_preflight"], "rejected_over_context_budget")

    def test_bad_preflight_count_is_backend_error_and_never_generates(self) -> None:
        calls: list[Any] = []

        def post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, str], bytes]:
            calls.append(url)
            return 200, {}, json.dumps({"input_tokens": True}).encode()

        with patch("evidencedesk_api.local_generation_core._http_post", side_effect=post):
            with self.assertRaisesRegex(BackendRequestError, "invalid_input_tokens"):
                self.provider.answer("q", [c("c1", "Serial is SN-73.")])
        self.assertEqual(len(calls), 1)
        self.assertIn("preflight_error_type", self.provider.diagnostics[-1])

    def test_rejects_duplicate_or_oversized_model_output(self) -> None:
        chunk = c("c1", "Serial SN-73.").chunk
        duplicate = {
            "evidence": [{"chunk_id": "c1", "quote": "Serial SN-73."}] * 2,
            "answer_spans": [{"chunk_id": "c1", "text": "SN-73"}],
            "status": "answered",
        }
        with self.assertRaisesRegex(ModelOutputValidationError, "duplicate_evidence"):
            _validate_model_json(json.dumps(duplicate), [_Chunk(chunk_id="c1", document_id="doc-c1", document_name="c1.pdf", page=1, section=None, text=chunk.text)])

    def test_rejects_context_limits(self) -> None:
        with self.assertRaisesRegex(ProviderInputError, "oversize_question"):
            self.provider.answer("x" * 8001, [])
        with self.assertRaisesRegex(ProviderInputError, "too_many_chunks"):
            self.provider.answer("q", [c(f"c{i}", "t") for i in range(21)])


if __name__ == "__main__":
    unittest.main()
