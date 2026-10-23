import unittest
import urllib.request
from typing import Any, cast
from unittest.mock import MagicMock, patch

from scripts.local_generation.http_qualification.smoke_http import (
    ASK_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    SmokeFailure,
    _NoRedirect,
    loopback_base,
    request_json,
    require_status,
    validate_abstention,
    validate_answered,
)


class SmokeContractTests(unittest.TestCase):
    def test_loopback_base_accepts_ephemeral_local_port(self) -> None:
        self.assertEqual(loopback_base("http://127.0.0.1:32768"), "http://127.0.0.1:32768")

    def test_loopback_base_rejects_remote_hosts_and_credentials(self) -> None:
        for value in (
            "http://example.com:8000",
            "http://user:secret@127.0.0.1:8000",
            "https://127.0.0.1:8000",
        ):
            with self.subTest(value=value), self.assertRaises(SmokeFailure):
                loopback_base(value)

    def test_success_requires_source_bound_citation(self) -> None:
        source = "Test-only synthetic record. Fixture-abc asset date 2032-03-17."
        payload: dict[str, Any] = {
            "status": "answered",
            "answerable": True,
            "answer": "2032-03-17",
            "supporting_document": "doc-a",
            "supporting_excerpt": "Fixture-abc asset date 2032-03-17.",
            "citations": [
                {
                    "document_id": "doc-a",
                    "page": 1,
                    "excerpt": "Fixture-abc asset date 2032-03-17.",
                }
            ],
        }
        validate_answered(payload, doc_id="doc-a", date="2032-03-17", source=source, step="test")
        payload["citations"][0]["document_id"] = "doc-b"
        with self.assertRaises(SmokeFailure):
            validate_answered(
                payload, doc_id="doc-a", date="2032-03-17", source=source, step="test"
            )

    def test_empty_evidence_requires_clean_abstention(self) -> None:
        validate_abstention(
            {
                "status": "abstained",
                "answerable": False,
                "citations": [],
                "supporting_document": None,
                "extracted_fields": {},
            }
        )
        with self.assertRaises(SmokeFailure):
            validate_abstention(
                {
                    "status": "answered",
                    "answerable": True,
                    "citations": [{"document_id": "foreign"}],
                    "supporting_document": "foreign",
                    "extracted_fields": {},
                }
            )

    def test_status_error_code_contract(self) -> None:
        self.assertEqual(require_status((200, []), 200, "list"), [])
        with self.assertRaises(SmokeFailure):
            require_status(
                (503, {"detail": {"code": "internal"}}),
                503,
                "unavailable",
                "local_model_unavailable",
            )

    def test_request_uses_explicit_long_ask_timeout(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"ok":true}'
        response.status = 200
        opener = MagicMock()
        opener.open.return_value = response
        with patch(
            "scripts.local_generation.http_qualification.smoke_http.urllib.request.build_opener",
            return_value=opener,
        ) as build:
            result = request_json(
                "http://127.0.0.1:8000",
                "/ask",
                timeout=ASK_REQUEST_TIMEOUT_SECONDS,
                json_body={"question": "synthetic"},
                step="ask",
            )
        self.assertEqual(result, (200, {"ok": True}))
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 240.0)
        self.assertEqual(DEFAULT_REQUEST_TIMEOUT_SECONDS, 20.0)
        handlers = build.call_args.args
        self.assertTrue(
            any(
                isinstance(item, urllib.request.ProxyHandler) and cast(Any, item).proxies == {}
                for item in handlers
            )
        )
        self.assertTrue(any(isinstance(item, _NoRedirect) for item in handlers))

    def test_ordinary_request_keeps_short_default_timeout(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"ok":true}'
        response.status = 200
        opener = MagicMock()
        opener.open.return_value = response
        with patch(
            "scripts.local_generation.http_qualification.smoke_http.urllib.request.build_opener",
            return_value=opener,
        ):
            request_json("http://127.0.0.1:8000", "/health", step="health")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_redirect_is_not_followed(self) -> None:
        handler = _NoRedirect()
        self.assertIsNone(
            handler.redirect_request(None, None, 302, "Found", {}, "https://example.com/collect")
        )

    def test_request_timeout_is_bounded(self) -> None:
        with self.assertRaises(SmokeFailure):
            request_json("http://127.0.0.1:8000", "/ask", timeout=301, step="ask")


if __name__ == "__main__":
    unittest.main()
