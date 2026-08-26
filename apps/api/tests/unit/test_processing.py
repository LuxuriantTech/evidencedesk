import io
import os

import pytest
from evidencedesk_api.processing import (
    DocumentParseError,
    ParsedPage,
    _parse_pdf_pages,
    chunk_pages,
    parse_document,
)
from evidencedesk_api.redaction import redact_pii
from reportlab.pdfgen import canvas


def test_synthetic_pii_is_redacted_with_counts() -> None:
    source = "Contact Mina Example at mina@example.test or +32 470 12 34 56. Reference SYN-ID-4829."

    result = redact_pii(source)

    assert "mina@example.test" not in result.text
    assert "+32 470 12 34 56" not in result.text
    assert "SYN-ID-4829" not in result.text
    assert result.text == (
        "Contact Mina Example at [EMAIL REDACTED] or [PHONE REDACTED]. "
        "Reference [IDENTIFIER REDACTED]."
    )
    assert result.counts == {"email": 1, "phone": 1, "identifier": 1}


def test_dates_times_and_amounts_are_not_misclassified_as_phone_numbers() -> None:
    source = "Effective date: 2026-01-15. Window: 09:14 to 09:32. Fee: EUR 48,000."

    result = redact_pii(source)

    assert result.text == source
    assert result.counts == {}


def test_markdown_parser_preserves_headings_as_sections() -> None:
    pages = parse_document(
        b"# Northwind dossier\n\n## Renewal\n\nRenewal date: 2027-05-31.\n",
        media_type="text/markdown",
    )

    assert pages == [
        ParsedPage(
            page=1,
            blocks=(("Northwind dossier", "Northwind dossier"),),
        ),
        ParsedPage(page=2, blocks=(("Renewal", "Renewal date: 2027-05-31."),)),
    ]


def test_plain_text_paragraphs_are_addressable_as_logical_pages() -> None:
    pages = parse_document(
        b"Supplier register\nEffective date: 2026-03-01\n\n"
        b"Renewal date: not recorded.\nRisk: renewal date is missing.",
        media_type="text/plain",
    )

    assert pages == [
        ParsedPage(
            page=1,
            blocks=((None, "Supplier register\nEffective date: 2026-03-01"),),
        ),
        ParsedPage(
            page=2,
            blocks=((None, "Renewal date: not recorded.\nRisk: renewal date is missing."),),
        ),
    ]


@pytest.mark.parametrize(
    ("data", "max_pages", "error_code"),
    [
        (b"\xff", 2, "invalid_text"),
        (b"  \n", 2, "no_extractable_text"),
        (b"Page one\n\nPage two", 1, "too_many_pages"),
    ],
)
def test_text_parser_rejects_invalid_empty_or_excessive_pages(
    data: bytes, max_pages: int, error_code: str
) -> None:
    with pytest.raises(DocumentParseError, match=rf"^{error_code}$"):
        parse_document(data, media_type="text/plain", max_pages=max_pages)


def test_chunking_never_crosses_page_boundaries_and_keeps_exact_text() -> None:
    pages = [
        ParsedPage(page=1, blocks=(("Scope", "Alpha obligation. Beta obligation."),)),
        ParsedPage(page=2, blocks=(("Renewal", "Renewal date: 2027-05-31."),)),
    ]

    chunks = chunk_pages(pages, max_chars=24)

    assert [chunk.page for chunk in chunks] == [1, 1, 2, 2]
    assert [chunk.section for chunk in chunks] == ["Scope", "Scope", "Renewal", "Renewal"]
    assert "Alpha obligation." in chunks[0].text
    assert "Beta obligation." in chunks[1].text
    assert all(len(chunk.text) <= 24 for chunk in chunks)


def test_chunking_respects_non_empty_line_boundaries() -> None:
    chunks = chunk_pages(
        [
            ParsedPage(
                page=1,
                blocks=((None, "Effective date: 2026-01-15\nAnnual fee: EUR 48,000"),),
            )
        ]
    )

    assert [chunk.text for chunk in chunks] == [
        "Effective date: 2026-01-15",
        "Annual fee: EUR 48,000",
    ]


def test_parser_rejects_text_beyond_the_extracted_character_budget() -> None:
    with pytest.raises(DocumentParseError, match=r"^too_much_extracted_text$"):
        parse_document(
            b"Organization: Synthetic Supplier Ltd.",
            media_type="text/plain",
            max_extracted_chars=16,
        )


def test_parser_rejects_pdf_beyond_the_page_budget() -> None:
    output = io.BytesIO()
    document = canvas.Canvas(output)
    document.drawString(72, 720, "Synthetic page one")
    document.showPage()
    document.drawString(72, 720, "Synthetic page two")
    document.save()

    with pytest.raises(DocumentParseError, match=r"^too_many_pages$"):
        parse_document(output.getvalue(), media_type="application/pdf", max_pages=1)


def test_pdf_parser_core_enforces_limits_before_returning_pages() -> None:
    output = io.BytesIO()
    document = canvas.Canvas(output)
    document.drawString(72, 720, "Synthetic page one")
    document.showPage()
    document.drawString(72, 720, "Synthetic page two")
    document.save()
    data = output.getvalue()

    pages = _parse_pdf_pages(data, max_pages=2, max_extracted_chars=100)
    assert [page.page for page in pages] == [1, 2]

    with pytest.raises(DocumentParseError, match=r"^too_many_pages$"):
        _parse_pdf_pages(data, max_pages=1, max_extracted_chars=100)
    with pytest.raises(DocumentParseError, match=r"^too_much_extracted_text$"):
        _parse_pdf_pages(data, max_pages=2, max_extracted_chars=5)
    with pytest.raises(DocumentParseError, match=r"^invalid_pdf$"):
        _parse_pdf_pages(b"not a pdf", max_pages=2, max_extracted_chars=100)


def test_pdf_parser_is_terminated_after_its_deadline() -> None:
    output = io.BytesIO()
    document = canvas.Canvas(output)
    document.drawString(72, 720, "Synthetic page")
    document.save()

    with pytest.raises(DocumentParseError, match=r"^pdf_processing_timeout$"):
        parse_document(
            output.getvalue(),
            media_type="application/pdf",
            pdf_timeout_seconds=0.001,
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX address-space limits are unavailable")
def test_pdf_parser_reports_a_hard_memory_limit() -> None:
    output = io.BytesIO()
    document = canvas.Canvas(output)
    document.setPageCompression(1)
    document.drawString(72, 720, "A" * 8_000_000)
    document.save()

    with pytest.raises(DocumentParseError, match=r"^pdf_resource_limit$"):
        parse_document(
            output.getvalue(),
            media_type="application/pdf",
            pdf_memory_bytes=64 * 1024 * 1024,
        )


def test_chunker_rejects_documents_beyond_the_chunk_budget() -> None:
    pages = [ParsedPage(page=1, blocks=((None, "First sentence. Second sentence."),))]

    with pytest.raises(DocumentParseError, match=r"^too_many_chunks$"):
        chunk_pages(pages, max_chars=20, max_chunks=1)


def test_chunker_bounds_a_single_unbroken_token() -> None:
    chunks = chunk_pages(
        [ParsedPage(page=1, blocks=((None, "A" * 25),))],
        max_chars=10,
    )

    assert [chunk.text for chunk in chunks] == ["A" * 10, "A" * 10, "A" * 5]
