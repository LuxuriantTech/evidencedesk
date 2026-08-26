from evidencedesk_api.processing import ParsedPage, chunk_pages, parse_document
from evidencedesk_api.redaction import redact_pii


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
