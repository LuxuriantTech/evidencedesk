import pytest
from evidencedesk_api.extraction import extract_supplier_fields
from evidencedesk_api.retrieval import EvidenceChunk


def _chunk(identifier: str, text: str, page: int = 1) -> EvidenceChunk:
    return EvidenceChunk(
        id=identifier,
        document_id="novel-supplier",
        document_name="novel-supplier.md",
        page=page,
        section=None,
        text=text,
        embedding=[],
    )


def test_supplier_extraction_handles_generic_english_labels_and_prose() -> None:
    chunks = [
        _chunk(
            "metadata",
            "Vendor legal entity — Lumen Relay Services LLC\n"
            "Category: operational support schedule\n"
            "This schedule takes effect on 3 October 2027.\n"
            "Agreed total: 18,750 USD\n"
            "Accountable lead — Jo Ives",
        ),
        _chunk(
            "terms",
            "Renewal is scheduled for October 3, 2028.\n"
            "Lumen shall preserve incident records for 45 days.\n"
            "Exposure note: incomplete exports could delay investigations.",
            2,
        ),
    ]

    result = extract_supplier_fields(chunks)

    assert result.organization_name.value == "Lumen Relay Services LLC"
    assert result.document_type.value == "operational support schedule"
    assert result.effective_date.value == "3 October 2027"
    assert result.renewal_date.value == "October 3, 2028"
    assert result.important_amounts.value == ("18,750 USD",)
    assert result.obligations.value == ("Lumen shall preserve incident records for 45 days.",)
    assert result.responsible_people.value == ("Jo Ives",)
    assert result.risks.value == ("incomplete exports could delay investigations.",)

    source_by_page = {chunk.page: chunk.text for chunk in chunks}
    for field in result.as_mapping().values():
        if field.value:
            assert field.citations
            assert all(
                citation.excerpt in source_by_page[citation.page]
                for citation in field.citations
            )


def test_supplier_extraction_handles_french_aliases_symbols_and_modal_language() -> None:
    chunks = [
        _chunk(
            "metadata",
            "Fournisseur : Quartz Atelier GmbH\n"
            "Nature du document : accord de maintenance\n"
            "Prise d'effet : le 20 août 2027\n"
            "Forfait annuel : 24 600 €\n"
            "Contact opérationnel : Ana Miro",
        ),
        _chunk(
            "terms",
            "Le contrat sera reconduit le 20 août 2028.\n"
            "Quartz s'engage à chiffrer chaque sauvegarde.\n"
            "Risque : la rotation des clés dépend encore d'une validation manuelle.",
            2,
        ),
    ]

    result = extract_supplier_fields(chunks)

    assert result.organization_name.value == "Quartz Atelier GmbH"
    assert result.document_type.value == "accord de maintenance"
    assert result.effective_date.value == "20 août 2027"
    assert result.renewal_date.value == "20 août 2028"
    assert result.important_amounts.value == ("24 600 €",)
    assert result.obligations.value == ("Quartz s'engage à chiffrer chaque sauvegarde.",)
    assert result.responsible_people.value == ("Ana Miro",)
    assert result.risks.value == (
        "la rotation des clés dépend encore d'une validation manuelle.",
    )


def test_unlabelled_title_and_legal_entity_are_safe_fallbacks() -> None:
    chunks = [
        _chunk("title", "MASTER SUPPLY AGREEMENT\nNimbus Workshop Inc."),
        _chunk("dates", "The arrangement commences on January 12, 2027.", 2),
    ]

    result = extract_supplier_fields(chunks)

    assert result.organization_name.value == "Nimbus Workshop Inc."
    assert result.document_type.value == "MASTER SUPPLY AGREEMENT"
    assert result.effective_date.value == "January 12, 2027"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Annual fee: EUR 48,000", "EUR 48,000"),
        ("Annual fee: $48,000 USD", "$48,000 USD"),
        ("Forfait : 48 000 EUR", "48 000 EUR"),
        ("Budget: GBP 12,600", "GBP 12,600"),
        ("Licence annuelle : 12 600 €", "12 600 €"),
    ],
)
def test_amount_extraction_supports_common_currency_formats(line: str, expected: str) -> None:
    result = extract_supplier_fields([_chunk("amount", line)])

    assert result.important_amounts.value == (expected,)
    assert result.important_amounts.citations[0].excerpt == expected


def test_conflicting_prose_renewal_dates_remain_ambiguous_and_sourced() -> None:
    chunks = [
        _chunk("first", "The agreement renews on 12 May 2028."),
        _chunk("second", "Renouvellement prévu le 30 juin 2028.", 2),
    ]

    result = extract_supplier_fields(chunks)

    assert result.renewal_date.value is None
    assert result.risks.value
    assert len(result.risks.citations) == 2
