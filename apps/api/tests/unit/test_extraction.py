from evidencedesk_api.extraction import extract_supplier_fields
from evidencedesk_api.retrieval import EvidenceChunk


def _chunk(identifier: str, text: str, page: int) -> EvidenceChunk:
    return EvidenceChunk(
        id=identifier,
        document_id="doc-1",
        document_name="northwind.md",
        page=page,
        section="Metadata",
        text=text,
        embedding=[],
    )


def test_supplier_extraction_links_each_value_to_exact_evidence() -> None:
    chunks = [
        _chunk(
            "meta",
            "Organization: Northwind Response Systems Ltd.\n"
            "Document type: Supplier Agreement\n"
            "Effective date: 2026-06-01\n"
            "Renewal date: 2027-05-31",
            1,
        ),
        _chunk(
            "terms",
            "Annual service fee: EUR 84,000.\n"
            "Obligation: The supplier must notify incidents within four hours.\n"
            "Responsible manager: Mina Example.\n"
            "Risk: Renewal notice differs from the addendum.",
            2,
        ),
    ]

    result = extract_supplier_fields(chunks)

    assert result.organization_name.value == "Northwind Response Systems Ltd."
    assert result.document_type.value == "Supplier Agreement"
    assert result.effective_date.value == "2026-06-01"
    assert result.renewal_date.value == "2027-05-31"
    assert result.important_amounts.value == ("EUR 84,000",)
    assert result.obligations.value == ("The supplier must notify incidents within four hours.",)
    assert result.responsible_people.value == ("Mina Example",)
    assert result.risks.value == ("Renewal notice differs from the addendum.",)

    for field in result.as_mapping().values():
        if field.value:
            assert field.citations
            assert all(
                citation.excerpt in chunks[citation.page - 1].text for citation in field.citations
            )


def test_conflicting_renewal_dates_are_exposed_as_a_risk() -> None:
    chunks = [
        _chunk("main", "Renewal date: 2027-05-31", 1),
        _chunk("addendum", "Renewal date: 2027-06-30", 2),
    ]

    result = extract_supplier_fields(chunks)

    assert result.renewal_date.value is None
    assert result.risks.value == ("Conflicting renewal dates: 2027-05-31, 2027-06-30",)
    assert len(result.risks.citations) == 2


def test_unlabelled_contract_language_is_extracted_with_evidence() -> None:
    chunks = [
        _chunk("title", "MASTER SERVICES AGREEMENT", 1),
        _chunk("party", "Northstar Logistics Systems Ltd.", 1),
        _chunk(
            "renewal",
            "The agreement renews on 2027-01-15 unless either party gives 60 days written notice.",
            2,
        ),
        _chunk(
            "notify",
            "Northstar shall notify Meridian of a confirmed security incident within 24 hours.",
            2,
        ),
        _chunk("owner", "Service owner: Mira Talen, [EMAIL REDACTED].", 2),
        _chunk(
            "subcontractor",
            "No subcontractor may access production data without written approval.",
            2,
        ),
    ]

    result = extract_supplier_fields(chunks)

    assert result.organization_name.value == "Northstar Logistics Systems Ltd."
    assert result.renewal_date.value == "2027-01-15"
    assert result.responsible_people.value == ("Mira Talen",)
    assert result.obligations.value == (
        "Northstar shall notify Meridian of a confirmed security incident within 24 hours.",
        "No subcontractor may access production data without written approval.",
    )
    assert all(
        result.as_mapping()[name].citations
        for name in (
            "organization_name",
            "renewal_date",
            "responsible_people",
            "obligations",
        )
    )
