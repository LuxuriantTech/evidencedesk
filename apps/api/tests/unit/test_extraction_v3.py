from evidencedesk_api.extraction import extract_supplier_fields_v3
from evidencedesk_api.providers import DeterministicEmbeddingProvider
from evidencedesk_api.retrieval import EvidenceChunk


def _chunk(identifier: str, text: str, page: int) -> EvidenceChunk:
    return EvidenceChunk(
        id=identifier,
        document_id="doc",
        document_name="handover.txt",
        page=page,
        section=None,
        text=text,
        embedding=[0.0] * 384,
    )


def test_semantic_extraction_handles_relational_prose_without_labels() -> None:
    chunks = [
        _chunk(
            "p1",
            "\n".join(
                [
                    "CONTINUITY HANDOVER FILE",
                    "The custodian covered here is Kestrel Archive Guild.",
                    "At sunrise on 12 January 2028, custody passes to the guild.",
                    "The next custody review is scheduled for 12 January 2029.",
                    "Lina Quill owns triage; Oren Bask owns restoration.",
                ]
            ),
            1,
        ),
        _chunk(
            "p2",
            "\n".join(
                [
                    "The approved continuity pool is £46,000 plus EUR 3,600 for storage media.",
                    "The guild preserves an offline index after every archive change.",
                    "A flood in the single vault could affect both media copies.",
                ]
            ),
            2,
        ),
    ]

    result = extract_supplier_fields_v3(
        chunks,
        embeddings=DeterministicEmbeddingProvider(dimension=384),
    )

    assert result.organization_name.value == "Kestrel Archive Guild"
    assert result.document_type.value == "CONTINUITY HANDOVER FILE"
    assert result.effective_date.value == "12 January 2028"
    assert result.renewal_date.value == "12 January 2029"
    assert result.important_amounts.value == ("£46,000", "EUR 3,600")
    assert result.responsible_people.value == ("Lina Quill", "Oren Bask")
    assert any("offline index" in value for value in result.obligations.value or ())
    assert any("single vault" in value for value in result.risks.value or ())
    for field in result.as_mapping().values():
        if field.value not in (None, ()):
            assert field.citations


def test_semantic_extraction_preserves_explicit_contract_fields_and_modals() -> None:
    chunks = [
        _chunk(
            "contract-p1",
            "\n".join(
                [
                    "MASTER SERVICES AGREEMENT",
                    "Northstar Logistics Systems Ltd.",
                    "Document type: Supplier master services agreement",
                    "Effective date: 2026-01-15",
                    "Annual platform fee: EUR 48,000",
                    "The provider shall maintain an operational service desk.",
                ]
            ),
            1,
        ),
        _chunk(
            "contract-p2",
            "\n".join(
                [
                    "The agreement renews on 2027-01-15 unless notice is given.",
                    "Northstar must notify the customer within 24 hours.",
                    "Service owner: Mira Talen, [EMAIL REDACTED].",
                    "No subcontractor may access production data without approval.",
                ]
            ),
            2,
        ),
    ]

    result = extract_supplier_fields_v3(
        chunks,
        embeddings=DeterministicEmbeddingProvider(dimension=384),
    )

    assert result.organization_name.value == "Northstar Logistics Systems Ltd."
    assert result.effective_date.value == "2026-01-15"
    assert result.renewal_date.value == "2027-01-15"
    assert result.responsible_people.value == ("Mira Talen",)
    assert result.obligations.value == (
        "The provider shall maintain an operational service desk.",
        "Northstar must notify the customer within 24 hours.",
        "No subcontractor may access production data without approval.",
    )
    assert "No subcontractor may access production data without approval." not in (
        result.risks.value or ()
    )
