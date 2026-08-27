from uuid import uuid4

from evidencedesk_api.schemas import AskResponse


def test_ask_response_exposes_explicit_grounding_fields() -> None:
    response = AskResponse(
        status="answered",
        answer="14 October 2031",
        confidence=0.91,
        mode="deterministic-evidence-v3",
        citations=[],
        correlation_id=uuid4(),
        answerable=True,
        supporting_document="supplier-note",
        supporting_page=2,
        supporting_excerpt="The agreement becomes effective on 14 October 2031.",
        ambiguity_reason=None,
        extracted_fields={"effective_date": "14 October 2031"},
        candidate_assessments=[
            {
                "answerable": True,
                "answer": "14 October 2031",
                "confidence": 0.91,
                "supporting_document": "supplier-note",
                "supporting_page": 2,
                "supporting_excerpt": (
                    "The agreement becomes effective on 14 October 2031."
                ),
                "ambiguity_reason": None,
                "extracted_fields": {"effective_date": "14 October 2031"},
                "supporting_chunk_id": "chunk-1",
            }
        ],
    )

    assert response.answerable is True
    assert response.supporting_page == 2
    assert response.extracted_fields["effective_date"] == "14 October 2031"
    assert response.candidate_assessments[0].supporting_chunk_id == "chunk-1"
