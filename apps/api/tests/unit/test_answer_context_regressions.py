from evidencedesk_api.answering import (
    DecisionConfig,
    DeterministicGroundedAnswerProvider,
    _scope_ranked,
)
from evidencedesk_api.providers import DeterministicEmbeddingProvider
from evidencedesk_api.retrieval import EvidenceChunk, RankedChunk


def _ranked(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "synthetic-record",
    document_name: str = "field-record.txt",
    page: int = 1,
) -> RankedChunk:
    chunk = EvidenceChunk(
        id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        page=page,
        section=None,
        text=text,
        embedding=[0.0] * 384,
    )
    return RankedChunk(
        chunk=chunk,
        score=0.8,
        lexical_score=0.8,
        dense_score=0.8,
        entity_score=0.0,
    )


def _answerer() -> DeterministicGroundedAnswerProvider:
    # Match the configured product thresholds; these tests must not turn off
    # the support gate to make synthetic examples pass.
    return DeterministicGroundedAnswerProvider(
        embeddings=DeterministicEmbeddingProvider(dimension=384),
        config=DecisionConfig(
            support_threshold=0.48,
            partial_support_threshold=0.38,
            contradiction_margin=0.08,
        ),
    )


def test_unrelated_permit_absence_does_not_override_present_permit_id() -> None:
    passage = (
        "WorkPlanID WP-26 explicitly references PermitID PM-52. "
        "The permit does not name a vessel operator."
    )
    result = _answerer().answer(
        "Which PermitID does WorkPlanID WP-26 explicitly reference?",
        [_ranked("work-plan", passage)],
    )

    assert result.status == "answered"
    assert result.answer == "PM-52"
    assert result.supporting_document == "synthetic-record"
    assert result.supporting_page == 1
    assert result.supporting_excerpt == "WorkPlanID WP-26 explicitly references PermitID PM-52."
    assert {citation.excerpt for citation in result.citations} == {
        "WorkPlanID WP-26 explicitly references PermitID PM-52."
    }
    assert "vessel operator" not in result.supporting_excerpt


def test_present_and_absent_same_id_are_ambiguous_with_both_proofs() -> None:
    present = "WorkPlanID WP-26 references PermitID PM-52."
    absent = "WorkPlanID WP-26 does not specify a PermitID."
    result = _answerer().answer(
        "Which PermitID does WorkPlanID WP-26 reference?",
        [_ranked("positive", present), _ranked("negative", absent)],
    )

    assert result.status == "ambiguous"
    assert result.answerable is False
    assert len(result.citations) == 2
    assert {citation.chunk_id for citation in result.citations} == {"positive", "negative"}
    assert {citation.excerpt for citation in result.citations} == {present, absent}


def test_multiword_field_label_does_not_select_neighboring_status() -> None:
    status = "For BuildingID BX-42, the evacuation assembly status is approved."
    point = "The evacuation assembly point for BuildingID BX-42 is Garden Court."
    result = _answerer().answer(
        "What is the evacuation assembly point for BuildingID BX-42?",
        [_ranked("building-record", f"{status} {point}")],
    )

    assert result.status == "answered"
    assert result.answer == "Garden Court"
    assert result.supporting_excerpt == point
    assert "approved" not in result.answer
    assert all("status" not in citation.excerpt.casefold() for citation in result.citations)


def test_unknown_measurement_unit_does_not_fall_through_to_neighboring_measure() -> None:
    passage = (
        "Ultrasonic return strength at receiver QN-3: 4.2 zph. "
        "Air temperature at receiver QN-3: 19 °C."
    )
    result = _answerer().answer(
        "What was the ultrasonic return strength at receiver QN-3?",
        [_ranked("receiver-log", passage)],
    )

    assert result.status == "abstained"
    assert result.answerable is False
    assert "19 °C" not in result.answer
    assert result.citations == ()


def test_french_source_reference_matches_equivalent_english_title() -> None:
    authorization = _ranked(
        "authorization",
        "Field authorization\nPermitID PM-52 | Entry point: West Pier.",
        document_id="authorization-doc",
        document_name="field_authorization.pdf",
    )
    report = _ranked(
        "report",
        "Inspection report\nPermitID PM-52 | Entry point: East Dock.",
        document_id="report-doc",
        document_name="inspection_report.pdf",
    )
    scoped = _scope_ranked(
        "Selon «field authorization», quel est le point d'entrée pour PermitID PM-52?",
        [authorization, report],
    )

    assert [item.chunk.document_id for item in scoped] == ["authorization-doc"]


def test_unique_header_context_scopes_date_and_time_in_later_chunks() -> None:
    header = _ranked(
        "field-log:p1:c0",
        "FieldLogID FL-94 | SiteCode VT-6",
        document_id="field-log",
        document_name="field_log.txt",
    )
    date = _ranked(
        "field-log:p1:c1",
        "Effective date: 2030-07-12",
        document_id="field-log",
        document_name="field_log.txt",
    )
    time = _ranked(
        "field-log:p1:c2",
        "Time: 06:45 CEST",
        document_id="field-log",
        document_name="field_log.txt",
    )
    answerer = _answerer()

    date_result = answerer.answer(
        "What effective date is recorded for FieldLogID FL-94?", [header, date, time]
    )
    time_result = answerer.answer(
        "What time is recorded for FieldLogID FL-94?", [header, date, time]
    )

    assert date_result.status == "answered"
    assert date_result.answer == "2030-07-12"
    assert date_result.supporting_excerpt == "Effective date: 2030-07-12"
    assert {citation.chunk_id for citation in date_result.citations} == {
        "field-log:p1:c0",
        "field-log:p1:c1",
    }
    assert time_result.status == "answered"
    assert time_result.answer == "06:45 CEST"
    assert time_result.supporting_excerpt == "Time: 06:45 CEST"
    assert {citation.chunk_id for citation in time_result.citations} == {
        "field-log:p1:c0",
        "field-log:p1:c2",
    }


def test_similar_concurrent_record_ids_do_not_join_to_one_field_value() -> None:
    passage = (
        "BatchID BA-12; BatchID BA-13. "
        "Nitrate concentration: 0.92 mg/L."
    )
    result = _answerer().answer(
        "What nitrate concentration is reported for BatchID BA-12?",
        [_ranked("competing-batches", passage)],
    )

    assert result.status == "abstained"
    assert result.answerable is False
    assert result.citations == ()
    assert "0.92 mg/L" not in result.answer


def test_identifier_link_alone_does_not_establish_equipment_malfunction() -> None:
    passage = (
        "IncidentID IN-32 references AssetTag HX-9 and EventID EV-4. "
        "A technician reported intermittent interruptions; the cause is unconfirmed "
        "and no final finding is recorded."
    )
    result = _answerer().answer(
        "Did AssetTag HX-9 malfunction during EventID EV-4?",
        [_ranked("incident", passage)],
    )

    assert result.status in {"abstained", "ambiguous", "partially_supported"}
    assert result.answerable is False
    assert not result.answer.casefold().startswith("yes")


def test_unrelated_timestamp_does_not_answer_when_completion_is_unrecorded() -> None:
    header = _ranked(
        "pump-log:p1:c0",
        "InspectionRecordID IR-62 | PumpID P-8",
        document_id="pump-log",
        document_name="pump_log.txt",
    )
    arrival = _ranked(
        "pump-log:p1:c1",
        "A technician arrived at 14:20 CEST.",
        document_id="pump-log",
        document_name="pump_log.txt",
    )
    completion = _ranked(
        "pump-log:p1:c2",
        "The pump inspection remains in progress; completion time is not recorded.",
        document_id="pump-log",
        document_name="pump_log.txt",
    )
    result = _answerer().answer(
        "At what time was inspection completion recorded for PumpID P-8?",
        [header, arrival, completion],
    )

    assert result.status != "answered"
    assert result.answerable is False
    assert "14:20" not in result.answer
    assert all("14:20" not in citation.excerpt for citation in result.citations)
