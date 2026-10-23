from evidencedesk_api.answering import (
    DecisionConfig,
    DeterministicGroundedAnswerProvider,
    _field_value,
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
    ordinal: int | None = None,
) -> RankedChunk:
    chunk = EvidenceChunk(
        id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        page=page,
        section=None,
        text=text,
        embedding=[0.0] * 384,
        ordinal=ordinal,
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
    passage = "BatchID BA-12; BatchID BA-13. Nitrate concentration: 0.92 mg/L."
    result = _answerer().answer(
        "What nitrate concentration is reported for BatchID BA-12?",
        [_ranked("competing-batches", passage)],
    )

    assert result.status == "abstained"
    assert result.answerable is False
    assert result.citations == ()
    assert "0.92 mg/L" not in result.answer


def test_unique_record_header_and_linked_batch_scope_a_later_measurement() -> None:
    header = _ranked(
        "sensor-log:p1:c0",
        (
            "ReadingRecordID RR-71 | CampaignID CP-8. "
            "The record binds SampleBatchID SB-6 to CampaignID CP-8."
        ),
        document_id="sensor-log",
    )
    reading = _ranked(
        "sensor-log:p1:c1",
        "At 08:30 CEST, nitrate concentration was 0.73 mg/L.",
        document_id="sensor-log",
    )
    result = _answerer().answer(
        (
            "At 08:30 CEST, what nitrate concentration is recorded for "
            "ReadingRecordID RR-71 and SampleBatchID SB-6?"
        ),
        [header, reading],
    )

    assert result.status == "answered"
    assert result.answer == "0.73 mg/L"
    assert result.supporting_excerpt == "At 08:30 CEST, nitrate concentration was 0.73 mg/L."
    assert {citation.chunk_id for citation in result.citations} == {
        "sensor-log:p1:c0",
        "sensor-log:p1:c1",
    }
    assert any("RR-71" in citation.excerpt for citation in result.citations)
    assert any("SB-6" in citation.excerpt for citation in result.citations)


def test_disconnected_record_identifiers_are_not_joined_across_context_sentences() -> None:
    header = _ranked(
        "work-order:p1:c0",
        "WorkOrderID WO-92 is open. ServiceRecordID SR-27 is inactive.",
        document_id="work-order",
    )
    result = _answerer().answer("Which ServiceRecordID belongs to WorkOrderID WO-92?", [header])

    assert result.status == "abstained"
    assert result.answerable is False
    assert result.citations == ()


def test_explicit_reference_relation_keeps_its_direction() -> None:
    reverse = _ranked(
        "authorization",
        "PermitID PM-71 authorizes WorkPlanID WP-58.",
        document_id="authorization",
    )
    direct = _ranked(
        "work-plan",
        "WorkPlanID WP-58 explicitly references PermitID PM-71.",
        document_id="work-plan",
    )
    question = "Which PermitID does WorkPlanID WP-58 explicitly reference?"
    wrong_direction = _answerer().answer(question, [reverse])
    passive_direction = _answerer().answer(
        question,
        [
            _ranked(
                "passive-work-plan",
                "WorkPlanID WP-58 is explicitly referenced by PermitID PM-71.",
                document_id="passive-work-plan",
            )
        ],
    )
    supported = _answerer().answer(question, [reverse, direct])

    assert wrong_direction.status == "abstained"
    assert passive_direction.status == "abstained"
    assert supported.status == "answered"
    assert supported.answer == "PM-71"
    assert supported.supporting_document == "work-plan"
    assert supported.supporting_excerpt == (
        "WorkPlanID WP-58 explicitly references PermitID PM-71."
    )


def test_reference_as_a_field_name_is_not_treated_as_a_directed_relation() -> None:
    result = _answerer().answer(
        "What is the reference number?",
        [_ranked("reference", "Reference number: RF-82.")],
    )

    assert result.status == "answered"
    assert result.answer == "RF-82"


def test_prose_count_is_extracted_from_its_assertion_not_an_unknown_measurement() -> None:
    result = _answerer().answer(
        "How many ceramic tiles did record TR-44 include?",
        [_ranked("count", "Record TR-44 includes twelve ceramic tiles.")],
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.supporting_excerpt is not None
    assert "twelve" in result.answer.casefold()
    assert "ceramic tiles" in result.supporting_excerpt.casefold()


def test_model_property_does_not_absorb_the_object_phrase_after_of() -> None:
    result = _answerer().answer(
        "Which model of ceramic seal was added by ChangeID CH-29?",
        [
            _ranked(
                "change",
                "ChangeID CH-29 added a ceramic seal, model CS-7, at no charge.",
            )
        ],
    )

    assert result.status == "answered"
    assert result.answer == "CS-7"


def test_unique_record_id_scopes_a_value_across_document_pages() -> None:
    header = _ranked(
        "materials:p1:c0",
        "WorkOrderID WO-83 | CustomerID CU-9. WorkOrderID WO-83 covers replacement parts.",
        page=1,
        document_id="materials",
        ordinal=0,
    )
    price = _ranked(
        "materials:p2:c0",
        "The approved unit price is EUR 17.50 per component.",
        page=2,
        document_id="materials",
        ordinal=1,
    )
    result = _answerer().answer(
        "What unit price was approved under WorkOrderID WO-83?", [header, price]
    )

    assert result.status == "answered"
    assert result.answer == "EUR 17.50"
    assert result.supporting_page == 2
    assert {citation.page for citation in result.citations} == {1, 2}


def test_later_record_does_not_invalidate_an_earlier_scoped_fact() -> None:
    header = _ranked(
        "materials:p1:c0",
        "WorkOrderID WO-83 covers replacement parts.",
        page=1,
        document_id="materials",
        ordinal=0,
    )
    price = _ranked(
        "materials:p2:c0",
        "The approved unit price is EUR 17.50 per component.",
        page=2,
        document_id="materials",
        ordinal=1,
    )
    other = _ranked(
        "materials:p3:c0",
        "WorkOrderID WO-84 covers separate replacement parts.",
        page=3,
        document_id="materials",
        ordinal=2,
    )
    result = _answerer().answer(
        "What unit price was approved under WorkOrderID WO-83?",
        [header, price, other],
    )

    assert result.status == "answered"
    assert result.answer == "EUR 17.50"


def test_record_scope_stops_before_a_competing_identifier() -> None:
    header = _ranked(
        "materials:p1:c0",
        "WorkOrderID WO-83 covers replacement parts.",
        page=1,
        document_id="materials",
        ordinal=0,
    )
    competing_record = _ranked(
        "materials:p2:c0",
        "WorkOrderID WO-84 covers a separate order.",
        page=2,
        document_id="materials",
        ordinal=1,
    )
    unlabelled_price = _ranked(
        "materials:p3:c0",
        "The approved unit price is EUR 17.50 per component.",
        page=3,
        document_id="materials",
        ordinal=2,
    )
    result = _answerer().answer(
        "What unit price was approved under WorkOrderID WO-83?",
        [header, competing_record, unlabelled_price],
    )

    assert result.status != "answered"


def test_french_completed_replacement_matches_english_action_but_not_schedule() -> None:
    question = (
        "À quelle heure le remplacement de batterie de PumpID PU-6 est-il noté comme terminé ?"
    )
    scheduled = _answerer().answer(
        question,
        [_ranked("scheduled", "Battery replacement is scheduled for 10:50 CEST.")],
    )
    completed = _answerer().answer(
        question,
        [
            _ranked(
                "service",
                (
                    "On 2031-08-04 at 10:50 CEST, technician B. Hart "
                    "replaced the battery for PumpID PU-6."
                ),
            )
        ],
    )

    assert scheduled.status != "answered"
    assert completed.status == "answered"
    assert completed.answer == "10:50 CEST"
    assert completed.supporting_excerpt is not None
    assert "replaced the battery" in completed.supporting_excerpt


def test_french_field_question_stops_at_the_copula_and_modifier() -> None:
    result = _answerer().answer(
        "Quel modèle est indiqué pour DeviceTag DV-5 dans CO-14 ?",
        [
            _ranked(
                "device-log",
                (
                    "CheckoutID CO-14 explicitly assigns DeviceTag DV-5 to "
                        "CampaignID CA-2. Model listed: TrackSense 7B."
                ),
            )
        ],
    )

    assert result.status == "answered"
    assert result.answer == "TrackSense 7B"
    assert result.supporting_excerpt == "Model listed: TrackSense 7B."


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


def test_spelled_count_is_returned_as_the_supported_value() -> None:
    result = _answerer().answer(
        "How many filters did WorkOrder WO-83 cover?",
        [_ranked("filter-order", "WorkOrder WO-83 covers twelve replacement filters.")],
    )

    assert result.status == "answered"
    assert result.answer.casefold().rstrip(".") == "twelve"
    assert result.supporting_excerpt == "WorkOrder WO-83 covers twelve replacement filters."


def test_count_ignores_digits_in_record_ids_and_neighboring_quantities() -> None:
    result = _answerer().answer(
        "How many ceramic tiles did record TR-44 include?",
        [
            _ranked(
                "tile-record",
                "Record TR-44 includes 12 pumps and eight ceramic tiles.",
            )
        ],
    )

    assert result.status == "answered"
    assert result.answer.casefold().rstrip(".") == "eight"


def test_count_binds_a_hyphenated_product_to_its_quantity() -> None:
    result = _answerer().answer(
        "How many Ion-3 pumps did contract AG-70 cover?",
        [_ranked("pump-contract", "Contract AG-70 covers twelve Ion-3 isolation pumps.")],
    )

    assert result.status == "answered"
    assert result.answer.casefold().rstrip(".") == "twelve"


def test_count_prefers_the_requested_delivery_state_and_keeps_its_bound() -> None:
    result = _answerer().answer(
        "How many units were received?",
        [_ranked("receipt", "12 units were planned; 8 units were received.")],
    )
    bounded = _answerer().answer(
        "How many items were delivered?",
        [_ranked("delivery", "At least twelve items were delivered.")],
    )

    assert result.status == "answered"
    assert result.answer == "8"
    assert bounded.status == "answered"
    assert bounded.answer is not None
    assert bounded.answer.casefold() == "at least twelve"


def test_requested_actor_does_not_conflict_with_another_actors_event_time() -> None:
    entry = _ranked(
        "access-log:p1:c0",
        "Access record AR-71: Nia Okafor entered the gallery at 08:42.",
        document_id="access-log",
        page=1,
        ordinal=0,
    )
    exit_event = _ranked(
        "access-log:p2:c0",
        "Pavel Rusu left the gallery at 16:18.",
        document_id="access-log",
        page=2,
        ordinal=1,
    )
    result = _answerer().answer(
        "At what time did Nia Okafor enter the gallery?", [entry, exit_event]
    )

    assert result.status == "answered"
    assert result.answer == "08:42"
    assert result.supporting_excerpt is not None
    assert "Nia Okafor entered the gallery at 08:42" in result.supporting_excerpt
    assert "Pavel Rusu" not in result.supporting_excerpt


def test_time_without_timezone_does_not_inject_a_null_text_value() -> None:
    result = _answerer().answer(
        "What time is recorded for ReadingID RD-35?",
        [
            _ranked(
                "reading-log",
                "ReadingID RD-35 identifies the 09:07 reading as 48 percent humidity.",
            )
        ],
    )

    assert result.status == "answered"
    assert result.answer == "09:07"
    assert "None" not in result.answer


def test_contrasted_record_times_are_returned_as_ambiguous() -> None:
    observed = _ranked(
        "sensor-log:p1:c0",
        "SensorID SX-5 records its morning reading at 08:40 on 3 June 2026.",
        document_id="sensor-log",
        page=1,
        ordinal=0,
    )
    transcription = _ranked(
        "sensor-log:p2:c0",
        "The correction sheet for SensorID SX-5 gives that same morning reading as "
        "08:55 rather than 08:40 on 3 June 2026, with no signed correction.",
        document_id="sensor-log",
        page=2,
        ordinal=1,
    )
    result = _answerer().answer(
        "At what time was the morning reading from SensorID SX-5 recorded on 3 June 2026?",
        [observed, transcription],
    )

    assert result.status == "ambiguous"
    assert result.answerable is False
    assert {citation.chunk_id for citation in result.citations} == {
        "sensor-log:p1:c0",
        "sensor-log:p2:c0",
    }


def test_same_reading_reference_allows_companion_id_but_not_unlinked_records() -> None:
    linked = _ranked(
        "log:p2:c0",
        "LogID LG-41 covers the same morning reading as LogID LG-41-1 and transcribes "
        "its time as 08:55 rather than 08:40 on 3 June 2026.",
        document_id="log",
        page=2,
    )
    result = _answerer().answer(
        "At what time was the morning reading from LogID LG-41 recorded on 3 June 2026?",
        [linked],
    )

    assert result.status == "ambiguous"
    assert {citation.chunk_id for citation in result.citations} == {"log:p2:c0"}


def test_explicitly_confirmed_time_correction_resolves_the_earlier_value() -> None:
    correction = _ranked(
        "reading-log:p2:c0",
        "The corrected and confirmed time for ReadingID RD-82 is 09:15 rather than "
        "09:00.",
        document_id="reading-log",
        page=2,
    )
    result = _answerer().answer(
        "What time is recorded for ReadingID RD-82?", [correction]
    )
    assert result.status == "answered"
    assert result.answer == "09:15"

    unrelated = _ranked(
        "other-log:p1:c0",
        "SensorID SN-41 records a separate reading at 08:55.",
        document_id="other-log",
        page=1,
    )
    exact = _ranked(
        "log:p1:c0",
        "LogID LG-41 records 08:40.",
        document_id="log",
        page=1,
    )
    exact_result = _answerer().answer(
        "What time is recorded for LogID LG-41?", [exact]
    )
    unlinked_result = _answerer().answer(
        "What time is recorded for LogID LG-41?", [unrelated]
    )
    assert exact_result.status == "answered"
    assert exact_result.answer == "08:40"
    assert unlinked_result.status == "abstained"


def test_count_skips_date_numbers_and_binds_the_quantity_to_its_object() -> None:
    receipt = _ranked(
        "receipt:p1:c0",
        "ReceiptID RC-72 records delivery of four rolls of barrier film on "
        "10 February 2026 for WorkOrderID WO-84.",
        document_id="receipt",
    )
    result = _answerer().answer(
        "How many rolls of barrier film arrived on ReceiptID RC-72?", [receipt]
    )
    assert result.status == "answered"
    assert result.answer == "four"


def test_record_header_and_local_product_identity_bind_cross_page_count() -> None:
    header = _ranked(
        "agreement:p1:c0",
        "AgreementID AG-2 covers Valve VX-8 and Pump PX-2.",
        document_id="agreement",
        page=1,
        ordinal=0,
    )
    valve_quantity = _ranked(
        "agreement:p2:c0",
        "The agreement lists twelve units of Valve VX-8.",
        document_id="agreement",
        page=2,
        ordinal=1,
    )
    pump_quantity = _ranked(
        "agreement:p2:c1",
        "The agreement lists eight units of Pump PX-2.",
        document_id="agreement",
        page=2,
        ordinal=2,
    )
    question = "How many units of Valve VX-8 are listed under AgreementID AG-2?"
    result = _answerer().answer(question, [header, valve_quantity])
    wrong_product = _answerer().answer(question, [header, pump_quantity])

    assert result.status == "answered"
    assert result.answer == "twelve"
    assert {citation.chunk_id for citation in result.citations} == {
        "agreement:p1:c0",
        "agreement:p2:c0",
    }
    assert wrong_product.status == "abstained"


def test_calendar_days_are_preserved_in_invoice_deadline() -> None:
    invoice = _ranked(
        "invoice:p2:c0",
        "Payment is due within 30 calendar days of invoice receipt.",
        document_id="invoice",
        page=2,
    )
    result = _answerer().answer("What is the payment deadline?", [invoice])
    assert result.status == "answered"
    assert result.answer == "30 calendar days"


def test_tax_qualifier_is_not_treated_as_an_excluded_field() -> None:
    price = _ranked(
        "plan:p1:c0",
        "PlanID PL-73 lists the agreed unit price for the KX-4 pump as EUR 18.40 excluding VAT.",
        document_id="plan",
    )
    result = _answerer().answer(
        "What is the agreed unit price for the KX-4 pump, excluding VAT, under PlanID PL-73?",
        [price],
    )
    assert result.status == "answered"
    assert "EUR 18.40" in result.answer


def test_french_cupboard_and_relative_humidity_labels_match_explicit_fields() -> None:
    cupboard = _ranked(
        "storage:p1:c0",
        "L'équipement est rangé dans the conservation cupboard C-4.",
        document_id="storage",
    )
    result = _answerer().answer("Dans quelle armoire se trouve l'équipement?", [cupboard])
    assert result.status == "answered"
    assert result.answer == "C-4"

    humidity = _ranked(
        "work-order:p1:c0",
        "The approved work order sets the target relative humidity at 52 percent.",
        document_id="work-order",
    )
    humidity_result = _answerer().answer(
        "What humidity target is approved?", [humidity]
    )
    assert humidity_result.status == "answered"
    assert humidity_result.answer == "52 percent"


def test_amount_marked_pending_confirmation_is_not_returned_as_final() -> None:
    detail = _ranked(
        "invoice:p1:c0",
        "InvoiceID IN-51 lists an amount of EUR 402.00.",
        document_id="invoice",
        page=1,
        ordinal=0,
    )
    status = _ranked(
        "invoice:p2:c0",
        "InvoiceID IN-51: the amount remains pending confirmation.",
        document_id="invoice",
        page=2,
        ordinal=1,
    )
    result = _answerer().answer(
        "What amount is listed for InvoiceID IN-51?", [detail, status]
    )

    assert result.status == "ambiguous"
    assert result.answerable is False
    assert {citation.chunk_id for citation in result.citations} == {
        "invoice:p1:c0",
        "invoice:p2:c0",
    }


def test_provisional_amount_stays_ambiguous_against_a_signed_record() -> None:
    reconciliation = _ranked(
        "reconciliation:p1:c0",
        "ReconciliationID RN-8 for AgreementID AG-70 records the LX-8 pump unit price "
        "as EUR 17.50.",
        document_id="reconciliation",
        document_name="reconciliation_note.txt",
        ordinal=0,
    )
    status = _ranked(
        "reconciliation:p2:c0",
        "The same note marked that figure to confirm and issued no correction.",
        document_id="reconciliation",
        document_name="reconciliation_note.txt",
        page=2,
        ordinal=1,
    )
    agreement = _ranked(
        "agreement:p1:c0",
        "AgreementID AG-70 covers LX-8 pump assemblies.",
        document_id="agreement",
        document_name="signed_agreement.txt",
        ordinal=0,
    )
    signed_price = _ranked(
        "agreement:p2:c0",
        "The agreed unit price is EUR 16.75 per pump.",
        document_id="agreement",
        document_name="signed_agreement.txt",
        page=2,
        ordinal=1,
    )
    result = _answerer().answer(
        "What unit price for the LX-8 pump was recorded under AgreementID AG-70?",
        [reconciliation, status, agreement, signed_price],
    )

    assert result.status == "ambiguous"
    assert result.answerable is False
    assert {citation.chunk_id for citation in result.citations} == {
        "reconciliation:p1:c0",
        "reconciliation:p2:c0",
        "agreement:p1:c0",
        "agreement:p2:c0",
    }


def test_invoice_total_value_is_full_amount_and_unresolved_conflict_is_ambiguous() -> None:
    invoice = _ranked(
        "invoice:p1:c0",
        "InvoiceID IV-35 reports a pre-tax total of EUR 8,400 for eight components.",
        document_id="invoice",
        document_name="invoice.txt",
        ordinal=0,
    )
    note_amount = _ranked(
        "review-note:p1:c0",
        "ReviewID RV-7 for InvoiceID IV-35 records a pre-tax total of EUR 8,600.",
        document_id="review-note",
        document_name="review_note.txt",
        ordinal=0,
    )
    note_status = _ranked(
        "review-note:p2:c0",
        "Its author marked both totals to confirm and issued no correction.",
        document_id="review-note",
        document_name="review_note.txt",
        page=2,
        ordinal=1,
    )
    question = "What was the total value of InvoiceID IV-35 before tax?"
    result = _answerer().answer(question, [invoice, note_amount, note_status])

    assert result.status == "ambiguous"
    assert result.answerable is False
    assert {citation.chunk_id for citation in result.citations} == {
        "invoice:p1:c0",
        "review-note:p1:c0",
        "review-note:p2:c0",
    }
    assert any("EUR 8,400" in citation.excerpt for citation in result.citations)
    assert any("EUR 8,600" in citation.excerpt for citation in result.citations)


def test_labeled_currency_field_keeps_thousands_separator() -> None:
    extracted = _field_value(
        "What is the deposit for InvoiceID IV-36?",
        "InvoiceID IV-36, deposit: EUR 12,450.",
    )

    assert extracted is not None
    assert extracted[0] == "EUR 12,450"


def test_unique_immediate_record_anaphora_uses_source_ordinal_not_retrieval_order() -> None:
    fact = _ranked(
        "opaque-fact-chunk",
        "The agreement covers twelve ceramic valves.",
        document_id="opaque-agreement",
        page=1,
        ordinal=1,
    )
    anchor = _ranked(
        "opaque-anchor-chunk",
        "AgreementID AG-73 was reviewed.",
        document_id="opaque-agreement",
        page=1,
        ordinal=0,
    )

    result = _answerer().answer(
        "How many ceramic valves are covered by AgreementID AG-73?", [fact, anchor]
    )

    assert result.status == "answered"
    assert result.answer == "twelve"
    assert {citation.chunk_id for citation in result.citations} == {
        "opaque-anchor-chunk",
        "opaque-fact-chunk",
    }


def test_missing_source_ordinal_gap_blocks_record_anaphora() -> None:
    anchor = _ranked(
        "opaque-anchor-chunk",
        "AgreementID AG-74 was reviewed.",
        document_id="opaque-agreement",
        ordinal=0,
    )
    fact = _ranked(
        "opaque-fact-chunk",
        "The agreement covers twelve ceramic valves.",
        document_id="opaque-agreement",
        ordinal=2,
    )

    result = _answerer().answer(
        "How many ceramic valves are covered by AgreementID AG-74?", [anchor, fact]
    )

    assert result.status == "abstained"
    assert result.citations == ()


def test_pronoun_anaphora_requires_one_typed_antecedent() -> None:
    anchor = _ranked(
        "opaque-anchor-chunk",
        "AgreementID AG-75 was reviewed.",
        document_id="opaque-agreement",
        ordinal=0,
    )
    fact = _ranked(
        "opaque-fact-chunk",
        "It covers twelve ceramic valves.",
        document_id="opaque-agreement",
        ordinal=1,
    )
    result = _answerer().answer(
        "How many ceramic valves are covered by AgreementID AG-75?", [fact, anchor]
    )
    assert result.status == "answered"
    assert result.answer == "twelve"

    competing_anchor = _ranked(
        "ambiguous-anchor-chunk",
        "AgreementID AG-75 was reviewed with InvoiceID IV-8.",
        document_id="opaque-agreement",
        ordinal=0,
    )
    ambiguous_fact = _ranked(
        "ambiguous-fact-chunk",
        "It covers twelve ceramic valves.",
        document_id="opaque-agreement",
        ordinal=1,
    )
    ambiguous = _answerer().answer(
        "How many ceramic valves are covered by AgreementID AG-75?",
        [competing_anchor, ambiguous_fact],
    )
    assert ambiguous.status == "abstained"


def test_multiple_matching_typed_anchors_block_definite_anaphora() -> None:
    anchor = _ranked(
        "opaque-anchor-chunk",
        "AgreementID AG-76 and AgreementID AG-77 were reviewed.",
        document_id="opaque-agreement",
        ordinal=0,
    )
    fact = _ranked(
        "opaque-fact-chunk",
        "The agreement covers twelve ceramic valves.",
        document_id="opaque-agreement",
        ordinal=1,
    )

    result = _answerer().answer(
        "How many ceramic valves are covered by AgreementID AG-76?", [anchor, fact]
    )

    assert result.status == "abstained"
