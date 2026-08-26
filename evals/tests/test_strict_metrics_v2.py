from evidencedesk_api.retrieval import Citation

from evals.runner import _answer_matches, _citation_matches, _normalize_typed, _value_matches


def _citation(*, document_id: str = "doc-a", page: int = 2, excerpt: str) -> Citation:
    return Citation(
        chunk_id="chunk",
        document_id=document_id,
        document_name=f"{document_id}.md",
        page=page,
        section=None,
        excerpt=excerpt,
    )


def test_strict_citation_rejects_the_right_text_on_the_wrong_document_or_page() -> None:
    expected = [
        {
            "document_id": "doc-a",
            "page": 2,
            "excerpt": "Annual fee: EUR 48,000",
        }
    ]

    assert not _citation_matches(
        _citation(document_id="doc-b", excerpt="EUR 48,000"), expected
    )
    assert not _citation_matches(
        _citation(page=3, excerpt="EUR 48,000"), expected
    )


def test_strict_citation_accepts_only_a_returned_subspan_of_expected_evidence() -> None:
    expected = [
        {
            "document_id": "doc-a",
            "page": 2,
            "excerpt": "Annual fee: EUR 48,000",
        }
    ]

    assert _citation_matches(_citation(excerpt="EUR 48,000"), expected)
    assert not _citation_matches(
        _citation(excerpt="The contract says Annual fee: EUR 48,000 and more."), expected
    )
    assert not _citation_matches(_citation(excerpt="EUR"), expected)


def test_typed_date_normalization_compares_iso_english_and_french_forms() -> None:
    assert _normalize_typed("2027-02-01", value_type="date") == "2027-02-01"
    assert _normalize_typed("1 February 2027", value_type="date") == "2027-02-01"
    assert _normalize_typed("1er février 2027", value_type="date") == "2027-02-01"
    assert _value_matches("1er février 2027", "2027-02-01", value_type="date")


def test_typed_money_normalization_preserves_currency_and_numeric_value() -> None:
    assert _value_matches("EUR 48,000", "48 000 €", value_type="money")
    assert _value_matches("$36,000 USD", "USD 36000", value_type="money")
    assert not _value_matches("EUR 48,000", "USD 48,000", value_type="money")
    assert not _value_matches("EUR 48,000", "EUR 4,800", value_type="money")


def test_text_value_matching_is_not_bidirectional_substring_matching() -> None:
    assert _value_matches(
        "The supplier shall retain records for 45 days.",
        "the supplier shall retain records for 45 days",
        value_type="text",
    )
    assert not _value_matches("retain records", "retain records for 45 days", value_type="text")


def test_french_verb_prefix_is_not_mistaken_for_the_month_of_may() -> None:
    sentence = "Alpine s'engage à maintenir une ligne d'incident."

    assert _answer_matches(sentence, sentence)


def test_modal_may_is_not_mistaken_for_a_date() -> None:
    sentence = "An incomplete index may confuse retrieval."

    assert _answer_matches(sentence, sentence)
