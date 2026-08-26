from evals.benchmark import select_method


def _result(
    method: str,
    *,
    answerable: int,
    citations: int,
    abstentions: int,
    retrieval_hits: int | None = None,
    latency: float = 10.0,
) -> dict[str, object]:
    return {
        "retrieval_method": method,
        "citation_precision": 1.0,
        "citation_case_accuracy": answerable / 30,
        "abstention_accuracy": abstentions / 20,
        "extraction_f1": 1.0,
        "error_rate": 0.0,
        "latency_p95_ms": latency,
        "metric_counts": {
            "answerable_correct": answerable,
            "citation_correct": citations,
            "abstention_correct": abstentions,
            "retrieval_hits_at_5": retrieval_hits if retrieval_hits is not None else answerable,
        },
    }


def test_selection_prefers_the_simpler_method_when_quality_is_equal() -> None:
    results = {
        "lexical": _result("lexical", answerable=27, citations=27, abstentions=18),
        "dense": _result("dense", answerable=27, citations=27, abstentions=18, latency=8),
        "hybrid": _result("hybrid", answerable=27, citations=27, abstentions=18, latency=7),
        "hybrid_rerank": _result(
            "hybrid_rerank", answerable=27, citations=27, abstentions=18, latency=6
        ),
    }

    selected, decision = select_method(results)

    assert selected == "lexical"
    assert decision["reranker_retained"] is False


def test_reranker_requires_an_additional_correct_case_and_no_regression() -> None:
    results = {
        "lexical": _result("lexical", answerable=25, citations=25, abstentions=18),
        "dense": _result("dense", answerable=26, citations=26, abstentions=18),
        "hybrid": _result("hybrid", answerable=27, citations=27, abstentions=18),
        "hybrid_rerank": _result(
            "hybrid_rerank", answerable=28, citations=28, abstentions=18
        ),
    }

    selected, decision = select_method(results)

    assert selected == "hybrid_rerank"
    assert decision["reranker_retained"] is True
    assert decision["reranker_additional_answerable_correct"] == 1


def test_reranker_is_rejected_when_abstention_regresses() -> None:
    results = {
        "lexical": _result("lexical", answerable=25, citations=25, abstentions=18),
        "dense": _result("dense", answerable=26, citations=26, abstentions=18),
        "hybrid": _result("hybrid", answerable=27, citations=27, abstentions=18),
        "hybrid_rerank": _result(
            "hybrid_rerank", answerable=28, citations=28, abstentions=17
        ),
    }

    selected, decision = select_method(results)

    assert selected == "hybrid"
    assert decision["reranker_retained"] is False


def test_hybrid_wins_an_end_metric_tie_when_retrieval_recall_is_higher() -> None:
    results = {
        "lexical": _result(
            "lexical", answerable=29, citations=30, abstentions=20, retrieval_hits=25
        ),
        "dense": _result(
            "dense", answerable=27, citations=27, abstentions=20, retrieval_hits=22
        ),
        "hybrid": _result(
            "hybrid", answerable=29, citations=30, abstentions=20, retrieval_hits=30
        ),
        "hybrid_rerank": _result(
            "hybrid_rerank", answerable=29, citations=30, abstentions=20, retrieval_hits=29
        ),
    }

    selected, decision = select_method(results)

    assert selected == "hybrid"
    assert decision["reranker_retained"] is False
