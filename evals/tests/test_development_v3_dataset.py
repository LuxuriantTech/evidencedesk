from collections import Counter
from pathlib import Path

from evals.validate_dataset import validate_manifest
from scripts.generate_development_v3 import build_dataset


def test_development_v3_is_grouped_diverse_and_deterministic() -> None:
    first = build_dataset()
    second = build_dataset()

    assert first == second
    corpus, combined, calibration, selection, generation = first
    assert corpus["synthetic_only"] is True
    assert len(corpus["documents"]) == 8
    assert len(combined["cases"]) == 48
    assert len(combined["extraction_targets"]) == 8

    kinds = Counter(case["kind"] for case in combined["cases"])
    assert kinds == {
        "answerable": 32,
        "unanswerable": 8,
        "ambiguous": 4,
        "adversarial": 4,
    }
    assert len(calibration["cases"]) == 24
    assert len(selection["cases"]) == 24

    calibration_families = {
        case["template_family"] for case in combined["cases"] if case["partition"] == "calibration"
    }
    selection_families = {
        case["template_family"] for case in combined["cases"] if case["partition"] == "selection"
    }
    calibration_formulations = {
        case["formulation_family"]
        for case in combined["cases"]
        if case["partition"] == "calibration"
    }
    selection_formulations = {
        case["formulation_family"]
        for case in combined["cases"]
        if case["partition"] == "selection"
    }
    assert len(calibration_families) == 4
    assert len(selection_families) == 4
    assert calibration_families.isdisjoint(selection_families)
    assert calibration_formulations.isdisjoint(selection_formulations)

    assert generation["contains_real_personal_data"] is False
    assert generation["generation_method"] == "deterministic-handwritten-template-families"
    assert generation["partition_rule"] == "grouped-by-template-and-formulation-family"
    assert generation["features"] == {
        "ambiguous_evidence": True,
        "contradictions": True,
        "cross_passage_evidence": True,
        "implicit_facts": True,
        "in_document_prompt_injection": True,
        "multiple_entities": True,
        "synonyms": True,
        "varied_dates_and_amounts": True,
    }


def test_generated_development_v3_manifests_pass_structural_validation(tmp_path: Path) -> None:
    corpus, combined, calibration, selection, _generation = build_dataset()
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(__import__("json").dumps(corpus), encoding="utf-8")

    for name, manifest, expected_cases in (
        ("combined", combined, 48),
        ("calibration", calibration, 24),
        ("selection", selection, 24),
    ):
        manifest_path = tmp_path / f"{name}.json"
        manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
        report = validate_manifest(manifest_path, corpus_path=corpus_path)
        assert report.total_cases == expected_cases
        assert report.document_ids_are_known is True
        assert report.citations_are_exact is True
        assert report.extraction_expectations_are_traceable is True
        assert report.synthetic_only is True
