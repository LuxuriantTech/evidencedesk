from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from scripts.generate_demo_corpus import generate_pdf


def test_generated_contract_pdf_has_only_the_versioned_source_pages(tmp_path: Path) -> None:
    output = tmp_path / "contract.pdf"

    generate_pdf(output)

    reader = PdfReader(output)
    assert len(reader.pages) == 2
    assert "Effective date: 2026-01-15" in reader.pages[0].extract_text()
    assert "within 24 hours" in reader.pages[1].extract_text()
