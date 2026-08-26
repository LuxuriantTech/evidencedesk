"""Generate the deterministic text-PDF shipped with the synthetic corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]


def generate_pdf(output: Path) -> None:
    source = json.loads(
        (ROOT / "datasets/sources/northstar_master_services_agreement_source.json").read_text(
            encoding="utf-8"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output), pagesize=A4, invariant=1, pageCompression=1)
    canvas.setTitle("Synthetic Northstar Master Services Agreement")
    canvas.setAuthor("EvidenceDesk synthetic corpus generator")
    canvas.setFont("Helvetica", 11)
    _, height = A4
    for page_number, page in enumerate(source["pages"]):
        y = height - 72
        for line in page.splitlines():
            canvas.drawString(54, y, line)
            y -= 20
        if page_number < len(source["pages"]) - 1:
            canvas.showPage()
            canvas.setFont("Helvetica", 11)
    canvas.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/generated/northstar_master_services_agreement.pdf",
    )
    args = parser.parse_args()
    generate_pdf(args.output)


if __name__ == "__main__":
    main()
