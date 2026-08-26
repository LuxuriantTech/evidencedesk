import io
import re
from dataclasses import dataclass

from pypdf import PdfReader


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedPage:
    page: int
    blocks: tuple[tuple[str | None, str], ...]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    page: int
    section: str | None
    ordinal: int
    text: str


def _parse_markdown(text: str) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    blocks: list[tuple[str | None, str]] = []
    section: str | None = None
    paragraph: list[str] = []
    page_number = 1

    def flush() -> None:
        if paragraph:
            blocks.append((section, " ".join(paragraph).strip()))
            paragraph.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush()
            if len(heading.group(1)) == 2 and blocks:
                pages.append(ParsedPage(page=page_number, blocks=tuple(blocks)))
                blocks.clear()
                page_number += 1
            section = heading.group(2).strip()
            if len(heading.group(1)) == 1:
                blocks.append((section, section))
        elif not line:
            flush()
        else:
            paragraph.append(line)
    flush()
    if blocks:
        pages.append(ParsedPage(page=page_number, blocks=tuple(blocks)))
    return pages


def parse_document(data: bytes, *, media_type: str) -> list[ParsedPage]:
    if media_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise DocumentParseError("encrypted_pdf")
            pages = [
                ParsedPage(page=index, blocks=((None, (page.extract_text() or "").strip()),))
                for index, page in enumerate(reader.pages, start=1)
            ]
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError("invalid_pdf") from exc
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("invalid_text") from exc
        if media_type == "text/markdown":
            pages = _parse_markdown(text)
        else:
            logical_pages = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            pages = [
                ParsedPage(page=index, blocks=((None, content),))
                for index, content in enumerate(logical_pages, start=1)
            ]
    if not pages or not any(text.strip() for page in pages for _, text in page.blocks):
        raise DocumentParseError("no_extractable_text")
    return pages


def _split_text(text: str, max_chars: int) -> list[str]:
    sentences = [
        part.strip()
        for line in text.splitlines()
        for part in re.split(r"(?<=[.!?])\s+", line)
        if part.strip()
    ]
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        current: list[str] = []
        length = 0
        for word in sentence.split():
            added = len(word) + (1 if current else 0)
            if current and length + added > max_chars:
                pieces.append(" ".join(current))
                current = [word]
                length = len(word)
            else:
                current.append(word)
                length += added
        if current:
            pieces.append(" ".join(current))
    return pieces


def chunk_pages(pages: list[ParsedPage], *, max_chars: int = 900) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    ordinal = 0
    for page in pages:
        for section, text in page.blocks:
            for piece in _split_text(text, max_chars):
                chunks.append(
                    DocumentChunk(page=page.page, section=section, ordinal=ordinal, text=piece)
                )
                ordinal += 1
    return chunks
