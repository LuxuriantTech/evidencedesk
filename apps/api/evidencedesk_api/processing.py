import io
import math
import multiprocessing
import re
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

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


def _parse_pdf_pages(
    data: bytes, *, max_pages: int, max_extracted_chars: int
) -> list[ParsedPage]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentParseError("encrypted_pdf")
        if len(reader.pages) > max_pages:
            raise DocumentParseError("too_many_pages")
        pages: list[ParsedPage] = []
        extracted_chars = 0
        for index, page in enumerate(reader.pages, start=1):
            extracted = (page.extract_text() or "").strip()
            extracted_chars += len(extracted)
            if extracted_chars > max_extracted_chars:
                raise DocumentParseError("too_much_extracted_text")
            pages.append(ParsedPage(page=index, blocks=((None, extracted),)))
        return pages
    except DocumentParseError:
        raise
    except (MemoryError, OSError):
        raise
    except Exception as exc:
        raise DocumentParseError("invalid_pdf") from exc


def _apply_pdf_resource_limits(memory_bytes: int, timeout_seconds: float) -> None:
    try:
        import resource
    except ImportError:
        return
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    cpu_seconds = max(1, math.ceil(timeout_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))


def _pdf_parser_child(
    connection: Connection,
    data: bytes,
    max_pages: int,
    max_extracted_chars: int,
    memory_bytes: int,
    timeout_seconds: float,
) -> None:
    try:
        _apply_pdf_resource_limits(memory_bytes, timeout_seconds)
        pages = _parse_pdf_pages(
            data,
            max_pages=max_pages,
            max_extracted_chars=max_extracted_chars,
        )
        connection.send(("ok", pages))
    except DocumentParseError as exc:
        connection.send(("error", str(exc)))
    except (MemoryError, OSError, ValueError):
        connection.send(("error", "pdf_resource_limit"))
    except BaseException:
        connection.send(("error", "invalid_pdf"))
    finally:
        connection.close()


def _stop_process(process: BaseProcess) -> None:
    if not process.is_alive():
        process.join(timeout=0.1)
        return
    process.terminate()
    process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)


def _parse_pdf_isolated(
    data: bytes,
    *,
    max_pages: int,
    max_extracted_chars: int,
    timeout_seconds: float,
    memory_bytes: int,
) -> list[ParsedPage]:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_pdf_parser_child,
        args=(
            send,
            data,
            max_pages,
            max_extracted_chars,
            memory_bytes,
            timeout_seconds,
        ),
        daemon=True,
    )
    process.start()
    send.close()
    try:
        if not receive.poll(timeout_seconds):
            _stop_process(process)
            raise DocumentParseError("pdf_processing_timeout")
        try:
            status, payload = receive.recv()
        except EOFError as exc:
            raise DocumentParseError("pdf_resource_limit") from exc
    finally:
        receive.close()
        _stop_process(process)

    if status != "ok":
        raise DocumentParseError(str(payload))
    if not isinstance(payload, list) or not all(
        isinstance(page, ParsedPage) for page in payload
    ):
        raise DocumentParseError("invalid_pdf")
    return payload


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


def parse_document(
    data: bytes,
    *,
    media_type: str,
    max_pages: int = 200,
    max_extracted_chars: int = 1_000_000,
    pdf_timeout_seconds: float = 10.0,
    pdf_memory_bytes: int = 512 * 1024 * 1024,
) -> list[ParsedPage]:
    if media_type == "application/pdf":
        pages = _parse_pdf_isolated(
            data,
            max_pages=max_pages,
            max_extracted_chars=max_extracted_chars,
            timeout_seconds=pdf_timeout_seconds,
            memory_bytes=pdf_memory_bytes,
        )
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("invalid_text") from exc
        if len(text) > max_extracted_chars:
            raise DocumentParseError("too_much_extracted_text")
        if media_type == "text/markdown":
            pages = _parse_markdown(text)
        else:
            logical_pages = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            pages = [
                ParsedPage(page=index, blocks=((None, content),))
                for index, content in enumerate(logical_pages, start=1)
            ]
        if len(pages) > max_pages:
            raise DocumentParseError("too_many_pages")
    if not pages or not any(text.strip() for page in pages for _, text in page.blocks):
        raise DocumentParseError("no_extractable_text")
    return pages


def _sentences(text: str) -> Iterator[str]:
    boundary = re.compile(r"(?<=[.!?])\s+")
    for line in io.StringIO(text):
        start = 0
        for match in boundary.finditer(line):
            sentence = line[start : match.start()].strip()
            if sentence:
                yield sentence
            start = match.end()
        sentence = line[start:].strip()
        if sentence:
            yield sentence


def _split_text(text: str, max_chars: int) -> Iterator[str]:
    for sentence in _sentences(text):
        if len(sentence) <= max_chars:
            yield sentence
            continue
        current: list[str] = []
        length = 0
        for match in re.finditer(r"\S+", sentence):
            word = match.group(0)
            if len(word) > max_chars:
                if current:
                    yield " ".join(current)
                    current = []
                    length = 0
                for offset in range(0, len(word), max_chars):
                    yield word[offset : offset + max_chars]
                continue
            added = len(word) + (1 if current else 0)
            if current and length + added > max_chars:
                yield " ".join(current)
                current = [word]
                length = len(word)
            else:
                current.append(word)
                length += added
        if current:
            yield " ".join(current)


def chunk_pages(
    pages: list[ParsedPage], *, max_chars: int = 900, max_chunks: int = 5_000
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    ordinal = 0
    for page in pages:
        for section, text in page.blocks:
            for piece in _split_text(text, max_chars):
                if len(chunks) >= max_chunks:
                    raise DocumentParseError("too_many_chunks")
                chunks.append(
                    DocumentChunk(page=page.page, section=section, ordinal=ordinal, text=piece)
                )
                ordinal += 1
    return chunks
