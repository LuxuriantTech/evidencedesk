from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

import structlog
from arq import Retry
from evidencedesk_api.config import Settings
from evidencedesk_api.extraction import extract_supplier_fields
from evidencedesk_api.models import (
    AuditEvent,
    Chunk,
    Document,
    DocumentStatus,
    Extraction,
    ProcessingTask,
    TaskStatus,
)
from evidencedesk_api.observability import TASKS, configure_logging
from evidencedesk_api.processing import DocumentParseError, ParsedPage, chunk_pages, parse_document
from evidencedesk_api.providers import EmbeddingProvider
from evidencedesk_api.redaction import redact_pii
from evidencedesk_api.retrieval import EvidenceChunk
from evidencedesk_api.storage import DocumentStorage
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


@dataclass(slots=True)
class WorkerContext:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    storage: DocumentStorage
    embeddings: EmbeddingProvider


async def _chunk_count(session: AsyncSession, document_id: UUID) -> int:
    value = await session.scalar(
        select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
    )
    return int(value or 0)


async def _record_storage_failure(
    worker: WorkerContext, document_id: UUID, task_id: UUID, *, retry: bool
) -> None:
    async with worker.session_factory() as session:
        task = await session.get(ProcessingTask, task_id)
        document = await session.get(Document, document_id)
        if task is None or document is None:
            return
        task.status = TaskStatus.QUEUED if retry else TaskStatus.FAILED
        document.status = DocumentStatus.QUEUED if retry else DocumentStatus.FAILED
        task.error_code = "storage_temporarily_unavailable" if retry else "storage_unavailable"
        document.error_code = task.error_code
        session.add(
            AuditEvent(
                actor_id=None,
                action="document.process",
                document_id=document_id,
                result="retry" if retry else "failed",
                correlation_id=task.correlation_id,
                metadata_safe={"error_code": task.error_code, "attempt": task.attempts},
            )
        )
        await session.commit()


async def _record_parse_failure(
    worker: WorkerContext, document_id: UUID, task_id: UUID, error_code: str
) -> dict[str, object]:
    async with worker.session_factory() as session:
        task = await session.get(ProcessingTask, task_id)
        document = await session.get(Document, document_id)
        if task is not None and document is not None:
            task.status = TaskStatus.FAILED
            task.error_code = error_code
            task.completed_at = datetime.now(UTC)
            document.status = DocumentStatus.FAILED
            document.error_code = error_code
            session.add(
                AuditEvent(
                    actor_id=None,
                    action="document.process",
                    document_id=document_id,
                    result="failed",
                    correlation_id=task.correlation_id,
                    metadata_safe={"error_code": error_code, "attempt": task.attempts},
                )
            )
            await session.commit()
    return {"status": "failed", "error_code": error_code}


async def process_document(
    ctx: dict[str, Any], document_id_raw: str, task_id_raw: str, correlation_id_raw: str
) -> dict[str, object]:
    configure_logging()
    logger = structlog.get_logger("evidencedesk.worker")
    started = perf_counter()
    worker: WorkerContext = ctx["worker"]
    document_id = UUID(document_id_raw)
    task_id = UUID(task_id_raw)
    UUID(correlation_id_raw)

    async with worker.session_factory() as session:
        task = await session.scalar(
            select(ProcessingTask).where(ProcessingTask.id == task_id).with_for_update()
        )
        document = await session.get(Document, document_id)
        if task is None or document is None or task.document_id != document.id:
            TASKS.labels("missing").inc()
            return {"status": "missing"}
        if document.status == DocumentStatus.COMPLETED:
            chunks = await _chunk_count(session, document_id)
            logger.info(
                "document_processing_already_completed",
                document_id=str(document_id),
                task_id=str(task_id),
                correlation_id=str(task.correlation_id),
                chunks=chunks,
            )
            return {
                "status": "already_completed",
                "chunks": chunks,
            }
        task.attempts += 1
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now(UTC)
        task.error_code = None
        document.status = DocumentStatus.PROCESSING
        document.error_code = None
        attempt = task.attempts
        storage_key = document.storage_key
        media_type = document.media_type
        document_name = document.filename
        await session.commit()

    try:
        raw_data = worker.storage.get(storage_key)
    except OSError:
        should_retry = attempt < 3
        await _record_storage_failure(worker, document_id, task_id, retry=should_retry)
        if should_retry:
            TASKS.labels("retry").inc()
            logger.warning(
                "document_processing_retry",
                document_id=str(document_id),
                task_id=str(task_id),
                attempt=attempt,
            )
            raise Retry(defer=attempt * 2) from None
        TASKS.labels("failed").inc()
        return {"status": "failed", "error_code": "storage_unavailable"}

    try:
        pages = parse_document(raw_data, media_type=media_type)
    except DocumentParseError as exc:
        TASKS.labels("failed").inc()
        logger.warning(
            "document_processing_failed",
            document_id=str(document_id),
            task_id=str(task_id),
            error_code=str(exc),
        )
        return await _record_parse_failure(worker, document_id, task_id, str(exc))

    redacted_pages = [
        ParsedPage(
            page=page.page,
            blocks=tuple((section, redact_pii(text).text) for section, text in page.blocks),
        )
        for page in pages
    ]
    parsed_chunks = chunk_pages(redacted_pages)
    chunk_models: list[Chunk] = []
    evidence_chunks: list[EvidenceChunk] = []
    for parsed in parsed_chunks:
        chunk_id = uuid4()
        embedding = worker.embeddings.embed(parsed.text)
        chunk_models.append(
            Chunk(
                id=chunk_id,
                document_id=document_id,
                page=parsed.page,
                section=parsed.section,
                ordinal=parsed.ordinal,
                text=parsed.text,
                embedding=embedding,
            )
        )
        evidence_chunks.append(
            EvidenceChunk(
                id=str(chunk_id),
                document_id=str(document_id),
                document_name=document_name,
                page=parsed.page,
                section=parsed.section,
                text=parsed.text,
                embedding=embedding,
            )
        )
    extraction = extract_supplier_fields(evidence_chunks)

    async with worker.session_factory() as session:
        task = await session.scalar(
            select(ProcessingTask).where(ProcessingTask.id == task_id).with_for_update()
        )
        document = await session.get(Document, document_id)
        if task is None or document is None:
            TASKS.labels("missing").inc()
            return {"status": "missing"}
        if document.status == DocumentStatus.COMPLETED:
            return {
                "status": "already_completed",
                "chunks": await _chunk_count(session, document_id),
            }
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await session.execute(delete(Extraction).where(Extraction.document_id == document_id))
        session.add_all(chunk_models)
        session.add(
            Extraction(
                document_id=document_id,
                schema_version="supplier-v1",
                payload=asdict(extraction),
            )
        )
        completed_at = datetime.now(UTC)
        task.status = TaskStatus.COMPLETED
        task.completed_at = completed_at
        task.error_code = None
        document.status = DocumentStatus.COMPLETED
        document.completed_at = completed_at
        document.error_code = None
        session.add(
            AuditEvent(
                actor_id=None,
                action="document.process",
                document_id=document_id,
                result="completed",
                correlation_id=task.correlation_id,
                metadata_safe={"chunks": len(chunk_models), "attempt": task.attempts},
            )
        )
        await session.commit()
    duration_ms = round((perf_counter() - started) * 1_000, 3)
    TASKS.labels("completed").inc()
    logger.info(
        "document_processing_completed",
        document_id=str(document_id),
        task_id=str(task_id),
        chunks=len(chunk_models),
        duration_ms=duration_ms,
    )
    return {"status": "completed", "chunks": len(chunk_models)}
