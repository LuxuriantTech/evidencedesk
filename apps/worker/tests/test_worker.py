import asyncio
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from arq import Retry
from evidencedesk_api.config import Settings
from evidencedesk_api.db import build_engine, build_session_factory
from evidencedesk_api.extraction import extract_supplier_fields
from evidencedesk_api.models import (
    Chunk,
    Document,
    DocumentStatus,
    Dossier,
    ProcessingTask,
    TaskStatus,
)
from evidencedesk_api.providers import DeterministicEmbeddingProvider
from evidencedesk_api.storage import LocalDocumentStorage
from evidencedesk_worker.jobs import (
    WorkerContext,
    _record_parse_failure,
    _record_processing_failure,
    _record_storage_failure,
    process_document,
)
from sqlalchemy import func, select, text

DATABASE_URL = (
    "postgresql+asyncpg://evidencedesk:evidencedesk-local-only@127.0.0.1:55432/evidencedesk"
)


class FailsOnceStorage(LocalDocumentStorage):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.failed = False

    def get(self, key: str) -> bytes:
        if not self.failed:
            self.failed = True
            raise OSError("synthetic transient failure containing no document data")
        return super().get(key)


class AlwaysUnavailableStorage(LocalDocumentStorage):
    def get(self, key: str) -> bytes:
        raise OSError("synthetic persistent storage outage")


class AlwaysFailsEmbedding:
    mode = "synthetic-failure"
    model_id = "synthetic-failure@tests"
    dimension = 384
    estimated_cost_usd = 0.0

    def embed(self, _text: str) -> list[float]:
        raise RuntimeError("synthetic provider failure with no document data")

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(texts)


class InvalidDimensionEmbedding:
    mode = "synthetic-invalid-dimension"
    model_id = "synthetic-invalid-dimension@tests"
    dimension = 384
    estimated_cost_usd = 0.0

    def embed(self, _text: str) -> list[float]:
        return [1.0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(texts)


async def _prepare(
    tmp_path: Path, *, storage_type: type[LocalDocumentStorage] = LocalDocumentStorage
) -> tuple[WorkerContext, UUID, UUID]:
    settings = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path,
        jwt_secret="worker-test-secret-at-least-32-characters",
        demo_admin_password="not-used-admin",
        demo_analyst_password="not-used-analyst",
        demo_reader_password="not-used-reader",
    )
    engine = build_engine(settings)
    factory = build_session_factory(engine)
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "TRUNCATE audit_events, processing_tasks, extractions, chunks, documents, "
                "dossiers, users, evaluation_runs CASCADE"
            )
        )
    dossier_id = uuid4()
    document_id = uuid4()
    task_id = uuid4()
    correlation_id = uuid4()
    storage = storage_type(tmp_path)
    storage_key = f"{document_id}/document.txt"
    storage.put(
        storage_key,
        b"Organization: Northwind Response Systems Ltd.\n"
        b"Document type: Supplier Agreement\n"
        b"Renewal date: 2027-05-31.\n"
        b"Obligation: Notify incidents within four hours.\n"
        b"Contact: synthetic.person@example.test.",
    )
    async with factory() as session:
        session.add(Dossier(id=dossier_id, name="Synthetic", description="", is_synthetic=True))
        await session.flush()
        session.add(
            Document(
                id=document_id,
                dossier_id=dossier_id,
                filename="supplier.txt",
                media_type="text/plain",
                size_bytes=220,
                content_sha256="a" * 64,
                storage_key=storage_key,
                status=DocumentStatus.QUEUED,
                is_synthetic=True,
                correlation_id=correlation_id,
            )
        )
        session.add(
            ProcessingTask(
                id=task_id,
                document_id=document_id,
                status=TaskStatus.QUEUED,
                attempts=0,
                correlation_id=correlation_id,
            )
        )
        await session.commit()
    return (
        WorkerContext(
            settings=settings,
            engine=engine,
            session_factory=factory,
            storage=storage,
            embeddings=DeterministicEmbeddingProvider(dimension=384),
        ),
        document_id,
        task_id,
    )


@pytest.mark.integration
def test_worker_processes_redacts_and_is_idempotent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    async def scenario() -> None:
        context, document_id, task_id = await _prepare(tmp_path)
        try:
            first = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            second = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            async with context.session_factory() as session:
                document = await session.get(Document, document_id)
                task = await session.get(ProcessingTask, task_id)
                chunks = (
                    await session.scalars(select(Chunk).where(Chunk.document_id == document_id))
                ).all()
                assert document is not None and document.status == DocumentStatus.COMPLETED
                assert task is not None and task.status == TaskStatus.COMPLETED
                assert task.attempts == 1
                assert chunks
                assert all("synthetic.person@example.test" not in chunk.text for chunk in chunks)
                assert any("[EMAIL REDACTED]" in chunk.text for chunk in chunks)
                assert first == {"status": "completed", "chunks": len(chunks)}
                assert second == {"status": "already_completed", "chunks": len(chunks)}
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())
    logs = capsys.readouterr().out
    assert "document_processing_completed" in logs
    assert "synthetic.person@example.test" not in logs


@pytest.mark.integration
def test_worker_routes_grounded_v3_mode_to_v3_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evidencedesk_worker.jobs as jobs

    async def scenario() -> None:
        context, document_id, task_id = await _prepare(tmp_path)
        context.settings.answer_mode = "grounded-local-v3"
        calls: list[object] = []

        def v3(chunks: list[object], *, embeddings: object) -> object:
            calls.append(embeddings)
            return extract_supplier_fields(chunks)  # type: ignore[arg-type]

        monkeypatch.setattr(jobs, "extract_supplier_fields_v3", v3)
        try:
            result = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            assert result["status"] == "completed"
            assert calls == [context.embeddings]
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())


@pytest.mark.integration
def test_worker_retries_transient_storage_error_then_resumes(tmp_path: Path) -> None:
    async def scenario() -> None:
        context, document_id, task_id = await _prepare(tmp_path, storage_type=FailsOnceStorage)
        try:
            with pytest.raises(Retry):
                await process_document(
                    {"worker": context}, str(document_id), str(task_id), str(uuid4())
                )
            result = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            async with context.session_factory() as session:
                task = await session.get(ProcessingTask, task_id)
                count = await session.scalar(
                    select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
                )
                assert task is not None and task.attempts == 2
                assert task.status == TaskStatus.COMPLETED
                assert count and count > 0
                assert result["status"] == "completed"
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())


@pytest.mark.integration
def test_worker_stops_after_three_storage_attempts(tmp_path: Path) -> None:
    async def scenario() -> None:
        context, document_id, task_id = await _prepare(
            tmp_path, storage_type=AlwaysUnavailableStorage
        )
        try:
            for _attempt in range(2):
                with pytest.raises(Retry):
                    await process_document(
                        {"worker": context}, str(document_id), str(task_id), str(uuid4())
                    )
            result = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            async with context.session_factory() as session:
                task = await session.get(ProcessingTask, task_id)
                document = await session.get(Document, document_id)
                assert task is not None and task.attempts == 3
                assert task.status == TaskStatus.FAILED
                assert task.error_code == "storage_unavailable"
                assert document is not None and document.status == DocumentStatus.FAILED
                assert result == {"status": "failed", "error_code": "storage_unavailable"}
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())


@pytest.mark.integration
def test_worker_reports_invalid_pdf_and_missing_job(tmp_path: Path) -> None:
    async def scenario() -> None:
        context, document_id, task_id = await _prepare(tmp_path)
        try:
            async with context.session_factory() as session:
                document = await session.get(Document, document_id)
                assert document is not None
                document.media_type = "application/pdf"
                await session.commit()
            failed = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            assert failed["status"] == "failed"
            assert failed["error_code"] == "invalid_pdf"

            missing = await process_document(
                {"worker": context}, str(uuid4()), str(uuid4()), str(uuid4())
            )
            assert missing == {"status": "missing"}
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())


@pytest.mark.integration
def test_worker_retries_processing_failure_then_marks_terminal_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        context, document_id, task_id = await _prepare(tmp_path)
        context.embeddings = AlwaysFailsEmbedding()
        try:
            for _attempt in range(2):
                with pytest.raises(Retry):
                    await process_document(
                        {"worker": context}, str(document_id), str(task_id), str(uuid4())
                    )
            result = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            async with context.session_factory() as session:
                task = await session.get(ProcessingTask, task_id)
                document = await session.get(Document, document_id)
                count = await session.scalar(
                    select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
                )
                assert task is not None and task.attempts == 3
                assert task.status == TaskStatus.FAILED
                assert task.error_code == "processing_failed"
                assert document is not None and document.status == DocumentStatus.FAILED
                assert count == 0
                assert result == {"status": "failed", "error_code": "processing_failed"}
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())


@pytest.mark.integration
def test_worker_never_processes_a_deleted_document(tmp_path: Path) -> None:
    async def scenario() -> None:
        context, document_id, task_id = await _prepare(tmp_path)
        try:
            async with context.session_factory() as session:
                document = await session.get(Document, document_id)
                assert document is not None
                document.status = DocumentStatus.DELETED
                document.deleted_at = document.created_at
                await session.commit()

            result = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            async with context.session_factory() as session:
                task = await session.get(ProcessingTask, task_id)
                count = await session.scalar(
                    select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
                )
                assert task is not None and task.status == TaskStatus.FAILED
                assert task.error_code == "document_deleted"
                assert count == 0
                assert result == {"status": "deleted"}
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())


@pytest.mark.integration
def test_worker_retries_database_write_failure_then_marks_terminal_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        context, document_id, task_id = await _prepare(tmp_path)
        context.embeddings = InvalidDimensionEmbedding()
        try:
            for _attempt in range(2):
                with pytest.raises(Retry):
                    await process_document(
                        {"worker": context}, str(document_id), str(task_id), str(uuid4())
                    )
            result = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            async with context.session_factory() as session:
                task = await session.get(ProcessingTask, task_id)
                document = await session.get(Document, document_id)
                assert task is not None and task.attempts == 3
                assert task.status == TaskStatus.FAILED
                assert task.error_code == "processing_failed"
                assert document is not None and document.status == DocumentStatus.FAILED
                assert result == {"status": "failed", "error_code": "processing_failed"}
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())


@pytest.mark.integration
def test_worker_treats_document_budget_overflow_as_terminal(tmp_path: Path) -> None:
    async def scenario() -> None:
        context, document_id, task_id = await _prepare(tmp_path)
        context.settings.max_document_chunks = 1
        try:
            result = await process_document(
                {"worker": context}, str(document_id), str(task_id), str(uuid4())
            )
            async with context.session_factory() as session:
                task = await session.get(ProcessingTask, task_id)
                document = await session.get(Document, document_id)
                count = await session.scalar(
                    select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
                )
                assert task is not None and task.attempts == 1
                assert task.status == TaskStatus.FAILED
                assert task.error_code == "too_many_chunks"
                assert document is not None and document.status == DocumentStatus.FAILED
                assert count == 0
                assert result == {"status": "failed", "error_code": "too_many_chunks"}
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())


@pytest.mark.integration
def test_late_failure_handlers_never_revive_a_deleted_document(tmp_path: Path) -> None:
    async def scenario() -> None:
        context, document_id, task_id = await _prepare(tmp_path)
        try:
            async with context.session_factory() as session:
                document = await session.get(Document, document_id)
                assert document is not None
                document.status = DocumentStatus.DELETED
                document.deleted_at = document.created_at
                await session.commit()

            outcomes = [
                await _record_storage_failure(
                    context, document_id, task_id, retry=True
                ),
                await _record_processing_failure(
                    context, document_id, task_id, retry=True
                ),
                await _record_parse_failure(
                    context, document_id, task_id, "invalid_pdf"
                ),
            ]

            async with context.session_factory() as session:
                document = await session.get(Document, document_id)
                task = await session.get(ProcessingTask, task_id)
                assert document is not None and document.status == DocumentStatus.DELETED
                assert document.deleted_at is not None
                assert task is not None and task.status == TaskStatus.FAILED
                assert task.error_code == "document_deleted"
                assert outcomes == [
                    {"status": "deleted"},
                    {"status": "deleted"},
                    {"status": "deleted"},
                ]
        finally:
            await context.engine.dispose()

    asyncio.run(scenario())
