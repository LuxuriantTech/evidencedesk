import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from arq import Retry
from evidencedesk_api.config import Settings
from evidencedesk_api.db import build_engine, build_session_factory
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
from evidencedesk_worker.jobs import WorkerContext, process_document
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
