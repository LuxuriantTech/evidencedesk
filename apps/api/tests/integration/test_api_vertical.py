import asyncio
import hashlib
import io
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from evidencedesk_api import main as api_main
from evidencedesk_api.answering import DeterministicGroundedAnswerProvider
from evidencedesk_api.config import Settings
from evidencedesk_api.db import build_engine, build_session_factory
from evidencedesk_api.main import create_app
from evidencedesk_api.models import (
    Chunk,
    Document,
    DocumentStatus,
    Extraction,
    ProcessingTask,
    TaskStatus,
    User,
)
from evidencedesk_api.provider_registry import ProviderBundle
from evidencedesk_api.providers import DeterministicEmbeddingProvider
from evidencedesk_api.queueing import TaskQueue
from evidencedesk_api.retrieval import EvidenceChunk, ExtractiveAnswerProvider
from evidencedesk_api.security import verify_password
from evidencedesk_api.seed import seed_demo_data
from evidencedesk_api.storage import LocalDocumentStorage
from evidencedesk_worker import jobs as worker_jobs
from evidencedesk_worker.jobs import WorkerContext, process_document
from reportlab.pdfgen import canvas
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.testclient import TestClient

DATABASE_URL = (
    "postgresql+asyncpg://evidencedesk:evidencedesk-local-only@127.0.0.1:55432/evidencedesk"
)


class RecordingQueue(TaskQueue):
    def __init__(self) -> None:
        self.jobs: list[tuple[UUID, UUID, UUID]] = []

    async def enqueue_document(
        self, document_id: UUID, task_id: UUID, correlation_id: UUID
    ) -> None:
        self.jobs.append((document_id, task_id, correlation_id))

    async def close(self) -> None:
        return None

    async def ping(self) -> bool:
        return True


class UnavailableQueue(RecordingQueue):
    async def enqueue_document(
        self, document_id: UUID, task_id: UUID, correlation_id: UUID
    ) -> None:
        raise OSError("synthetic queue outage")

    async def ping(self) -> bool:
        return False


async def _reset_and_seed(settings: Settings) -> None:
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "TRUNCATE audit_events, processing_tasks, extractions, chunks, documents, "
                "dossiers, users, evaluation_runs CASCADE"
            )
        )
    await seed_demo_data(settings, engine=engine)
    await engine.dispose()


@pytest.fixture
def api(tmp_path: Path) -> Generator[tuple[TestClient, RecordingQueue, Settings], None, None]:
    settings = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path,
        jwt_secret="integration-secret-at-least-32-characters",
        public_demo_mode=False,
        answer_mode="extractive-local-hash",
        demo_admin_password="EvidenceDemo-Admin-2026!",
        demo_analyst_password="EvidenceDemo-Analyst-2026!",
        demo_reader_password="EvidenceDemo-Reader-2026!",
    )
    asyncio.run(_reset_and_seed(settings))
    queue = RecordingQueue()
    with TestClient(create_app(settings=settings, task_queue=queue)) as client:
        yield client, queue, settings


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/token", data={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return str(response.json()["access_token"])


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
def test_reseeding_refreshes_configured_demo_password(tmp_path: Path) -> None:
    initial = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path,
        jwt_secret="integration-secret-at-least-32-characters",
        demo_admin_password="InitialDemo-Admin-2026!",
        demo_analyst_password="InitialDemo-Analyst-2026!",
        demo_reader_password="InitialDemo-Reader-2026!",
    )
    asyncio.run(_reset_and_seed(initial))
    refreshed = initial.model_copy(
        update={"demo_admin_password": "RefreshedDemo-Admin-2026!"}
    )

    async def scenario() -> str:
        engine = build_engine(refreshed)
        try:
            await seed_demo_data(refreshed, engine=engine)
            async with build_session_factory(engine)() as session:
                admin = await session.scalar(
                    select(User).where(User.username == "demo.admin")
                )
                assert admin is not None
                return admin.password_hash
        finally:
            await engine.dispose()

    password_hash = asyncio.run(scenario())
    assert verify_password("RefreshedDemo-Admin-2026!", password_hash)
    assert not verify_password("InitialDemo-Admin-2026!", password_hash)


@pytest.mark.integration
def test_public_demo_seed_deactivates_existing_admin_account(tmp_path: Path) -> None:
    enabled = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path,
        jwt_secret="integration-secret-at-least-32-characters",
        public_demo_mode=True,
        demo_admin_enabled=True,
        demo_admin_password="EvidenceDemo-Admin-2026!",
        demo_analyst_password="EvidenceDemo-Analyst-2026!",
        demo_reader_password="EvidenceDemo-Reader-2026!",
    )
    asyncio.run(_reset_and_seed(enabled))
    public = enabled.model_copy(
        update={"demo_admin_enabled": False, "demo_reader_enabled": False}
    )

    async def scenario() -> tuple[bool, bool, bool]:
        engine = build_engine(public)
        try:
            await seed_demo_data(public, engine=engine)
            async with build_session_factory(engine)() as session:
                admin = await session.scalar(select(User).where(User.username == "demo.admin"))
                analyst = await session.scalar(
                    select(User).where(User.username == "demo.analyst")
                )
                reader = await session.scalar(select(User).where(User.username == "demo.reader"))
                assert admin is not None and analyst is not None and reader is not None
                return admin.is_active, analyst.is_active, reader.is_active
        finally:
            await engine.dispose()

    admin_active, analyst_active, reader_active = asyncio.run(scenario())
    assert admin_active is False
    assert analyst_active is True
    assert reader_active is False


@pytest.mark.integration
def test_auth_upload_idempotence_audit_and_deletion(
    api: tuple[TestClient, RecordingQueue, Settings],
) -> None:
    client, queue, settings = api
    reader = _token(client, "demo.reader", "EvidenceDemo-Reader-2026!")
    analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
    admin = _token(client, "demo.admin", "EvidenceDemo-Admin-2026!")

    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"
    metrics = client.get("/metrics/")
    assert metrics.status_code == 200
    assert "evidencedesk_http_requests_total" in metrics.text

    dossiers = client.get("/api/v1/dossiers", headers=_headers(reader))
    assert dossiers.status_code == 200
    dossier_id = dossiers.json()[0]["id"]

    denied = client.post(
        f"/api/v1/dossiers/{dossier_id}/documents",
        headers=_headers(reader),
        files={"file": ("notes.md", b"# Synthetic\n\nRenewal date: 2027-05-31", "text/markdown")},
        data={"is_synthetic": "true"},
    )
    assert denied.status_code == 403

    first = client.post(
        f"/api/v1/dossiers/{dossier_id}/documents",
        headers=_headers(analyst),
        files={
            "file": ("../notes.md", b"# Synthetic\n\nRenewal date: 2027-05-31", "text/markdown")
        },
        data={"is_synthetic": "true"},
    )
    assert first.status_code == 202, first.text
    assert first.json()["status"] == "queued"
    assert first.json()["filename"] == "notes.md"
    assert len(queue.jobs) == 1

    duplicate = client.post(
        f"/api/v1/dossiers/{dossier_id}/documents",
        headers=_headers(analyst),
        files={"file": ("copy.md", b"# Synthetic\n\nRenewal date: 2027-05-31", "text/markdown")},
        data={"is_synthetic": "true"},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["id"] == first.json()["id"]
    assert len(queue.jobs) == 1

    audit = client.get("/api/v1/audit-events", headers=_headers(admin))
    assert audit.status_code == 200
    assert {event["action"] for event in audit.json()} >= {
        "auth.login",
        "document.upload",
        "authorization.denied",
    }

    assert client.get("/api/v1/status", headers=_headers(reader)).status_code == 403
    status_response = client.get("/api/v1/status", headers=_headers(admin))
    assert status_response.status_code == 200
    assert status_response.json()["mode"] == "extractive-local-hash"
    assert status_response.json()["documents"]["queued"] == 1

    deleted = client.delete(f"/api/v1/documents/{first.json()['id']}", headers=_headers(admin))
    assert deleted.status_code == 204

    async def assert_task_cancelled() -> None:
        engine = build_engine(settings)
        try:
            async with build_session_factory(engine)() as session:
                task = await session.get(ProcessingTask, UUID(first.json()["task_id"]))
                assert task is not None
                assert task.status == TaskStatus.FAILED
                assert task.error_code == "document_deleted"
        finally:
            await engine.dispose()

    asyncio.run(assert_task_cancelled())


@pytest.mark.integration
def test_api_rejects_an_oversize_multipart_before_route_parsing(
    api: tuple[TestClient, RecordingQueue, Settings],
) -> None:
    client, queue, settings = api
    analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
    dossier_id = client.get("/api/v1/dossiers", headers=_headers(analyst)).json()[0]["id"]

    response = client.post(
        f"/api/v1/dossiers/{dossier_id}/documents",
        headers=_headers(analyst),
        files={
            "file": (
                "oversize.txt",
                b"x" * (settings.max_request_bytes + 1),
                "text/plain",
            )
        },
        data={"is_synthetic": "true"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_too_large"
    assert queue.jobs == []


@pytest.mark.integration
def test_deletion_waits_for_worker_persistence_and_removes_derived_rows(
    api: tuple[TestClient, RecordingQueue, Settings],
) -> None:
    client, queue, settings = api
    analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
    admin = _token(client, "demo.admin", "EvidenceDemo-Admin-2026!")
    dossier_id = client.get("/api/v1/dossiers", headers=_headers(analyst)).json()[0]["id"]
    uploaded = client.post(
        f"/api/v1/dossiers/{dossier_id}/documents",
        headers=_headers(analyst),
        files={"file": ("race.txt", b"Synthetic content for deletion race.", "text/plain")},
        data={"is_synthetic": "true"},
    )
    assert uploaded.status_code == 202
    document_id, task_id, _correlation_id = queue.jobs[0]

    async def scenario() -> None:
        engine = build_engine(settings)
        factory = build_session_factory(engine)
        try:
            async with factory() as worker_session:
                task = await worker_session.scalar(
                    select(ProcessingTask)
                    .where(ProcessingTask.id == task_id)
                    .with_for_update()
                )
                document = await worker_session.scalar(
                    select(Document).where(Document.id == document_id).with_for_update()
                )
                assert task is not None and document is not None
                worker_session.add(
                    Chunk(
                        document_id=document_id,
                        page=1,
                        section=None,
                        ordinal=0,
                        text="Synthetic content for deletion race.",
                        embedding=[0.0] * 384,
                    )
                )
                worker_session.add(
                    Extraction(
                        document_id=document_id,
                        schema_version="supplier-v1",
                        payload={"synthetic": True},
                    )
                )
                task.status = TaskStatus.COMPLETED
                document.status = DocumentStatus.COMPLETED
                await worker_session.flush()

                deletion = asyncio.create_task(
                    asyncio.to_thread(
                        client.delete,
                        f"/api/v1/documents/{document_id}",
                        headers=_headers(admin),
                    )
                )
                await asyncio.sleep(0.2)
                assert not deletion.done()
                await worker_session.commit()
                response = await asyncio.wait_for(deletion, timeout=5)
                assert response.status_code == 204

            async with factory() as verification_session:
                document = await verification_session.get(Document, document_id)
                chunk_count = await verification_session.scalar(
                    select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
                )
                extraction_count = await verification_session.scalar(
                    select(func.count(Extraction.id)).where(Extraction.document_id == document_id)
                )
                assert document is not None and document.status == DocumentStatus.DELETED
                assert chunk_count == 0
                assert extraction_count == 0
        finally:
            await engine.dispose()

    asyncio.run(scenario())


@pytest.mark.integration
def test_public_demo_rejects_unattested_upload(
    tmp_path: Path,
) -> None:
    approved = b"Approved, versioned synthetic supplier note."
    allowlist = tmp_path / "public-demo.json"
    allowlist.write_text(
        json.dumps(
            {
                "synthetic_only": True,
                "documents": [
                    {
                        "filename": "approved.txt",
                        "sha256": hashlib.sha256(approved).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path / "storage",
        jwt_secret="integration-secret-at-least-32-characters",
        public_demo_mode=True,
        answer_mode="extractive-local-hash",
        public_demo_allowlist=allowlist,
        demo_admin_password="EvidenceDemo-Admin-2026!",
        demo_analyst_password="EvidenceDemo-Analyst-2026!",
        demo_reader_password="EvidenceDemo-Reader-2026!",
    )
    asyncio.run(_reset_and_seed(settings))
    queue = RecordingQueue()

    with TestClient(create_app(settings=settings, task_queue=queue)) as client:
        analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
        dossier_id = client.get("/api/v1/dossiers", headers=_headers(analyst)).json()[0]["id"]

        unattested = client.post(
            f"/api/v1/dossiers/{dossier_id}/documents",
            headers=_headers(analyst),
            files={"file": ("unapproved.txt", b"unapproved content", "text/plain")},
            data={"is_synthetic": "false"},
        )
        assert unattested.status_code == 422
        assert unattested.json()["detail"]["code"] == "synthetic_attestation_required"

        falsely_attested = client.post(
            f"/api/v1/dossiers/{dossier_id}/documents",
            headers=_headers(analyst),
            files={"file": ("unapproved.txt", b"unapproved content", "text/plain")},
            data={"is_synthetic": "true"},
        )
        assert falsely_attested.status_code == 422
        assert falsely_attested.json()["detail"]["code"] == "public_demo_file_not_approved"

        accepted = client.post(
            f"/api/v1/dossiers/{dossier_id}/documents",
            headers=_headers(analyst),
            files={"file": ("approved.txt", approved, "text/plain")},
            data={"is_synthetic": "true"},
        )
        assert accepted.status_code == 202


@pytest.mark.integration
def test_processed_document_can_be_queried_with_exact_evidence(
    api: tuple[TestClient, RecordingQueue, Settings],
) -> None:
    client, queue, settings = api
    analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
    reader = _token(client, "demo.reader", "EvidenceDemo-Reader-2026!")
    dossier_id = client.get("/api/v1/dossiers", headers=_headers(reader)).json()[0]["id"]
    source = (
        b"Organization: Northwind Response Systems Ltd.\n"
        b"Document type: Supplier Agreement.\n"
        b"Renewal date: 2027-05-31.\n"
        b"Obligation: The supplier must notify incidents within four hours.\n"
        b"Contact: synthetic.person@example.test."
    )
    uploaded = client.post(
        f"/api/v1/dossiers/{dossier_id}/documents",
        headers=_headers(analyst),
        files={"file": ("supplier.txt", source, "text/plain")},
        data={"is_synthetic": "true"},
    )
    assert uploaded.status_code == 202
    document_id, task_id, correlation_id = queue.jobs[0]

    async def run_worker() -> None:
        engine = build_engine(settings)
        worker = WorkerContext(
            settings=settings,
            engine=engine,
            session_factory=build_session_factory(engine),
            storage=LocalDocumentStorage(settings.storage_root),
            embeddings=DeterministicEmbeddingProvider(dimension=384),
        )
        try:
            result = await process_document(
                {"worker": worker}, str(document_id), str(task_id), str(correlation_id)
            )
            assert result["status"] == "completed"
        finally:
            await engine.dispose()

    asyncio.run(run_worker())

    async def mark_chunks_as_legacy() -> None:
        engine = build_engine(settings)
        try:
            async with build_session_factory(engine)() as session:
                chunks = (
                    await session.scalars(select(Chunk).where(Chunk.document_id == document_id))
                ).all()
                assert chunks
                for chunk in chunks:
                    chunk.embedding_model_id = "deterministic-hash-v0:384"
                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(mark_chunks_as_legacy())

    answer = client.post(
        f"/api/v1/dossiers/{dossier_id}/ask",
        headers=_headers(reader),
        json={"question": "How quickly must the supplier notify incidents?"},
    )
    assert answer.status_code == 200, answer.text
    payload = answer.json()
    assert payload["status"] == "answered"
    assert payload["mode"] == "extractive-local-hash"
    assert payload["citations"][0]["document_id"] == str(document_id)
    assert payload["citations"][0]["page"] == 1
    assert payload["citations"][0]["excerpt"] in payload["answer"]

    missing = client.post(
        f"/api/v1/dossiers/{dossier_id}/ask",
        headers=_headers(reader),
        json={"question": "What is the cyber-insurance deductible?"},
    )
    assert missing.status_code == 200
    assert missing.json()["status"] == "abstained"
    assert missing.json()["citations"] == []

    extraction = client.get(f"/api/v1/dossiers/{dossier_id}/extraction", headers=_headers(reader))
    assert extraction.status_code == 200
    assert extraction.json()[0]["payload"]["organization_name"]["value"] == (
        "Northwind Response Systems Ltd."
    )

    content = client.get(f"/api/v1/documents/{document_id}/content", headers=_headers(reader))
    assert content.status_code == 200
    serialized = str(content.json())
    assert "synthetic.person@example.test" not in serialized
    assert "[EMAIL REDACTED]" in serialized

    admin = _token(client, "demo.admin", "EvidenceDemo-Admin-2026!")
    events = client.get("/api/v1/audit-events", headers=_headers(admin)).json()
    observed = {(event["action"], event["document_id"], event["actor_id"]) for event in events}
    assert (
        "document.content.view",
        str(document_id),
        client.get("/api/v1/auth/me", headers=_headers(reader)).json()["id"],
    ) in observed
    assert any(
        action == "document.extraction.view" and event_document_id == str(document_id)
        for action, event_document_id, _actor_id in observed
    )


@pytest.mark.integration
def test_grounded_api_provider_abstains_on_negation_and_cites_explicit_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path,
        jwt_secret="integration-secret-at-least-32-characters",
        public_demo_mode=False,
        answer_mode="grounded-local-v3",
        demo_admin_password="EvidenceDemo-Admin-2026!",
        demo_analyst_password="EvidenceDemo-Analyst-2026!",
        demo_reader_password="EvidenceDemo-Reader-2026!",
    )
    asyncio.run(_reset_and_seed(settings))
    queue = RecordingQueue()
    original_builder = cast(Any, api_main).build_provider_bundle
    embedding = DeterministicEmbeddingProvider(dimension=384)

    def build_with_local_test_embedding(mode: str, **kwargs: Any) -> ProviderBundle:
        return cast(
            ProviderBundle,
            original_builder(
                mode,
                embedding_factory=lambda: embedding,
                **kwargs,
            ),
        )

    monkeypatch.setattr(api_main, "build_provider_bundle", build_with_local_test_embedding)
    with TestClient(create_app(settings=settings, task_queue=queue)) as client:
        assert isinstance(
            cast(Any, client).app.state.providers.answer,
            DeterministicGroundedAnswerProvider,
        )
        analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
        reader = _token(client, "demo.reader", "EvidenceDemo-Reader-2026!")
        dossier_id = client.get("/api/v1/dossiers", headers=_headers(reader)).json()[0]["id"]
        source = (
            b"The service did not become effective on 27 August 2031.\n"
            b"Renewal date: not recorded."
        )
        uploaded = client.post(
            f"/api/v1/dossiers/{dossier_id}/documents",
            headers=_headers(analyst),
            files={"file": ("northstar_service_agreement.txt", source, "text/plain")},
            data={"is_synthetic": "true"},
        )
        assert uploaded.status_code == 202, uploaded.text
        document_id, task_id, correlation_id = queue.jobs[0]

        async def run_worker() -> None:
            engine = build_engine(settings)
            worker = WorkerContext(
                settings=settings,
                engine=engine,
                session_factory=build_session_factory(engine),
                storage=LocalDocumentStorage(settings.storage_root),
                embeddings=embedding,
            )
            try:
                result = await process_document(
                    {"worker": worker}, str(document_id), str(task_id), str(correlation_id)
                )
                assert result["status"] == "completed"
            finally:
                await engine.dispose()

        asyncio.run(run_worker())

        negated = client.post(
            f"/api/v1/dossiers/{dossier_id}/ask",
            headers=_headers(reader),
            json={"question": "When did the Northstar service become effective?"},
        )
        assert negated.status_code == 200, negated.text
        assert negated.json()["status"] == "abstained"
        assert negated.json()["citations"] == []

        absent = client.post(
            f"/api/v1/dossiers/{dossier_id}/ask",
            headers=_headers(reader),
            json={"question": "Is the renewal date listed for Northstar?"},
        )
        assert absent.status_code == 200, absent.text
        payload = absent.json()
        assert payload["status"] == "answered"
        assert payload["answer"] == "No. Renewal date: not recorded."
        assert payload["citations"][0]["document_id"] == str(document_id)
        assert payload["citations"][0]["excerpt"] == "Renewal date: not recorded."


@pytest.mark.integration
def test_grounded_api_binds_measurements_to_time_and_preserves_same_role_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path,
        jwt_secret="integration-secret-at-least-32-characters",
        public_demo_mode=False,
        answer_mode="grounded-local-v3",
        demo_admin_password="EvidenceDemo-Admin-2026!",
        demo_analyst_password="EvidenceDemo-Analyst-2026!",
        demo_reader_password="EvidenceDemo-Reader-2026!",
    )
    asyncio.run(_reset_and_seed(settings))
    queue = RecordingQueue()
    original_builder = cast(Any, api_main).build_provider_bundle
    embedding = DeterministicEmbeddingProvider(dimension=384)

    def build_with_local_test_embedding(mode: str, **kwargs: Any) -> ProviderBundle:
        return cast(
            ProviderBundle,
            original_builder(mode, embedding_factory=lambda: embedding, **kwargs),
        )

    monkeypatch.setattr(api_main, "build_provider_bundle", build_with_local_test_embedding)
    with TestClient(create_app(settings=settings, task_queue=queue)) as client:
        assert isinstance(
            cast(Any, client).app.state.providers.answer,
            DeterministicGroundedAnswerProvider,
        )
        analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
        reader = _token(client, "demo.reader", "EvidenceDemo-Reader-2026!")
        dossier_id = client.get("/api/v1/dossiers", headers=_headers(reader)).json()[0]["id"]
        source = (
            b"Station RQ-19/7 uses sensor series VTX-3/B.\n"
            b"At 08:15, pressure at station RQ-19/7 was 42 kPa.\n"
            b"At 08:45, pressure at station RQ-19/7 was 57 kPa.\n"
            b"Maintenance contact: not recorded."
        )
        uploaded = client.post(
            f"/api/v1/dossiers/{dossier_id}/documents",
            headers=_headers(analyst),
            files={"file": ("station_rq-19-7_monitoring.txt", source, "text/plain")},
            data={"is_synthetic": "true"},
        )
        assert uploaded.status_code == 202, uploaded.text
        document_id, task_id, correlation_id = queue.jobs[0]

        async def run_worker() -> None:
            engine = build_engine(settings)
            worker = WorkerContext(
                settings=settings,
                engine=engine,
                session_factory=build_session_factory(engine),
                storage=LocalDocumentStorage(settings.storage_root),
                embeddings=embedding,
            )
            try:
                result = await process_document(
                    {"worker": worker}, str(document_id), str(task_id), str(correlation_id)
                )
                assert result["status"] == "completed"
            finally:
                await engine.dispose()

        asyncio.run(run_worker())

        scoped = client.post(
            f"/api/v1/dossiers/{dossier_id}/ask",
            headers=_headers(reader),
            json={"question": "What pressure was recorded at 08:15 for station RQ-19/7?"},
        )
        assert scoped.status_code == 200, scoped.text
        scoped_payload = scoped.json()
        assert scoped_payload["status"] == "answered"
        assert "42 kPa" in scoped_payload["answer"]
        assert "57 kPa" not in scoped_payload["answer"]
        assert scoped_payload["citations"][0]["document_id"] == str(document_id)
        assert scoped_payload["citations"][0]["page"] == 1
        assert scoped_payload["citations"][0]["excerpt"] == (
            "At 08:15, pressure at station RQ-19/7 was 42 kPa."
        )
        assert "Maintenance contact" not in scoped_payload["answer"]

        unscoped = client.post(
            f"/api/v1/dossiers/{dossier_id}/ask",
            headers=_headers(reader),
            json={"question": "What pressure readings are recorded for station RQ-19/7?"},
        )
        assert unscoped.status_code == 200, unscoped.text
        ambiguous_payload = unscoped.json()
        assert ambiguous_payload["status"] == "ambiguous"
        assert len(ambiguous_payload["citations"]) == 2
        assert {item["document_id"] for item in ambiguous_payload["citations"]} == {
            str(document_id)
        }
        assert {item["page"] for item in ambiguous_payload["citations"]} == {1}
        cited_values = " ".join(item["excerpt"] for item in ambiguous_payload["citations"])
        assert "42 kPa" in cited_values
        assert "57 kPa" in cited_values


@pytest.mark.integration
def test_grounded_api_keeps_same_record_context_across_pdf_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path,
        jwt_secret="integration-secret-at-least-32-characters",
        public_demo_mode=False,
        answer_mode="grounded-local-v3",
        pdf_parse_timeout_seconds=30.0,
        demo_admin_password="EvidenceDemo-Admin-2026!",
        demo_analyst_password="EvidenceDemo-Analyst-2026!",
        demo_reader_password="EvidenceDemo-Reader-2026!",
    )
    asyncio.run(_reset_and_seed(settings))
    queue = RecordingQueue()
    original_builder = cast(Any, api_main).build_provider_bundle
    original_rank = cast(Any, api_main).hybrid_rank
    original_extract = cast(Any, worker_jobs).extract_supplier_fields_v3
    embedding = DeterministicEmbeddingProvider(dimension=384)
    provider_chunks: list[EvidenceChunk] = []
    worker_chunks: list[EvidenceChunk] = []

    def build_with_local_test_embedding(mode: str, **kwargs: Any) -> ProviderBundle:
        return cast(
            ProviderBundle,
            original_builder(mode, embedding_factory=lambda: embedding, **kwargs),
        )

    def capture_provider_chunks(
        question: str, chunks: list[EvidenceChunk], **kwargs: Any
    ) -> Any:
        provider_chunks.extend(chunks)
        return original_rank(question, chunks, **kwargs)

    def capture_worker_chunks(
        chunks: list[EvidenceChunk], *, embeddings: Any
    ) -> Any:
        worker_chunks.extend(chunks)
        return original_extract(chunks, embeddings=embeddings)

    monkeypatch.setattr(api_main, "build_provider_bundle", build_with_local_test_embedding)
    monkeypatch.setattr(api_main, "hybrid_rank", capture_provider_chunks)
    monkeypatch.setattr(worker_jobs, "extract_supplier_fields_v3", capture_worker_chunks)

    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.drawString(
        72,
        720,
        "Record XN-31/R applies to auxiliary sensor AUX-5 at the north archive intake.",
    )
    pdf.showPage()
    pdf.drawString(
        72,
        720,
        "At 07:35 on 18 June 2026, auxiliary sensor AUX-5 read 7.2 degrees Celsius.",
    )
    pdf.save()
    primary_pdf = pdf_buffer.getvalue()

    with TestClient(create_app(settings=settings, task_queue=queue)) as client:
        assert isinstance(
            cast(Any, client).app.state.providers.answer,
            DeterministicGroundedAnswerProvider,
        )
        analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
        reader = _token(client, "demo.reader", "EvidenceDemo-Reader-2026!")
        dossier_id = client.get("/api/v1/dossiers", headers=_headers(reader)).json()[0]["id"]

        primary_upload = client.post(
            f"/api/v1/dossiers/{dossier_id}/documents",
            headers=_headers(analyst),
            files={"file": ("xn-31-r_service_record.pdf", primary_pdf, "application/pdf")},
            data={"is_synthetic": "true"},
        )
        assert primary_upload.status_code == 202, primary_upload.text
        primary_document_id = primary_upload.json()["id"]

        competing_source = (
            b"Record XN-31/S applies to auxiliary sensor AUX-5 at the south archive intake.\n"
            b"At 07:35 on 18 June 2026, auxiliary sensor AUX-5 read 9.6 degrees Celsius.\n"
            b"Calibration date: 19 June 2026."
        )
        competing_upload = client.post(
            f"/api/v1/dossiers/{dossier_id}/documents",
            headers=_headers(analyst),
            files={
                "file": (
                    "xn-31-s_service_record.txt",
                    competing_source,
                    "text/plain",
                )
            },
            data={"is_synthetic": "true"},
        )
        assert competing_upload.status_code == 202, competing_upload.text
        competing_document_id = competing_upload.json()["id"]

        async def process_queued_documents() -> None:
            engine = build_engine(settings)
            worker = WorkerContext(
                settings=settings,
                engine=engine,
                session_factory=build_session_factory(engine),
                storage=LocalDocumentStorage(settings.storage_root),
                embeddings=embedding,
            )
            try:
                for document_id, task_id, correlation_id in queue.jobs:
                    result = await process_document(
                        {"worker": worker},
                        str(document_id),
                        str(task_id),
                        str(correlation_id),
                    )
                    assert result["status"] == "completed"
            finally:
                await engine.dispose()

        asyncio.run(process_queued_documents())

        worker_source_order = [
            (chunk.page, chunk.ordinal)
            for chunk in worker_chunks
            if chunk.document_id == str(primary_document_id)
        ]
        assert worker_source_order == [(1, 0), (2, 1)]

        answer = client.post(
            f"/api/v1/dossiers/{dossier_id}/ask",
            headers=_headers(reader),
            json={
                "question": (
                    "What temperature did record XN-31/R report for auxiliary sensor AUX-5 "
                    "at 07:35?"
                )
            },
        )
        assert answer.status_code == 200, answer.text
        answer_payload = answer.json()
        provider_source_order = [
            (chunk.page, chunk.ordinal)
            for chunk in provider_chunks
            if chunk.document_id == str(primary_document_id)
        ]
        assert set(provider_source_order) == {(1, 0), (2, 1)}
        assert answer_payload["status"] == "answered"
        assert "7.2" in answer_payload["answer"]
        assert "9.6" not in answer_payload["answer"]
        assert {item["document_id"] for item in answer_payload["citations"]} == {
            str(primary_document_id)
        }
        assert {item["page"] for item in answer_payload["citations"]} == {1, 2}
        excerpts = {item["excerpt"] for item in answer_payload["citations"]}
        assert (
            "Record XN-31/R applies to auxiliary sensor AUX-5 at the north archive intake."
            in excerpts
        )
        assert (
            "At 07:35 on 18 June 2026, auxiliary sensor AUX-5 read 7.2 degrees Celsius."
            in excerpts
        )
        assert "9.6" not in " ".join(excerpts)

        absent = client.post(
            f"/api/v1/dossiers/{dossier_id}/ask",
            headers=_headers(reader),
            json={"question": "What calibration date does record XN-31/R state?"},
        )
        assert absent.status_code == 200, absent.text
        assert absent.json()["status"] == "abstained"
        assert absent.json()["citations"] == []
        assert str(competing_document_id) != str(primary_document_id)


@pytest.mark.integration
def test_evaluation_endpoint_persists_only_calculated_results(
    api: tuple[TestClient, RecordingQueue, Settings],
) -> None:
    client, _queue, _settings = api
    reader = _token(client, "demo.reader", "EvidenceDemo-Reader-2026!")
    admin = _token(client, "demo.admin", "EvidenceDemo-Admin-2026!")

    denied = client.post(
        "/api/v1/evaluations/run", headers=_headers(reader), json={"split": "development"}
    )
    assert denied.status_code == 403

    probe = DeterministicEmbeddingProvider(dimension=384)
    probe.mode = "local-semantic-onnx-v1"
    probe.model_id = "semantic-probe@frozen"
    cast(Any, client).app.state.providers = ProviderBundle(
        embedding=probe,
        answer=ExtractiveAnswerProvider(mode="extractive-local-onnx"),
    )

    executed = client.post(
        "/api/v1/evaluations/run", headers=_headers(admin), json={"split": "development"}
    )
    assert executed.status_code == 201, executed.text
    result = executed.json()
    assert result["split"] == "development"
    assert result["mode"] == "extractive-local-onnx"
    assert result["metrics"]["case_count"] == 50
    assert result["metrics"]["embedding_model_id"] == "semantic-probe@frozen"
    assert result["metrics"]["retrieval_method"] == "hybrid"
    assert result["metrics"]["estimated_cost_usd"] == 0.0
    assert result["metrics"]["citation_precision"] >= 0

    listed = client.get("/api/v1/evaluations", headers=_headers(reader))
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == result["id"]


@pytest.mark.integration
def test_authentication_and_missing_resources_fail_closed(
    api: tuple[TestClient, RecordingQueue, Settings],
) -> None:
    client, _queue, _settings = api

    assert client.get("/api/v1/auth/me").status_code == 401
    assert (
        client.post(
            "/api/v1/auth/token",
            data={"username": "demo.reader", "password": "incorrect"},
        ).status_code
        == 401
    )
    reader = _token(client, "demo.reader", "EvidenceDemo-Reader-2026!")
    me = client.get("/api/v1/auth/me", headers=_headers(reader))
    assert me.status_code == 200
    assert me.json()["role"] == "reader"

    unknown = "00000000-0000-0000-0000-000000000001"
    assert client.get(f"/api/v1/documents/{unknown}", headers=_headers(reader)).status_code == 404
    assert (
        client.get(f"/api/v1/documents/{unknown}/content", headers=_headers(reader)).status_code
        == 404
    )
    assert (
        client.get(f"/api/v1/dossiers/{unknown}/documents", headers=_headers(reader)).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/v1/dossiers/{unknown}/ask",
            headers=_headers(reader),
            json={"question": "Is any evidence available?"},
        ).status_code
        == 404
    )


@pytest.mark.integration
def test_unavailable_queue_fails_readiness_and_marks_upload_failed(tmp_path: Path) -> None:
    settings = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path,
        jwt_secret="integration-secret-at-least-32-characters",
        public_demo_mode=False,
        answer_mode="extractive-local-hash",
        demo_admin_password="EvidenceDemo-Admin-2026!",
        demo_analyst_password="EvidenceDemo-Analyst-2026!",
        demo_reader_password="EvidenceDemo-Reader-2026!",
    )
    asyncio.run(_reset_and_seed(settings))
    queue = UnavailableQueue()

    with TestClient(create_app(settings=settings, task_queue=queue)) as client:
        assert client.get("/ready").status_code == 503
        analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
        admin = _token(client, "demo.admin", "EvidenceDemo-Admin-2026!")
        dossier_id = client.get("/api/v1/dossiers", headers=_headers(analyst)).json()[0]["id"]
        response = client.post(
            f"/api/v1/dossiers/{dossier_id}/documents",
            headers=_headers(analyst),
            files={"file": ("queue-test.txt", b"synthetic content", "text/plain")},
            data={"is_synthetic": "true"},
        )
        assert response.status_code == 503
        system_status = client.get("/api/v1/status", headers=_headers(admin)).json()
        assert system_status["documents"]["failed"] == 1
        audit = client.get("/api/v1/audit-events", headers=_headers(admin)).json()
        assert any(
            event["action"] == "document.enqueue" and event["result"] == "failed" for event in audit
        )
