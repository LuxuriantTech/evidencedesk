import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from evidencedesk_api.config import Settings
from evidencedesk_api.db import build_engine, build_session_factory
from evidencedesk_api.main import create_app
from evidencedesk_api.providers import DeterministicEmbeddingProvider
from evidencedesk_api.queueing import TaskQueue
from evidencedesk_api.seed import seed_demo_data
from evidencedesk_api.storage import LocalDocumentStorage
from evidencedesk_worker.jobs import WorkerContext, process_document
from sqlalchemy import text
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
def api(tmp_path: Path) -> AsyncIterator[tuple[TestClient, RecordingQueue, Settings]]:
    settings = Settings(
        database_url=DATABASE_URL,
        redis_url="redis://127.0.0.1:56379/0",
        storage_root=tmp_path,
        jwt_secret="integration-secret-at-least-32-characters",
        public_demo_mode=True,
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
def test_auth_upload_idempotence_audit_and_deletion(
    api: tuple[TestClient, RecordingQueue, Settings],
) -> None:
    client, queue, _settings = api
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
    assert status_response.json()["mode"] == "extractive-local"
    assert status_response.json()["documents"]["queued"] == 1

    deleted = client.delete(f"/api/v1/documents/{first.json()['id']}", headers=_headers(admin))
    assert deleted.status_code == 204


@pytest.mark.integration
def test_public_demo_rejects_unattested_upload(
    api: tuple[TestClient, RecordingQueue, Settings],
) -> None:
    client, _, _settings = api
    analyst = _token(client, "demo.analyst", "EvidenceDemo-Analyst-2026!")
    dossier_id = client.get("/api/v1/dossiers", headers=_headers(analyst)).json()[0]["id"]

    response = client.post(
        f"/api/v1/dossiers/{dossier_id}/documents",
        headers=_headers(analyst),
        files={"file": ("real.txt", b"not allowed in public mode", "text/plain")},
        data={"is_synthetic": "false"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "synthetic_attestation_required"


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

    answer = client.post(
        f"/api/v1/dossiers/{dossier_id}/ask",
        headers=_headers(reader),
        json={"question": "How quickly must the supplier notify incidents?"},
    )
    assert answer.status_code == 200, answer.text
    payload = answer.json()
    assert payload["status"] == "answered"
    assert payload["mode"] == "extractive-local"
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

    executed = client.post(
        "/api/v1/evaluations/run", headers=_headers(admin), json={"split": "development"}
    )
    assert executed.status_code == 201, executed.text
    result = executed.json()
    assert result["split"] == "development"
    assert result["metrics"]["case_count"] == 21
    assert result["metrics"]["estimated_cost_usd"] == 0.0
    assert result["metrics"]["citation_precision"] >= 0

    listed = client.get("/api/v1/evaluations", headers=_headers(reader))
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == result["id"]
