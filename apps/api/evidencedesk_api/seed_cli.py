"""Idempotently seed the local, synthetic-only EvidenceDesk demonstration."""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from evidencedesk_worker.jobs import WorkerContext, process_document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evidencedesk_api.config import Settings, get_settings
from evidencedesk_api.db import build_engine, build_session_factory
from evidencedesk_api.models import (
    AuditEvent,
    Document,
    DocumentStatus,
    EvaluationRun,
    ProcessingTask,
    TaskStatus,
)
from evidencedesk_api.provider_registry import build_provider_bundle
from evidencedesk_api.providers import EmbeddingProvider
from evidencedesk_api.seed import DEMO_DOSSIER_ID, seed_demo_data
from evidencedesk_api.storage import LocalDocumentStorage, build_storage_key
from evidencedesk_api.uploads import load_public_demo_hashes

_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "markdown": "text/markdown",
    "txt": "text/plain",
}

_EVALUATION_METADATA = {
    "schema_version",
    "created_at",
    "dataset_version",
    "parameters_version",
    "seed",
    "split",
    "mode",
    "verdict",
    "cases",
    "extractions",
    "extraction_evaluation",
}


def _source_path(document: dict[str, object]) -> Path:
    filename = str(document["filename"])
    if filename.endswith(".pdf"):
        return Path("datasets/generated") / filename
    return Path(str(document["source_path"]))


def _assert_seed_digest_allowed(
    settings: Settings,
    digest: str,
    public_demo_hashes: frozenset[str] | None = None,
) -> None:
    if not settings.public_demo_mode:
        return
    allowed = public_demo_hashes or load_public_demo_hashes(settings.public_demo_allowlist)
    if digest not in allowed:
        raise RuntimeError("demo corpus document is not approved")


def _artifact_created_at(path: Path, result: dict[str, object]) -> datetime:
    value = result.get("created_at")
    if not isinstance(value, str):
        raise RuntimeError(f"evaluation artifact has no timestamp: {path.name}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"evaluation artifact timestamp is invalid: {path.name}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"evaluation artifact timestamp has no timezone: {path.name}")
    return parsed


def _build_seed_embeddings(settings: Settings) -> EmbeddingProvider:
    return build_provider_bundle(
        settings.answer_mode,
        model_path=settings.embedding_model_path,
        manifest_path=settings.embedding_manifest_path,
    ).embedding


async def _add_document(
    settings: Settings,
    manifest_document: dict[str, object],
    worker: WorkerContext,
    public_demo_hashes: frozenset[str],
) -> None:
    source = _source_path(manifest_document)
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    _assert_seed_digest_allowed(settings, digest, public_demo_hashes)
    factory = worker.session_factory

    async with factory() as session:
        existing = await session.scalar(
            select(Document).where(
                Document.dossier_id == DEMO_DOSSIER_ID,
                Document.content_sha256 == digest,
                Document.deleted_at.is_(None),
            )
        )
        if existing is not None:
            return
        document_id = uuid4()
        task_id = uuid4()
        correlation_id = uuid4()
        filename = str(manifest_document["filename"])
        document_format = str(manifest_document["format"])
        storage_key = build_storage_key(document_id, filename)
        worker.storage.put(storage_key, data)
        document = Document(
            id=document_id,
            dossier_id=DEMO_DOSSIER_ID,
            filename=filename,
            media_type=_MEDIA_TYPES[document_format],
            size_bytes=len(data),
            content_sha256=digest,
            storage_key=storage_key,
            status=DocumentStatus.QUEUED,
            is_synthetic=True,
            correlation_id=correlation_id,
        )
        task = ProcessingTask(
            id=task_id,
            document_id=document_id,
            status=TaskStatus.QUEUED,
            attempts=0,
            correlation_id=correlation_id,
        )
        session.add_all([document, task])
        session.add(
            AuditEvent(
                actor_id=None,
                action="demo.seed",
                document_id=document_id,
                result="queued",
                correlation_id=correlation_id,
                metadata_safe={"dataset_document_id": str(manifest_document["id"])},
            )
        )
        await session.commit()

    result = await process_document(
        {"worker": worker}, str(document_id), str(task_id), str(correlation_id)
    )
    if result.get("status") != "completed":
        raise RuntimeError(f"demo seed processing failed: {result}")


async def _seed_evaluation_artifacts(
    factory: async_sessionmaker[AsyncSession],
    artifacts_dir: Path = Path("artifacts/evaluations"),
) -> None:
    async with factory() as session:
        for path, result in _load_evaluation_artifacts(artifacts_dir):
            required = {
                "schema_version",
                "dataset_version",
                "parameters_version",
                "split",
                "mode",
                "verdict",
            }
            if not isinstance(result, dict) or required - result.keys():
                raise RuntimeError(f"invalid evaluation artifact: {path.name}")
            existing = await session.scalar(
                select(EvaluationRun).where(
                    EvaluationRun.dataset_version == str(result["dataset_version"]),
                    EvaluationRun.parameters_version == str(result["parameters_version"]),
                    EvaluationRun.split == str(result["split"]),
                )
            )
            if existing is not None:
                continue
            session.add(
                EvaluationRun(
                    dataset_version=str(result["dataset_version"]),
                    parameters_version=str(result["parameters_version"]),
                    split=str(result["split"]),
                    mode=str(result["mode"]),
                    metrics={
                        key: value
                        for key, value in result.items()
                        if key not in _EVALUATION_METADATA
                    },
                    verdict=str(result["verdict"]),
                    created_at=_artifact_created_at(path, result),
                )
            )
        await session.commit()


def _load_evaluation_artifacts(artifacts_dir: Path) -> list[tuple[Path, dict[str, object]]]:
    if not artifacts_dir.exists():
        return []
    loaded: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(artifacts_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError(f"invalid evaluation artifact: {path.name}")
        if result.get("schema_version") not in {
            "evaluation-result-v1",
            "holdout-v4-raw-result-v2",
        }:
            continue
        loaded.append((path, result))
    return loaded


async def seed(settings: Settings) -> None:
    engine = build_engine(settings)
    try:
        await seed_demo_data(settings, engine=engine)
        factory = build_session_factory(engine)
        worker = WorkerContext(
            settings=settings,
            engine=engine,
            session_factory=factory,
            storage=LocalDocumentStorage(settings.storage_root),
            embeddings=_build_seed_embeddings(settings),
        )
        public_demo_hashes = (
            load_public_demo_hashes(settings.public_demo_allowlist)
            if settings.public_demo_mode
            else frozenset()
        )
        manifest = json.loads(settings.corpus_manifest.read_text(encoding="utf-8"))
        if manifest.get("synthetic_only") is not True:
            raise RuntimeError("demo corpus must be synthetic-only")
        documents = manifest.get("documents")
        if not isinstance(documents, list):
            raise RuntimeError("demo corpus manifest has no documents list")
        for document in documents:
            if not isinstance(document, dict):
                raise RuntimeError("demo corpus document is invalid")
            await _add_document(settings, document, worker, public_demo_hashes)
        await _seed_evaluation_artifacts(factory)
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(seed(get_settings()))


if __name__ == "__main__":
    main()
