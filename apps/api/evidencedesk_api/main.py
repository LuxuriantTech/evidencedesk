from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

import jwt
import structlog
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import RequestResponseEndpoint

from evals.runner import evaluate_manifest
from evidencedesk_api.auth import Action, UserPrincipal, role_allows
from evidencedesk_api.config import Settings, get_settings
from evidencedesk_api.db import build_engine, build_session_factory, get_session
from evidencedesk_api.models import (
    AuditEvent,
    Chunk,
    Document,
    DocumentStatus,
    Dossier,
    EvaluationRun,
    Extraction,
    ProcessingTask,
    TaskStatus,
    User,
)
from evidencedesk_api.observability import (
    HTTP_LATENCY,
    HTTP_REQUESTS,
    SEARCH_LATENCY,
    configure_logging,
    metrics_app,
)
from evidencedesk_api.provider_registry import ProviderBundle, build_provider_bundle
from evidencedesk_api.queueing import TaskQueue, build_task_queue
from evidencedesk_api.request_limits import RequestBodyLimitMiddleware
from evidencedesk_api.retrieval import EvidenceChunk, RetrievalMethod, hybrid_rank
from evidencedesk_api.schemas import (
    AskRequest,
    AskResponse,
    AuditEventView,
    CitationView,
    ContentChunkView,
    DocumentExtractionView,
    DocumentView,
    DossierView,
    EvaluationRequest,
    EvaluationRunView,
    TokenResponse,
    UserView,
)
from evidencedesk_api.security import create_access_token, decode_access_token, verify_password
from evidencedesk_api.storage import LocalDocumentStorage, build_storage_key
from evidencedesk_api.uploads import UploadRejected, load_public_demo_hashes, validate_upload

bearer = HTTPBearer(auto_error=False)


def _correlation_id(request: Request) -> UUID:
    value = getattr(request.state, "correlation_id", None)
    return value if isinstance(value, UUID) else uuid4()


def _audit(
    session: AsyncSession,
    *,
    actor_id: UUID | None,
    action: str,
    result: str,
    correlation_id: UUID,
    document_id: UUID | None = None,
    metadata_safe: dict[str, object] | None = None,
) -> None:
    session.add(
        AuditEvent(
            actor_id=actor_id,
            action=action,
            document_id=document_id,
            result=result,
            correlation_id=correlation_id,
            metadata_safe=metadata_safe or {},
        )
    )


async def current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UserPrincipal:
    if credentials is None:
        raise HTTPException(status_code=401, detail={"code": "authentication_required"})
    try:
        principal = decode_access_token(
            credentials.credentials, secret=request.app.state.settings.jwt_secret
        )
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail={"code": "invalid_token"}) from None
    user = await session.get(User, principal.id)
    if user is None or not user.is_active or user.role != principal.role:
        raise HTTPException(status_code=401, detail={"code": "invalid_token"})
    return principal


def require_action(action: Action) -> Callable[..., object]:
    async def dependency(
        request: Request,
        principal: Annotated[UserPrincipal, Depends(current_user)],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> UserPrincipal:
        if role_allows(principal.role, action):
            return principal
        _audit(
            session,
            actor_id=principal.id,
            action="authorization.denied",
            result="denied",
            correlation_id=_correlation_id(request),
            metadata_safe={"requested_action": action.value},
        )
        await session.commit()
        raise HTTPException(status_code=403, detail={"code": "forbidden"})

    return dependency


def _document_view(document: Document, task_id: UUID | None = None) -> DocumentView:
    return DocumentView.model_validate(document).model_copy(update={"task_id": task_id})


def create_app(*, settings: Settings | None = None, task_queue: TaskQueue | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    if active_settings.max_request_bytes <= active_settings.max_upload_bytes:
        raise ValueError("max_request_bytes must exceed max_upload_bytes")
    public_demo_hashes = (
        load_public_demo_hashes(active_settings.public_demo_allowlist)
        if active_settings.public_demo_mode
        else frozenset()
    )
    configure_logging()
    logger = structlog.get_logger("evidencedesk.api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = build_engine(active_settings)
        app.state.engine = engine
        app.state.session_factory = build_session_factory(engine)
        app.state.settings = active_settings
        app.state.storage = LocalDocumentStorage(active_settings.storage_root)
        app.state.task_queue = task_queue or await build_task_queue(active_settings.redis_url)
        app.state.providers = build_provider_bundle(
            active_settings.answer_mode,
            model_path=active_settings.embedding_model_path,
            manifest_path=active_settings.embedding_manifest_path,
        )
        app.state.public_demo_hashes = public_demo_hashes
        try:
            yield
        finally:
            await app.state.task_queue.close()
            await engine.dispose()

    app = FastAPI(title="EvidenceDesk API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=active_settings.max_request_bytes,
    )
    app.mount("/metrics", metrics_app())

    @app.middleware("http")
    async def correlation_middleware(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        raw = request.headers.get("X-Correlation-ID")
        try:
            correlation_id = UUID(raw) if raw else uuid4()
        except ValueError:
            correlation_id = uuid4()
        request.state.correlation_id = correlation_id
        started = perf_counter()
        response = await call_next(request)
        duration = perf_counter() - started
        response.headers["X-Correlation-ID"] = str(correlation_id)
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        HTTP_REQUESTS.labels(request.method, route_path, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method, route_path).observe(duration)
        logger.info(
            "http_request",
            method=request.method,
            route=route_path,
            status=response.status_code,
            duration_ms=round(duration * 1_000, 3),
            correlation_id=str(correlation_id),
        )
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "evidencedesk-api"}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, str]:
        try:
            async with request.app.state.session_factory() as session:
                await session.execute(text("SELECT 1"))
            queue: TaskQueue = request.app.state.task_queue
            if not await queue.ping():
                raise RuntimeError("queue ping failed")
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail={"code": "dependencies_unavailable"}
            ) from exc
        return {"status": "ready"}

    @app.post("/api/v1/auth/token", response_model=TokenResponse)
    async def login(
        request: Request,
        form: Annotated[OAuth2PasswordRequestForm, Depends()],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> TokenResponse:
        user = await session.scalar(select(User).where(User.username == form.username))
        correlation_id = _correlation_id(request)
        if (
            user is None
            or not user.is_active
            or not verify_password(form.password, user.password_hash)
        ):
            _audit(
                session,
                actor_id=user.id if user else None,
                action="auth.login",
                result="denied",
                correlation_id=correlation_id,
            )
            await session.commit()
            raise HTTPException(status_code=401, detail={"code": "invalid_credentials"})
        principal = UserPrincipal(id=user.id, username=user.username, role=user.role)
        ttl = timedelta(minutes=active_settings.access_token_minutes)
        token = create_access_token(principal, secret=active_settings.jwt_secret, ttl=ttl)
        _audit(
            session,
            actor_id=user.id,
            action="auth.login",
            result="success",
            correlation_id=correlation_id,
        )
        await session.commit()
        return TokenResponse(
            access_token=token,
            expires_in=int(ttl.total_seconds()),
            user=UserView.model_validate(user),
        )

    @app.get("/api/v1/auth/me", response_model=UserView)
    async def me(principal: Annotated[UserPrincipal, Depends(current_user)]) -> UserView:
        return UserView(id=principal.id, username=principal.username, role=principal.role)

    @app.get("/api/v1/dossiers", response_model=list[DossierView])
    async def list_dossiers(
        _principal: Annotated[UserPrincipal, Depends(require_action(Action.VIEW_DOCUMENT))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> list[DossierView]:
        dossiers = (await session.scalars(select(Dossier).order_by(Dossier.name))).all()
        return [DossierView.model_validate(item) for item in dossiers]

    @app.get("/api/v1/dossiers/{dossier_id}/documents", response_model=list[DocumentView])
    async def list_documents(
        dossier_id: UUID,
        _principal: Annotated[UserPrincipal, Depends(require_action(Action.VIEW_DOCUMENT))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> list[DocumentView]:
        if await session.get(Dossier, dossier_id) is None:
            raise HTTPException(status_code=404, detail={"code": "dossier_not_found"})
        rows = (
            await session.execute(
                select(Document, ProcessingTask.id)
                .outerjoin(ProcessingTask, ProcessingTask.document_id == Document.id)
                .where(Document.dossier_id == dossier_id, Document.deleted_at.is_(None))
                .order_by(Document.created_at.desc())
            )
        ).all()
        return [_document_view(document, task_id) for document, task_id in rows]

    @app.post("/api/v1/dossiers/{dossier_id}/documents", response_model=DocumentView)
    async def upload_document(
        dossier_id: UUID,
        request: Request,
        principal: Annotated[UserPrincipal, Depends(require_action(Action.UPLOAD_DOCUMENT))],
        session: Annotated[AsyncSession, Depends(get_session)],
        file: Annotated[UploadFile, File()],
        is_synthetic: Annotated[bool, Form()],
    ) -> DocumentView | JSONResponse:
        correlation_id = _correlation_id(request)
        if active_settings.public_demo_mode and not is_synthetic:
            raise HTTPException(status_code=422, detail={"code": "synthetic_attestation_required"})
        if await session.get(Dossier, dossier_id) is None:
            raise HTTPException(status_code=404, detail={"code": "dossier_not_found"})
        data = await file.read(active_settings.max_upload_bytes + 1)
        try:
            validated = validate_upload(
                filename=file.filename or "upload",
                declared_content_type=file.content_type or "application/octet-stream",
                data=data,
                max_bytes=active_settings.max_upload_bytes,
            )
        except UploadRejected as exc:
            raise HTTPException(status_code=422, detail={"code": exc.code}) from exc
        if (
            active_settings.public_demo_mode
            and validated.sha256 not in request.app.state.public_demo_hashes
        ):
            raise HTTPException(
                status_code=422,
                detail={"code": "public_demo_file_not_approved"},
            )

        existing = await session.scalar(
            select(Document).where(
                Document.dossier_id == dossier_id,
                Document.content_sha256 == validated.sha256,
                Document.deleted_at.is_(None),
            )
        )
        if existing is not None:
            task_id = await session.scalar(
                select(ProcessingTask.id).where(ProcessingTask.document_id == existing.id)
            )
            _audit(
                session,
                actor_id=principal.id,
                action="document.upload",
                result="deduplicated",
                correlation_id=correlation_id,
                document_id=existing.id,
            )
            await session.commit()
            payload = _document_view(existing, task_id).model_dump(mode="json")
            return JSONResponse(status_code=200, content=payload)

        document_id = uuid4()
        task_id = uuid4()
        storage_key = build_storage_key(document_id, validated.safe_filename)
        storage: LocalDocumentStorage = request.app.state.storage
        storage.put(storage_key, validated.data)
        document = Document(
            id=document_id,
            dossier_id=dossier_id,
            filename=validated.safe_filename,
            media_type=validated.media_type,
            size_bytes=len(validated.data),
            content_sha256=validated.sha256,
            storage_key=storage_key,
            status=DocumentStatus.QUEUED,
            is_synthetic=is_synthetic,
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
        _audit(
            session,
            actor_id=principal.id,
            action="document.upload",
            result="queued",
            correlation_id=correlation_id,
            document_id=document_id,
            metadata_safe={"media_type": validated.media_type, "size_bytes": len(data)},
        )
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            storage.delete(storage_key)
            existing = await session.scalar(
                select(Document).where(
                    Document.dossier_id == dossier_id,
                    Document.content_sha256 == validated.sha256,
                    Document.deleted_at.is_(None),
                )
            )
            if existing is None:
                raise
            existing_task_id = await session.scalar(
                select(ProcessingTask.id).where(ProcessingTask.document_id == existing.id)
            )
            return JSONResponse(
                status_code=200,
                content=_document_view(existing, existing_task_id).model_dump(mode="json"),
            )

        queue: TaskQueue = request.app.state.task_queue
        try:
            await queue.enqueue_document(document_id, task_id, correlation_id)
        except Exception as exc:
            document.status = DocumentStatus.FAILED
            document.error_code = "queue_unavailable"
            task.status = TaskStatus.FAILED
            task.error_code = "queue_unavailable"
            _audit(
                session,
                actor_id=principal.id,
                action="document.enqueue",
                result="failed",
                correlation_id=correlation_id,
                document_id=document_id,
            )
            await session.commit()
            raise HTTPException(status_code=503, detail={"code": "queue_unavailable"}) from exc
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=_document_view(document, task_id).model_dump(mode="json"),
        )

    @app.delete("/api/v1/documents/{document_id}", status_code=204)
    async def delete_document(
        document_id: UUID,
        request: Request,
        principal: Annotated[UserPrincipal, Depends(require_action(Action.DELETE_DOCUMENT))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> Response:
        task = await session.scalar(
            select(ProcessingTask)
            .where(ProcessingTask.document_id == document_id)
            .with_for_update()
        )
        document = await session.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        if document is None or document.deleted_at is not None:
            raise HTTPException(status_code=404, detail={"code": "document_not_found"})
        storage: LocalDocumentStorage = request.app.state.storage
        storage.delete(document.storage_key)
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await session.execute(delete(Extraction).where(Extraction.document_id == document_id))
        if task is not None and task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
            task.status = TaskStatus.FAILED
            task.error_code = "document_deleted"
            task.completed_at = datetime.now(UTC)
        document.status = DocumentStatus.DELETED
        document.deleted_at = datetime.now(UTC)
        _audit(
            session,
            actor_id=principal.id,
            action="document.delete",
            result="success",
            correlation_id=_correlation_id(request),
            document_id=document_id,
        )
        await session.commit()
        return Response(status_code=204)

    @app.get("/api/v1/documents/{document_id}", response_model=DocumentView)
    async def get_document(
        document_id: UUID,
        _principal: Annotated[UserPrincipal, Depends(require_action(Action.VIEW_DOCUMENT))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> DocumentView:
        document = await session.get(Document, document_id)
        if document is None or document.deleted_at is not None:
            raise HTTPException(status_code=404, detail={"code": "document_not_found"})
        task_id = await session.scalar(
            select(ProcessingTask.id).where(ProcessingTask.document_id == document_id)
        )
        return _document_view(document, task_id)

    @app.get("/api/v1/documents/{document_id}/content", response_model=list[ContentChunkView])
    async def get_document_content(
        document_id: UUID,
        request: Request,
        principal: Annotated[UserPrincipal, Depends(require_action(Action.VIEW_DOCUMENT))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> list[ContentChunkView]:
        document = await session.get(Document, document_id)
        if document is None or document.deleted_at is not None:
            raise HTTPException(status_code=404, detail={"code": "document_not_found"})
        chunks = (
            await session.scalars(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.page, Chunk.ordinal)
            )
        ).all()
        _audit(
            session,
            actor_id=principal.id,
            action="document.content.view",
            result="success",
            correlation_id=_correlation_id(request),
            document_id=document_id,
            metadata_safe={"chunk_count": len(chunks)},
        )
        await session.commit()
        return [ContentChunkView.model_validate(chunk) for chunk in chunks]

    @app.post("/api/v1/dossiers/{dossier_id}/ask", response_model=AskResponse)
    async def ask_dossier(
        dossier_id: UUID,
        body: AskRequest,
        request: Request,
        principal: Annotated[UserPrincipal, Depends(require_action(Action.ASK_QUESTION))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> AskResponse:
        if await session.get(Dossier, dossier_id) is None:
            raise HTTPException(status_code=404, detail={"code": "dossier_not_found"})
        started = perf_counter()
        providers: ProviderBundle = request.app.state.providers
        query_embedding = providers.embedding.embed(body.question)
        eligible = (
            Document.dossier_id == dossier_id,
            Document.status == DocumentStatus.COMPLETED,
            Document.deleted_at.is_(None),
        )
        dense_rows = (
            await session.execute(
                select(Chunk, Document)
                .join(Document, Document.id == Chunk.document_id)
                .where(
                    *eligible,
                    Chunk.embedding_model_id == providers.embedding.model_id,
                )
                .order_by(Chunk.embedding.cosine_distance(query_embedding))
                .limit(20)
            )
        ).all()
        query = func.plainto_tsquery("english", body.question)
        sparse_rows = (
            await session.execute(
                select(Chunk, Document)
                .join(Document, Document.id == Chunk.document_id)
                .where(
                    *eligible,
                )
                .order_by(func.ts_rank_cd(Chunk.search_vector, query).desc())
                .limit(20)
            )
        ).all()
        candidates: dict[UUID, EvidenceChunk] = {}
        for chunk, document in [*dense_rows, *sparse_rows]:
            candidates[chunk.id] = EvidenceChunk(
                id=str(chunk.id),
                document_id=str(document.id),
                document_name=document.filename,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
                embedding=list(chunk.embedding),
                embedding_model_id=chunk.embedding_model_id,
            )
        ranked = hybrid_rank(body.question, list(candidates.values()), provider=providers.embedding)
        result = providers.answer.answer(body.question, ranked)
        latency_ms = round((perf_counter() - started) * 1_000, 3)
        SEARCH_LATENCY.labels(providers.answer.mode, result.status).observe(latency_ms / 1_000)
        correlation_id = _correlation_id(request)
        _audit(
            session,
            actor_id=principal.id,
            action="dossier.ask",
            result=result.status,
            correlation_id=correlation_id,
            metadata_safe={
                "dossier_id": str(dossier_id),
                "latency_ms": latency_ms,
                "citation_count": len(result.citations),
                "mode": providers.answer.mode,
            },
        )
        await session.commit()
        return AskResponse(
            status=result.status,
            answer=result.answer,
            confidence=result.confidence,
            mode=providers.answer.mode,
            citations=[CitationView.model_validate(item) for item in result.citations],
            correlation_id=correlation_id,
        )

    @app.get(
        "/api/v1/dossiers/{dossier_id}/extraction",
        response_model=list[DocumentExtractionView],
    )
    async def get_dossier_extraction(
        dossier_id: UUID,
        request: Request,
        principal: Annotated[UserPrincipal, Depends(require_action(Action.VIEW_EXTRACTION))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> list[DocumentExtractionView]:
        rows = (
            await session.execute(
                select(Extraction, Document)
                .join(Document, Document.id == Extraction.document_id)
                .where(
                    Document.dossier_id == dossier_id,
                    Document.status == DocumentStatus.COMPLETED,
                    Document.deleted_at.is_(None),
                )
                .order_by(Document.filename)
            )
        ).all()
        for _extraction, document in rows:
            _audit(
                session,
                actor_id=principal.id,
                action="document.extraction.view",
                result="success",
                correlation_id=_correlation_id(request),
                document_id=document.id,
                metadata_safe={"dossier_id": str(dossier_id)},
            )
        if rows:
            await session.commit()
        return [
            DocumentExtractionView(
                document_id=document.id,
                document_name=document.filename,
                schema_version=extraction.schema_version,
                payload=extraction.payload,
            )
            for extraction, document in rows
        ]

    @app.get("/api/v1/audit-events", response_model=list[AuditEventView])
    async def list_audit_events(
        _principal: Annotated[UserPrincipal, Depends(require_action(Action.VIEW_AUDIT))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> list[AuditEventView]:
        events = (
            await session.scalars(
                select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(200)
            )
        ).all()
        return [AuditEventView.model_validate(event) for event in events]

    @app.get("/api/v1/evaluations", response_model=list[EvaluationRunView])
    async def list_evaluations(
        _principal: Annotated[UserPrincipal, Depends(require_action(Action.VIEW_DOCUMENT))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> list[EvaluationRunView]:
        runs = (
            await session.scalars(
                select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(50)
            )
        ).all()
        return [EvaluationRunView.model_validate(run) for run in runs]

    @app.post("/api/v1/evaluations/run", response_model=EvaluationRunView, status_code=201)
    async def run_evaluation(
        body: EvaluationRequest,
        request: Request,
        principal: Annotated[UserPrincipal, Depends(require_action(Action.RUN_EVALUATION))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> EvaluationRunView:
        result = await run_in_threadpool(
            evaluate_manifest,
            active_settings.evaluation_manifest,
            active_settings.evaluation_corpus_manifest,
            split=body.split,
            provider=request.app.state.providers.embedding,
            method=RetrievalMethod.HYBRID,
        )
        excluded = {
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
        run = EvaluationRun(
            dataset_version=str(result["dataset_version"]),
            parameters_version=str(result["parameters_version"]),
            split=body.split,
            mode=str(result["mode"]),
            metrics={key: value for key, value in result.items() if key not in excluded},
            verdict=str(result["verdict"]),
        )
        session.add(run)
        await session.flush()
        _audit(
            session,
            actor_id=principal.id,
            action="evaluation.run",
            result=run.verdict.casefold(),
            correlation_id=_correlation_id(request),
            metadata_safe={
                "evaluation_id": str(run.id),
                "dataset_version": run.dataset_version,
                "parameters_version": run.parameters_version,
                "split": run.split,
            },
        )
        await session.commit()
        await session.refresh(run)
        return EvaluationRunView.model_validate(run)

    @app.get("/api/v1/status")
    async def admin_status(
        _principal: Annotated[UserPrincipal, Depends(require_action(Action.VIEW_AUDIT))],
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> dict[str, object]:
        rows = (
            await session.execute(
                select(Document.status, func.count(Document.id))
                .where(Document.deleted_at.is_(None))
                .group_by(Document.status)
            )
        ).all()
        documents = {status.value: int(count) for status, count in rows}
        for document_status in DocumentStatus:
            documents.setdefault(document_status.value, 0)
        failed_tasks = await session.scalar(
            select(func.count(ProcessingTask.id)).where(ProcessingTask.status == TaskStatus.FAILED)
        )
        return {
            "mode": active_settings.answer_mode,
            "public_demo_mode": active_settings.public_demo_mode,
            "documents": documents,
            "failed_tasks": int(failed_tasks or 0),
        }

    return app
