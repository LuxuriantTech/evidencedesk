from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from evidencedesk_api.auth import Role
from evidencedesk_api.models import DocumentStatus


class UserView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    role: Role


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_in: int
    user: UserView


class DossierView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str
    is_synthetic: bool


class DocumentView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dossier_id: UUID
    filename: str
    media_type: str
    status: DocumentStatus
    task_id: UUID | None = None
    correlation_id: UUID
    error_code: str | None
    created_at: datetime
    completed_at: datetime | None


class AuditEventView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    actor_id: UUID | None
    action: str
    document_id: UUID | None
    result: str
    correlation_id: UUID
    metadata_safe: dict[str, Any]
    created_at: datetime


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)


class CitationView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    document_id: str
    document_name: str
    page: int
    section: str | None
    excerpt: str


class PassageCandidateAssessmentView(BaseModel):
    answerable: bool
    answer: str | None
    confidence: float
    supporting_document: str | None
    supporting_page: int | None
    supporting_excerpt: str | None
    ambiguity_reason: str | None
    extracted_fields: dict[str, Any]
    supporting_chunk_id: str


class AskResponse(BaseModel):
    status: str
    answer: str
    confidence: float
    mode: str
    citations: list[CitationView]
    correlation_id: UUID
    answerable: bool
    supporting_document: str | None
    supporting_page: int | None
    supporting_excerpt: str | None
    ambiguity_reason: str | None
    extracted_fields: dict[str, Any]
    candidate_assessments: list[PassageCandidateAssessmentView]


class ContentChunkView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    page: int
    section: str | None
    ordinal: int
    text: str


class DocumentExtractionView(BaseModel):
    document_id: UUID
    document_name: str
    schema_version: str
    payload: dict[str, Any]


class EvaluationRequest(BaseModel):
    split: Literal["development"]


class EvaluationRunView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dataset_version: str
    parameters_version: str
    split: str
    mode: str
    metrics: dict[str, Any]
    verdict: str
    created_at: datetime
