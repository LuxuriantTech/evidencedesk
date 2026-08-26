from evidencedesk_api.models import Base, Chunk, Document, DocumentStatus
from pgvector.sqlalchemy import Vector
from sqlalchemy import UniqueConstraint


def test_database_schema_contains_operational_entities() -> None:
    assert {
        "users",
        "dossiers",
        "documents",
        "processing_tasks",
        "chunks",
        "extractions",
        "audit_events",
        "evaluation_runs",
    } <= set(Base.metadata.tables)


def test_chunk_embedding_is_a_384_dimension_pgvector() -> None:
    vector_type = Chunk.__table__.c.embedding.type

    assert isinstance(vector_type, Vector)
    assert vector_type.dim == 384
    assert {index.name for index in Chunk.__table__.indexes} >= {
        "ix_chunks_embedding_hnsw",
        "ix_chunks_search_vector_gin",
    }


def test_document_content_is_unique_per_active_dossier() -> None:
    index = next(
        index for index in Document.__table__.indexes if index.name == "uq_active_document_hash"
    )

    assert index.unique is True
    assert [column.name for column in index.columns] == ["dossier_id", "content_sha256"]
    assert str(index.dialect_options["postgresql"]["where"]) == "deleted_at IS NULL"


def test_chunk_order_is_unique_inside_a_document() -> None:
    constraints = [
        constraint
        for constraint in Chunk.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    ]

    assert any(
        [column.name for column in constraint.columns] == ["document_id", "page", "ordinal"]
        for constraint in constraints
    )
    assert DocumentStatus.QUEUED.value == "queued"
