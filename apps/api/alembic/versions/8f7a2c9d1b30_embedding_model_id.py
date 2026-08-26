"""record the embedding vector space

Revision ID: 8f7a2c9d1b30
Revises: 4125c5ebef4a
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8f7a2c9d1b30"
down_revision: str | None = "4125c5ebef4a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "embedding_model_id",
            sa.String(length=160),
            nullable=False,
            server_default="deterministic-hash-v1:384",
        ),
    )
    op.alter_column("chunks", "embedding_model_id", server_default=None)
    op.create_index(
        "ix_chunks_embedding_model_id",
        "chunks",
        ["embedding_model_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_model_id", table_name="chunks")
    op.drop_column("chunks", "embedding_model_id")
