# syntax=docker/dockerfile:1@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
FROM python:3.12-slim@sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /uvx /bin/
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim@sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17 AS runtime

RUN apt-get update \
    && apt-get upgrade --yes \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/apps/api:/app/apps/worker:/app

COPY --from=builder /app/.venv /app/.venv

COPY scripts/download_local_model.py ./scripts/download_local_model.py
COPY infra/models ./infra/models
RUN --mount=type=bind,source=models,target=/local-models,ro \
    --mount=type=cache,id=evidencedesk-model-v1,target=/model-cache \
    mkdir -p /model-cache/paraphrase-multilingual-minilm-l12-v2 \
    && if [ -f /local-models/paraphrase-multilingual-minilm-l12-v2/model_optimized.onnx ]; then \
         cp -a /local-models/paraphrase-multilingual-minilm-l12-v2/. \
           /model-cache/paraphrase-multilingual-minilm-l12-v2/; \
       fi \
    && python scripts/download_local_model.py \
         --manifest infra/models/paraphrase-multilingual-minilm-l12-v2.json \
         --output /model-cache/paraphrase-multilingual-minilm-l12-v2 \
    && mkdir -p /opt/evidencedesk/models/paraphrase-multilingual-minilm-l12-v2 \
    && cp -a /model-cache/paraphrase-multilingual-minilm-l12-v2/. \
         /opt/evidencedesk/models/paraphrase-multilingual-minilm-l12-v2/

COPY README.md ./
COPY alembic.ini ./
COPY apps/api ./apps/api
COPY apps/worker ./apps/worker
COPY evals ./evals
COPY datasets ./datasets
COPY artifacts/evaluations ./artifacts/evaluations
COPY scripts ./scripts

ENV EMBEDDING_MODEL_PATH=/opt/evidencedesk/models/paraphrase-multilingual-minilm-l12-v2 \
    EMBEDDING_MANIFEST_PATH=/app/infra/models/paraphrase-multilingual-minilm-l12-v2.json

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/documents \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000
