import logging
import sys

import structlog
from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.types import ASGIApp

HTTP_REQUESTS = Counter(
    "evidencedesk_http_requests_total",
    "HTTP requests handled by the API",
    ("method", "route", "status"),
)
HTTP_LATENCY = Histogram(
    "evidencedesk_http_request_duration_seconds",
    "API request duration",
    ("method", "route"),
)
SEARCH_LATENCY = Histogram(
    "evidencedesk_search_duration_seconds",
    "Document retrieval and answer duration",
    ("mode", "result"),
)
TASKS = Counter(
    "evidencedesk_document_tasks_total",
    "Document processing outcomes",
    ("result",),
)
MODEL_ERRORS = Counter(
    "evidencedesk_model_errors_total",
    "Model provider errors",
    ("provider",),
)


def configure_logging() -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def metrics_app() -> ASGIApp:
    return make_asgi_app()
