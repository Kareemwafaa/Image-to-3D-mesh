"""Structured logging with structlog.

Every line carries the context bound for the current request or job
(request_id, job_id, attempt, worker_id) and the active trace id, so logs can
be joined with traces. JSON in production, coloured key=value in development.
"""

from __future__ import annotations

import logging
import sys

import structlog


def _add_trace_ids(_, __, event_dict):
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except Exception:  # pragma: no cover - tracing is optional
        pass
    return event_dict


def setup_logging(level: str = "INFO", json: bool = True) -> None:
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_trace_ids,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(foreign_pre_chain=shared, processor=renderer))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "botocore", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = "app"):
    return structlog.get_logger(name)
