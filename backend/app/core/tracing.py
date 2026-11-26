"""OpenTelemetry tracing.

Off unless OTEL_EXPORTER_OTLP_ENDPOINT is set. The W3C trace context is
injected into every queue message and extracted by the worker, so a job's
spans (job.run -> stage.*) hang under the HTTP request that created it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

from opentelemetry import context, propagate, trace

_configured = False


def setup_tracing(service_name: str, endpoint: Optional[str]) -> bool:
    global _configured
    if not endpoint or _configured:
        return _configured
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")))
    trace.set_tracer_provider(provider)
    _configured = True
    return True


def instrument_app(app) -> None:
    if not _configured:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,healthz")
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument()
    except Exception:  # pragma: no cover - optional instrumentation
        pass


def instrument_engine(engine) -> None:
    """Trace SQL statements (called once the engine exists, in the app lifespan)."""
    if not _configured:
        return
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    except Exception:  # pragma: no cover - optional instrumentation
        pass


tracer = trace.get_tracer("image-to-3d")


def inject_context() -> Optional[str]:
    """The current trace context as a W3C traceparent string (None if no span)."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier.get("traceparent")


@contextmanager
def continued_span(name: str, traceparent: Optional[str], **attributes) -> Iterator[trace.Span]:
    """Start a span that continues the trace in `traceparent`."""
    ctx = propagate.extract({"traceparent": traceparent}) if traceparent else None
    token = context.attach(ctx) if ctx is not None else None
    try:
        with tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span
    finally:
        if token is not None:
            context.detach(token)


@contextmanager
def span(name: str, **attributes) -> Iterator[trace.Span]:
    """A child span of whatever is current."""
    with tracer.start_as_current_span(name, attributes=attributes) as s:
        yield s


class StageSpans:
    """Spans for pipeline stages, opened and closed from the pipeline thread.

    The pipeline runs in a worker thread and only reports "stage changed", so
    each stage span is started explicitly with job.run's context as parent
    (captured here, on the event loop) instead of relying on the thread's
    current context.
    """

    def __init__(self) -> None:
        self._parent = context.get_current()
        self._current: Optional[trace.Span] = None

    def enter(self, stage: str) -> None:
        self.close()
        self._current = tracer.start_span(f"stage.{stage}", context=self._parent)

    def close(self) -> None:
        if self._current is not None:
            self._current.end()
            self._current = None
