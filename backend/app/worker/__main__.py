"""GPU worker process: `python -m app.worker`.

Exposes Prometheus metrics on WORKER_METRICS_PORT and shuts down gracefully
on SIGTERM/SIGINT: it stops taking messages and finishes the current job (an
unfinished job's lease simply expires and another worker picks it up).
"""

from __future__ import annotations

import asyncio
import signal

from prometheus_client import start_http_server

from ..container import build_container
from ..core.config import get_settings
from ..core.logging import get_logger, setup_logging
from ..core.tracing import setup_tracing
from .pipeline import create_backend
from .runner import Worker


async def main() -> None:
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    setup_tracing(f"{s.service_name}-worker", s.otel_exporter_otlp_endpoint)
    log = get_logger("app.worker")
    if s.metrics_enabled:
        start_http_server(s.worker_metrics_port)
    c = build_container(s)
    await c.start()
    worker = Worker(c, create_backend(s))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.stop)
    log.info("worker_process_started", metrics_port=s.worker_metrics_port)
    try:
        await worker.run()
    finally:
        await c.close()


if __name__ == "__main__":
    asyncio.run(main())
