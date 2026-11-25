"""Prometheus metrics. Defined once at import, shared by gateway and worker."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
JOBS_SUBMITTED = Counter("jobs_submitted_total", "Jobs accepted by the gateway", ["idempotent_replay"])
JOBS_FINISHED = Counter("jobs_finished_total", "Jobs that reached a final state", ["status", "error_code"])
JOB_RETRIES = Counter("job_retries_total", "Job attempts scheduled for retry", ["error_code"])
JOBS_DEAD_LETTERED = Counter("jobs_dead_lettered_total", "Jobs moved to the dead-letter queue")
JOBS_RECLAIMED = Counter("jobs_reclaimed_total", "Messages reclaimed from a worker whose lease expired")
JOB_DURATION = Histogram(
    "job_duration_seconds",
    "Wall time of one job attempt",
    ["status"],
    buckets=(1, 5, 15, 30, 60, 120, 180, 300, 600, 900, 1800),
)
STAGE_DURATION = Histogram(
    "job_stage_duration_seconds",
    "Wall time per pipeline stage",
    ["stage"],
    buckets=(0.5, 1, 2.5, 5, 10, 20, 40, 60, 120, 240, 480),
)
QUEUE_DEPTH = Gauge("queue_depth", "Messages by queue state", ["state"])
JOBS_IN_PROGRESS = Gauge("worker_jobs_in_progress", "Jobs running on this worker")
RATE_LIMITED = Counter("rate_limited_total", "Requests refused by the rate limiter", ["bucket"])
SSE_CONNECTIONS = Gauge("sse_connections", "Open server-sent-event streams")
GPU_MEMORY = Gauge("worker_gpu_memory_bytes", "GPU memory allocated after the last job", ["kind"])
