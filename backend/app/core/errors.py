"""Error taxonomy.

HTTP side: every error leaves as RFC 9457 problem+json with a stable `code`.
Worker side: failures are classified as permanent (retrying cannot help) or
transient (retry with backoff), which drives the retry / dead-letter logic.
"""

from __future__ import annotations

from typing import Optional


class AppError(Exception):
    """An error with an HTTP status and a stable machine-readable code."""

    def __init__(self, status: int, code: str, detail: str, headers: Optional[dict] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.headers = headers or {}


# --- job failures ------------------------------------------------------------


class JobError(Exception):
    code = "internal"
    retryable = True

    def __init__(self, message: str = "", code: Optional[str] = None) -> None:
        super().__init__(message or self.__class__.__name__)
        if code:
            self.code = code


class PermanentJobError(JobError):
    """Bad input or a result that will not change on retry."""

    code = "invalid_input"
    retryable = False


class TransientJobError(JobError):
    """Infrastructure hiccup: storage, database, GPU memory."""

    code = "transient"
    retryable = True


class JobTimeout(PermanentJobError):
    code = "timeout"


class JobCancelled(PermanentJobError):
    code = "cancelled"


def classify(exc: BaseException) -> JobError:
    """Map any exception raised while running a job to a JobError."""
    if isinstance(exc, JobError):
        return exc
    from PIL import UnidentifiedImageError

    from ..pipeline.errors import MeshingError, PreprocessError

    if isinstance(exc, PreprocessError):
        return PermanentJobError(str(exc), code="no_foreground")
    if isinstance(exc, MeshingError):
        return PermanentJobError(str(exc), code="empty_mesh")
    if isinstance(exc, UnidentifiedImageError):
        return PermanentJobError(str(exc), code="bad_image")
    name = type(exc).__name__
    if name == "OutOfMemoryError":  # torch.cuda.OutOfMemoryError, without importing torch
        return TransientJobError(str(exc), code="gpu_oom")
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return TransientJobError(f"{name}: {exc}", code="transient")
    return JobError(f"{name}: {exc}", code="internal")
