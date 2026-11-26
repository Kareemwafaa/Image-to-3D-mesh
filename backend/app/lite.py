"""Lite mode: the whole system in one process, for one free notebook GPU.

    cd backend && python -m app.lite

Same gateway, worker and pipeline code as the full stack, with the
infrastructure swapped for in-process versions: SQLite instead of Postgres,
a local directory instead of S3, an asyncio queue instead of Redis Streams.
HOST and PORT (default 0.0.0.0:8000) and every other setting still come
from the environment.
"""

from __future__ import annotations

import os


def main() -> None:
    from .core.config import lite_defaults

    for key, value in lite_defaults().items():
        os.environ.setdefault(key, value)
    # Before torch is imported: grow memory segments instead of fragmenting a small GPU.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import uvicorn

    from .core.config import get_settings
    from .main import create_app

    s = get_settings()
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_config=None, proxy_headers=True, timeout_keep_alive=30)


if __name__ == "__main__":
    main()
