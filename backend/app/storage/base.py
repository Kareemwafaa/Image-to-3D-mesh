"""Object storage interface. Keys look like `inputs/<sha256>.png` or `jobs/<id>/result.glb`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Protocol

CHUNK = 1 << 20


class ObjectNotFound(Exception):
    pass


class ObjectStorage(Protocol):
    async def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...
    async def put_file(self, key: str, path: Path, content_type: str) -> None: ...
    async def download(self, key: str, dest: Path) -> None: ...
    async def size(self, key: str) -> Optional[int]:
        """Size in bytes, or None if the object does not exist."""

    def iter_chunks(self, key: str) -> Iterator[bytes]:
        """Blocking iterator over the object's bytes (Starlette runs it in a thread)."""

    async def delete_prefix(self, prefix: str) -> None: ...
    async def ping(self) -> None: ...
