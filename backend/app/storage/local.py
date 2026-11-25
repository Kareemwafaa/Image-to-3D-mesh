from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from .base import CHUNK, ObjectNotFound


class LocalStorage:
    """Files under a root directory. Writes are atomic (temp file + rename)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError(f"key escapes storage root: {key!r}")
        return path

    def _write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    async def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(self._write, key, data)

    async def put_file(self, key: str, path: Path, content_type: str) -> None:
        await asyncio.to_thread(lambda: self._write(key, Path(path).read_bytes()))

    async def download(self, key: str, dest: Path) -> None:
        src = self._path(key)
        if not src.is_file():
            raise ObjectNotFound(key)
        await asyncio.to_thread(shutil.copyfile, src, dest)

    async def size(self, key: str) -> Optional[int]:
        path = self._path(key)
        return path.stat().st_size if path.is_file() else None

    def iter_chunks(self, key: str) -> Iterator[bytes]:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFound(key)
        with path.open("rb") as f:
            while chunk := f.read(CHUNK):
                yield chunk

    async def delete_prefix(self, prefix: str) -> None:
        target = self._path(prefix.rstrip("/"))
        await asyncio.to_thread(shutil.rmtree, target, True)

    async def ping(self) -> None:
        if not self.root.is_dir():
            raise OSError(f"storage root missing: {self.root}")
