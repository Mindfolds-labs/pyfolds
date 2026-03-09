from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from .foldio import FoldReader, FoldWriter


class AsyncFoldReader:
    def __init__(self, path: str, use_mmap: bool = True):
        self._reader = FoldReader(path, use_mmap=use_mmap)

    async def __aenter__(self) -> "AsyncFoldReader":
        await asyncio.to_thread(self._reader.__enter__)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await asyncio.to_thread(self._reader.__exit__, exc_type, exc, tb)

    async def read_header(self) -> Dict[str, Any]:
        return await asyncio.to_thread(lambda: dict(self._reader.header))

    async def read_chunk_bytes(self, name: str, verify: bool = True) -> bytes:
        return await asyncio.to_thread(self._reader.read_chunk_bytes, name, verify)


class AsyncFoldWriter:
    def __init__(self, path: str, **kwargs: Any):
        self._writer = FoldWriter(path, **kwargs)

    async def __aenter__(self) -> "AsyncFoldWriter":
        await asyncio.to_thread(self._writer.__enter__)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await asyncio.to_thread(self._writer.__exit__, exc_type, exc, tb)

    async def add_chunk(self, name: str, ctype4: str, payload: bytes) -> None:
        await asyncio.to_thread(self._writer.add_chunk, name, ctype4, payload)

    async def finalize(self, metadata: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._writer.finalize, metadata)
