import asyncio

from pyfolds.serialization.async_io import AsyncFoldReader, AsyncFoldWriter


def test_async_reader_writer(tmp_path):
    p = tmp_path / "a.fold"

    async def run():
        async with AsyncFoldWriter(str(p), compress="none") as w:
            await w.add_chunk("x", "JSON", b"{}")
            await w.finalize({"m": 1})
        async with AsyncFoldReader(str(p), use_mmap=False) as r:
            h = await r.read_header()
            b = await r.read_chunk_bytes("x")
            assert h["index_len"] > 0
            assert b == b"{}"

    asyncio.run(run())
