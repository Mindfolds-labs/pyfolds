import asyncio
from pyfolds.serialization.async_io import AsyncFoldWriter

async def main():
    async with AsyncFoldWriter("async.fold", compress="none") as w:
        await w.add_chunk("x", "JSON", b"{}")
        await w.finalize({"async": True})

asyncio.run(main())
