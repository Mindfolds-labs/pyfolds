from pyfolds.serialization.foldio import FoldWriter, FoldReader

with FoldWriter("basic.fold", compress="none", security_level="basic") as w:
    w.add_chunk("msg", "JSON", b'{"hello":"world"}')
    w.finalize({"example": "basic"})

with FoldReader("basic.fold") as r:
    print(r.read_chunk_bytes("msg"))
