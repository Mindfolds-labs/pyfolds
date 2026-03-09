from pyfolds.serialization.foldio import FoldWriter, FoldReader

key = b"k" * 32
with FoldWriter("high.fold", compress="none", security_level="high", encrypt=True, encryption_key=key) as w:
    w.add_chunk("msg", "JSON", b'{"secret":true}')
    w.finalize({"example": "high"})

with FoldReader("high.fold", decryption_key=key) as r:
    print(r.read_chunk_bytes("msg"))
