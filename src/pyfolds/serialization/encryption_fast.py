from __future__ import annotations

import os
from typing import Iterable, List, Tuple


class FastEncryptor:
    def __init__(self, key: bytes):
        self.key = key
        self._aesgcm = None

    def _lazy(self):
        if self._aesgcm is None:
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            except Exception as exc:
                raise RuntimeError(
                    "Criptografia AES-GCM requer pacote 'cryptography'. Instale com: pip install cryptography"
                ) from exc
            if len(self.key) != 32:
                raise ValueError("AES-256-GCM requer chave de 32 bytes")
            self._aesgcm = AESGCM(self.key)
        return self._aesgcm

    def encrypt(self, data: bytes, aad: bytes = b"") -> Tuple[bytes, bytes]:
        aes = self._lazy()
        nonce = os.urandom(12)
        return nonce, aes.encrypt(nonce, data, aad)

    def decrypt(self, nonce: bytes, ciphertext: bytes, aad: bytes = b"") -> bytes:
        aes = self._lazy()
        return aes.decrypt(nonce, ciphertext, aad)

    def encrypt_chunked(self, chunks: Iterable[bytes], aad_prefix: bytes = b"chunk") -> List[Tuple[bytes, bytes]]:
        out: List[Tuple[bytes, bytes]] = []
        for i, chunk in enumerate(chunks):
            out.append(self.encrypt(chunk, aad=aad_prefix + b":" + str(i).encode("ascii")))
        return out
