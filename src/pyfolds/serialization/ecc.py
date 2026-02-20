"""Codecs ECC plugáveis para o container .fold/.mind."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ECCResult:
    """Resultado da codificação ECC para um bloco de bytes."""

    ecc_algo: str
    ecc_bytes: bytes


class ECCCodec(Protocol):
    """Contrato de codec ECC utilizado por chunk."""

    name: str

    def encode(self, data: bytes) -> ECCResult:
        """Gera bytes de paridade para os dados informados."""

    def decode(self, data: bytes, ecc_bytes: bytes) -> bytes:
        """Tenta corrigir dados usando bytes de paridade."""


class NoECC:
    """Codec nulo (somente detecção via CRC/SHA)."""

    name = "none"

    def encode(self, data: bytes) -> ECCResult:
        return ECCResult(ecc_algo=self.name, ecc_bytes=b"")

    def decode(self, data: bytes, ecc_bytes: bytes) -> bytes:
        return data


class ReedSolomonECC:
    """Codec Reed-Solomon opcional para correção de corrupção por chunk.

    Os dados são segmentados em blocos com tamanho máximo de ``255 - symbols``
    (limite de GF(2^8)), evitando crescimento de memória quadrático ao processar
    chunks grandes.
    """

    def __init__(self, symbols: int = 32):
        if symbols <= 0:
            raise ValueError("symbols deve ser > 0")
        if symbols >= 255:
            raise ValueError("symbols deve ser < 255")

        import reedsolo

        self._reedsolo = reedsolo
        self._codec = reedsolo.RSCodec(symbols)
        self.symbols = symbols
        self.block_size = 255 - symbols
        self.name = f"rs({symbols})"

    def encode(self, data: bytes) -> ECCResult:
        if not data:
            return ECCResult(ecc_algo=self.name, ecc_bytes=b"")

        parity = bytearray()
        for offset in range(0, len(data), self.block_size):
            block = data[offset: offset + self.block_size]
            encoded = self._codec.encode(block)
            parity.extend(encoded[len(block):])

        return ECCResult(ecc_algo=self.name, ecc_bytes=bytes(parity))

    def decode(self, data: bytes, ecc_bytes: bytes) -> bytes:
        if not ecc_bytes:
            return data

        decoded = bytearray()
        ecc_offset = 0

        for offset in range(0, len(data), self.block_size):
            block = data[offset: offset + self.block_size]
            parity_len = len(block) and self.symbols or 0
            block_ecc = ecc_bytes[ecc_offset: ecc_offset + parity_len]
            ecc_offset += parity_len

            try:
                recovered = self._codec.decode(block + block_ecc)
            except self._reedsolo.ReedSolomonError as exc:
                raise RuntimeError(f"ECC decode falhou ({self.name}): {exc}") from exc

            if isinstance(recovered, tuple):
                recovered = recovered[0]
            decoded.extend(recovered)

        if ecc_offset != len(ecc_bytes):
            raise RuntimeError(
                f"ECC decode falhou ({self.name}): tamanho de paridade inconsistente"
            )

        return bytes(decoded)


class ECCProtector:
    """Proteção simples de um chunk único via Reed-Solomon.

    Esta API é útil para casos fora do container ``.fold/.mind`` em que o
    payload completo precisa de proteção ECC direta (ex.: sidecar de checkpoint).
    """

    def __init__(self, error_bytes: int = 10):
        if error_bytes <= 0:
            raise ValueError("error_bytes deve ser > 0")
        self.error_bytes = error_bytes

        import reedsolo

        self._reedsolo = reedsolo
        self._codec = reedsolo.RSCodec(error_bytes)

    def protect_chunk(self, data: bytes) -> bytes:
        """Adiciona bytes de paridade ao chunk."""
        return bytes(self._codec.encode(data))

    def recover_chunk(self, protected_data: bytes) -> bytes:
        """Tenta corrigir erros e retorna o payload original."""
        try:
            recovered = self._codec.decode(protected_data)
        except self._reedsolo.ReedSolomonError as exc:
            raise RuntimeError(f"Falha na recuperação ECC: {exc}") from exc

        if isinstance(recovered, tuple):
            recovered = recovered[0]
        return bytes(recovered)


def ecc_from_protection(level: str) -> ECCCodec:
    """Mapeia nível de proteção em codec ECC.

    Níveis:
        - off/none/0 -> NoECC
        - low -> ReedSolomonECC(16)
        - med -> ReedSolomonECC(32)
        - high -> ReedSolomonECC(64)
    """

    normalized = (level or "off").lower()
    if normalized in {"off", "none", "0"}:
        return NoECC()
    if normalized == "low":
        return ReedSolomonECC(16)
    if normalized == "med":
        return ReedSolomonECC(32)
    if normalized == "high":
        return ReedSolomonECC(64)
    raise ValueError("protection deve ser: off|low|med|high")
