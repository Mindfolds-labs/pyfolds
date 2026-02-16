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
    """Codec Reed-Solomon opcional para correção de corrupção por chunk."""

    def __init__(self, symbols: int = 32):
        if symbols <= 0:
            raise ValueError("symbols deve ser > 0")
        import reedsolo

        self._reedsolo = reedsolo
        self._codec = reedsolo.RSCodec(symbols)
        self.symbols = symbols
        self.name = f"rs({symbols})"

    def encode(self, data: bytes) -> ECCResult:
        encoded = self._codec.encode(data)
        ecc_len = len(encoded) - len(data)
        ecc = encoded[-ecc_len:] if ecc_len > 0 else b""
        return ECCResult(ecc_algo=self.name, ecc_bytes=ecc)

    def decode(self, data: bytes, ecc_bytes: bytes) -> bytes:
        if not ecc_bytes:
            return data
        try:
            decoded = self._codec.decode(data + ecc_bytes)
        except self._reedsolo.ReedSolomonError as exc:
            raise RuntimeError(f"ECC decode falhou ({self.name}): {exc}") from exc

        if isinstance(decoded, tuple):
            decoded = decoded[0]
        return decoded


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
