"""Pluggable ECC codecs for .fold/.mind chunk protection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ECCResult:
    """ECC encoding output."""

    algo: str
    ecc_bytes: bytes


class ECCCodec(Protocol):
    """Protocol for chunk-level error correction codecs."""

    name: str

    def encode(self, data: bytes) -> ECCResult:
        ...

    def decode(self, data: bytes, ecc_bytes: bytes) -> bytes:
        ...


class NoECC:
    """No-op codec used when ECC is disabled."""

    name = "none"

    def encode(self, data: bytes) -> ECCResult:
        return ECCResult(algo=self.name, ecc_bytes=b"")

    def decode(self, data: bytes, ecc_bytes: bytes) -> bytes:
        return data


class ReedSolomonECC:
    """Reed-Solomon chunk-level ECC.

    Notes:
        - Requires optional dependency ``reedsolo``.
        - ``symbols`` controls redundancy and correction capacity.
    """

    def __init__(self, symbols: int = 32):
        import reedsolo

        self._rs = reedsolo.RSCodec(symbols)
        self._symbols = symbols
        self.name = f"rs({symbols})"

    def encode(self, data: bytes) -> ECCResult:
        encoded = self._rs.encode(data)
        ecc_len = len(encoded) - len(data)
        ecc = encoded[-ecc_len:] if ecc_len > 0 else b""
        return ECCResult(algo=self.name, ecc_bytes=ecc)

    def decode(self, data: bytes, ecc_bytes: bytes) -> bytes:
        import reedsolo

        if not ecc_bytes:
            return data

        try:
            decoded = self._rs.decode(data + ecc_bytes)
        except reedsolo.ReedSolomonError as exc:
            raise RuntimeError(f"ECC decode failed ({self.name}): {exc}") from exc

        return decoded[0] if isinstance(decoded, tuple) else decoded


def ecc_from_protection(level: str) -> ECCCodec:
    """Maps protection levels to ECC codecs.

    Supported levels:
      - off / none / 0
      - low -> RS(16)
      - med -> RS(32)
      - high -> RS(64)
    """

    norm = (level or "off").lower()

    if norm in {"off", "none", "0"}:
        return NoECC()
    if norm == "low":
        return ReedSolomonECC(16)
    if norm == "med":
        return ReedSolomonECC(32)
    if norm == "high":
        return ReedSolomonECC(64)

    raise ValueError("protection must be one of: off|low|med|high")
