"""Serialization strategies for PyFolds artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast
import io
import json

import torch
import zstandard as zstd
from reedsolo import RSCodec  # type: ignore[import-untyped]


class SerializationStrategy(ABC):
    """Abstract serialization strategy.

    Parameters
    ----------
    path : str | Path
        Destination or source path.
    payload : dict[str, Any]
        Serializable state payload.

    Returns
    -------
    dict[str, Any]
        Loaded payload.

    Examples
    --------
    >>> strategy = JSONStrategy()
    >>> data = {"theta": 0.5}
    >>> _ = strategy.save("tmp.json", data)
    """

    @abstractmethod
    def save(self, path: str | Path, payload: dict[str, Any]) -> Path:
        """Save payload to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> dict[str, Any]:
        """Load payload from disk."""


class ZstdFoldStrategy(SerializationStrategy):
    """Serialize payload using torch bytes + Zstd + Reed-Solomon parity."""

    def __init__(self, zstd_level: int = 5, ecc_symbols: int = 10):
        self._compressor = zstd.ZstdCompressor(level=zstd_level)
        self._decompressor = zstd.ZstdDecompressor()
        self._ecc = RSCodec(ecc_symbols)

    def save(self, path: str | Path, payload: dict[str, Any]) -> Path:
        target = Path(path)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        compressed = self._compressor.compress(buffer.getvalue())
        protected = self._ecc.encode(compressed)
        target.write_bytes(protected)
        return target

    def load(self, path: str | Path) -> dict[str, Any]:
        protected = Path(path).read_bytes()
        decoded = self._ecc.decode(protected)[0]
        decompressed = self._decompressor.decompress(decoded)
        loaded = torch.load(
            io.BytesIO(decompressed),
            map_location="cpu",
            weights_only=False,  # payload contém metadados não-tensor
        )
        if not isinstance(loaded, dict):
            raise TypeError("Decoded payload must be a dictionary")
        return cast(dict[str, Any], loaded)


class JSONStrategy(SerializationStrategy):
    """Serialize metadata payloads as JSON."""

    def save(self, path: str | Path, payload: dict[str, Any]) -> Path:
        target = Path(path)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def load(self, path: str | Path) -> dict[str, Any]:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError("JSON payload must be a dictionary")
        return cast(dict[str, Any], loaded)


class TorchCheckpointStrategy(SerializationStrategy):
    """Serialize payloads using ``torch.save`` checkpoints."""

    def save(self, path: str | Path, payload: dict[str, Any]) -> Path:
        target = Path(path)
        torch.save(payload, target)
        return target

    def load(self, path: str | Path) -> dict[str, Any]:
        loaded = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,  # payload contém metadados não-tensor
        )
        if not isinstance(loaded, dict):
            raise TypeError("Torch checkpoint payload must be a dictionary")
        return cast(dict[str, Any], loaded)
