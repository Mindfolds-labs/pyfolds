"""Chunked .fold/.mind container with mmap-friendly index and optional ECC."""

from __future__ import annotations

import io
import json
import hashlib
import mmap
import os
import struct
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .ecc import NoECC, ecc_from_protection

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional dependency
    zstd = None

MAGIC = b"FOLDSv1\0"
HEADER_SIZE = 8192
CHUNK_HDR_FMT = ">4sIQQII"  # ctype, flags, raw_len, comp_len, crc32, ecc_len
CHUNK_HDR_SIZE = struct.calcsize(CHUNK_HDR_FMT)

FLAG_COMP_NONE = 0
FLAG_COMP_ZSTD = 1


try:
    import google_crc32c

    def crc32c_u32(data: bytes) -> int:
        return int.from_bytes(google_crc32c.value(data).to_bytes(4, "big"), "big")

except Exception:  # pragma: no cover - optional dependency
    import zlib

    def crc32c_u32(data: bytes) -> int:
        return zlib.crc32(data) & 0xFFFFFFFF


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cfg_to_dict(cfg: Any) -> Any:
    if cfg is None:
        return None
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    return cfg


def _compress(raw: bytes, enabled: bool) -> tuple[bytes, int]:
    if not enabled:
        return raw, FLAG_COMP_NONE

    if zstd is None:
        return raw, FLAG_COMP_NONE

    return zstd.ZstdCompressor(level=3).compress(raw), FLAG_COMP_ZSTD


def _decompress(comp: bytes, flags: int) -> bytes:
    if flags == FLAG_COMP_ZSTD:
        if zstd is None:
            raise RuntimeError("zstandard is required to read compressed chunks")
        return zstd.ZstdDecompressor().decompress(comp)
    return comp


def build_nuclear_npz(neuron: torch.nn.Module) -> bytes:
    """Builds scientific-state chunk for partial reads independent of torch_state."""

    payload: Dict[str, np.ndarray] = {
        "N": neuron.N.detach().cpu().numpy(),
        "I": neuron.I.detach().cpu().numpy(),
        "W": neuron.W.detach().cpu().numpy(),
        "protection": neuron.protection.detach().cpu().numpy(),
        "theta": np.array([float(neuron.theta.item())], dtype=np.float32),
        "r_hat": np.array([float(neuron.r_hat.item())], dtype=np.float32),
    }

    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    return buf.getvalue()


def serialize_manifest(
    neuron: torch.nn.Module,
    *,
    kind: str,
    protection: str,
    tags: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bytes:
    mode = getattr(getattr(neuron, "mode", None), "value", None)
    step = int(getattr(getattr(neuron, "step_id", None), "item", lambda: 0)())

    manifest = {
        "format": "folds",
        "version": "1.1.0",
        "kind": kind,
        "model_type": neuron.__class__.__name__,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "protection": protection,
        "tags": tags or {},
        "shape": {
            "n_dendrites": int(len(neuron.dendrites)),
            "n_synapses_per_dendrite": int(neuron.cfg.n_synapses_per_dendrite),
            "total_synapses": int(neuron.N.numel()),
        },
        "expression": {
            "theta": float(neuron.theta.item()),
            "r_hat": float(neuron.r_hat.item()),
            "N_mean": float(neuron.N.float().mean().item()),
            "I_mean": float(neuron.I.float().mean().item()),
            "mode": mode,
            "step_id": step,
        },
        "routing": {
            "resume_training": "torch_state",
            "scientific": "nuclear_arrays",
            "audit": ["manifest", "metrics", "history", "accumulator_state"],
        },
        "extra": extra or {},
    }
    return _json_bytes(manifest)


def serialize_metrics(neuron: torch.nn.Module) -> Optional[bytes]:
    if not hasattr(neuron, "get_metrics"):
        return None

    try:
        metrics = neuron.get_metrics()
    except Exception:
        return None

    return _json_bytes(metrics)


def serialize_accumulator_state(neuron: torch.nn.Module) -> Optional[bytes]:
    if not hasattr(neuron, "stats_acc"):
        return None

    acc = neuron.stats_acc
    state: Dict[str, Any] = {
        "track_extra": bool(getattr(acc, "track_extra", False)),
        "history_enabled": bool(getattr(acc, "_history_enabled", False)),
        "buffers": {},
    }

    for name in (
        "acc_x",
        "acc_gated",
        "acc_spikes",
        "acc_count",
        "initialized",
        "acc_v_dend",
        "acc_u",
        "acc_theta",
        "acc_r_hat",
        "acc_adaptation",
    ):
        if hasattr(acc, name):
            t = getattr(acc, name).detach().cpu()
            state["buffers"][name] = {
                "dtype": str(t.dtype),
                "shape": list(t.shape),
                "data": t.reshape(-1).tolist(),
            }

    return _json_bytes(state)


def serialize_history(neuron: torch.nn.Module) -> Optional[bytes]:
    if not hasattr(neuron, "stats_acc"):
        return None
    acc = neuron.stats_acc
    if not getattr(acc, "_history_enabled", False):
        return None
    if not hasattr(acc, "_history"):
        return None

    hist = {k: list(v) for k, v in acc._history.items()}
    return _json_bytes(hist)


class FoldWriter:
    """Writes .fold/.mind chunk container."""

    def __init__(self, path: str, *, protection: str = "off"):
        self.path = path
        self.tmp_path = f"{path}.tmp"
        self.ecc = ecc_from_protection(protection)
        self._chunks: List[Dict[str, Any]] = []

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._f = open(self.tmp_path, "wb")
        self._f.write(MAGIC)
        self._f.write(b"\0" * (HEADER_SIZE - len(MAGIC)))
        return self

    def __exit__(self, exc_type, exc, tb):
        if hasattr(self, "_f") and self._f:
            self._f.close()

    def add_chunk(self, name: str, ctype4: str, payload: bytes, *, compress: bool = True) -> None:
        if len(ctype4) != 4:
            raise ValueError("ctype4 must have exactly 4 chars")

        comp, flags = _compress(payload, enabled=compress)

        ecc_res = self.ecc.encode(comp)
        crc = crc32c_u32(comp)
        sha = _sha256_hex(comp)

        offset = self._f.tell()
        hdr = struct.pack(
            CHUNK_HDR_FMT,
            ctype4.encode("ascii"),
            flags,
            len(payload),
            len(comp),
            crc,
            len(ecc_res.ecc_bytes),
        )
        self._f.write(hdr)
        self._f.write(comp)
        if ecc_res.ecc_bytes:
            self._f.write(ecc_res.ecc_bytes)

        self._chunks.append(
            {
                "name": name,
                "ctype": ctype4,
                "flags": flags,
                "offset": offset,
                "header_len": CHUNK_HDR_SIZE,
                "comp_len": len(comp),
                "uncomp_len": len(payload),
                "crc32c": crc,
                "sha256": sha,
                "ecc_algo": ecc_res.algo,
                "ecc_len": len(ecc_res.ecc_bytes),
            }
        )

    def finalize(self, metadata: Dict[str, Any]) -> None:
        index = {
            "format": "folds",
            "version": "1.1.0",
            "created_at_unix": time.time(),
            "metadata": metadata,
            "chunks": self._chunks,
        }
        index_bytes = _json_bytes(index)
        self.add_chunk("__index__", "INDX", index_bytes, compress=False)
        idx = self._chunks[-1]

        header = {
            "format": "folds",
            "version": "1.1.0",
            "index_offset": idx["offset"],
            "index_len": idx["uncomp_len"],
            "metadata": metadata,
        }

        self._f.flush()
        self._f.close()

        with open(self.tmp_path, "rb") as f:
            file_sha = _sha256_hex(f.read())

        header["file_sha256"] = file_sha
        raw = _json_bytes(header)

        if len(raw) > (HEADER_SIZE - len(MAGIC)):
            raise RuntimeError("header exceeded fixed HEADER_SIZE")

        with open(self.tmp_path, "r+b") as f:
            f.seek(len(MAGIC))
            f.write(raw)
            f.write(b"\0" * ((HEADER_SIZE - len(MAGIC)) - len(raw)))

        os.replace(self.tmp_path, self.path)


class FoldReader:
    """Reads .fold/.mind chunk container with optional mmap."""

    def __init__(self, path: str, *, use_mmap: bool = True):
        self.path = path
        self.use_mmap = use_mmap
        self.header: Dict[str, Any] = {}
        self.index: Dict[str, Any] = {}

    def __enter__(self):
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ) if self.use_mmap else None
        self._read_header()
        self._read_index()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._mm is not None:
            self._mm.close()
        if hasattr(self, "_f") and self._f:
            self._f.close()

    def _read_at(self, off: int, size: int) -> bytes:
        if self._mm is not None:
            return self._mm[off : off + size]
        self._f.seek(off)
        return self._f.read(size)

    def _read_header(self) -> None:
        raw = self._read_at(0, HEADER_SIZE)
        if raw[: len(MAGIC)] != MAGIC:
            raise ValueError("invalid .fold/.mind magic")
        header_json = raw[len(MAGIC) :].rstrip(b"\0")
        self.header = json.loads(header_json.decode("utf-8"))

    def _read_index(self) -> None:
        idx_off = int(self.header["index_offset"])
        hdr = self._read_at(idx_off, CHUNK_HDR_SIZE)
        _ctype, flags, _uncomp, comp_len, _crc, _ecc_len = struct.unpack(CHUNK_HDR_FMT, hdr)
        comp = self._read_at(idx_off + CHUNK_HDR_SIZE, comp_len)
        raw = _decompress(comp, flags)
        self.index = json.loads(raw.decode("utf-8"))

    def list_chunks(self) -> List[str]:
        return [c["name"] for c in self.index["chunks"]]

    def _chunk_by_name(self, name: str) -> Dict[str, Any]:
        for chunk in self.index["chunks"]:
            if chunk["name"] == name:
                return chunk
        raise KeyError(f"chunk '{name}' does not exist")

    def read_chunk(self, name: str, *, verify: bool = True) -> bytes:
        chunk = self._chunk_by_name(name)
        off = int(chunk["offset"])

        hdr = self._read_at(off, CHUNK_HDR_SIZE)
        _ctype, flags, raw_len, comp_len, crc, ecc_len = struct.unpack(CHUNK_HDR_FMT, hdr)
        comp_off = off + CHUNK_HDR_SIZE
        comp = self._read_at(comp_off, comp_len)
        ecc_bytes = self._read_at(comp_off + comp_len, ecc_len) if ecc_len else b""

        ecc_algo = chunk.get("ecc_algo", "none")
        if ecc_algo != "none" and ecc_bytes:
            if ecc_algo.startswith("rs(") and ecc_algo.endswith(")"):
                symbols = int(ecc_algo[3:-1])
                if symbols == 16:
                    codec = ecc_from_protection("low")
                elif symbols == 32:
                    codec = ecc_from_protection("med")
                else:
                    codec = ecc_from_protection("high")
                comp = codec.decode(comp, ecc_bytes)
            else:
                comp = NoECC().decode(comp, ecc_bytes)

        if verify:
            if crc32c_u32(comp) != crc:
                raise RuntimeError(f"CRC mismatch in chunk '{name}'")
            if _sha256_hex(comp) != chunk["sha256"]:
                raise RuntimeError(f"SHA256 mismatch in chunk '{name}'")

        raw = _decompress(comp, flags)
        if len(raw) != raw_len:
            raise RuntimeError(f"invalid decompressed size in chunk '{name}'")
        return raw

    def read_json(self, name: str, *, verify: bool = True) -> Dict[str, Any]:
        return json.loads(self.read_chunk(name, verify=verify).decode("utf-8"))

    def read_torch(self, name: str = "torch_state", *, map_location: str = "cpu", verify: bool = True) -> Any:
        return torch.load(io.BytesIO(self.read_chunk(name, verify=verify)), map_location=map_location)


def save_fold_or_mind(
    neuron: torch.nn.Module,
    path: str,
    *,
    kind: str = "fold",
    protection: str = "off",
    tags: Optional[Dict[str, str]] = None,
    include_accumulator: bool = True,
    include_history: bool = True,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves MPJRD neuron to chunked .fold/.mind file."""

    mode = getattr(getattr(neuron, "mode", None), "value", None)
    step_id = int(getattr(getattr(neuron, "step_id", None), "item", lambda: 0)())

    torch_payload = {
        "state_dict": neuron.state_dict(),
        "config": _cfg_to_dict(getattr(neuron, "cfg", None)),
        "mode": mode,
        "step_id": step_id,
    }
    buf = io.BytesIO()
    torch.save(torch_payload, buf)

    meta = {
        "kind": kind,
        "model_type": neuron.__class__.__name__,
        "protection": protection,
        "tags": tags or {},
    }

    with FoldWriter(path, protection=protection) as w:
        w.add_chunk(
            "manifest",
            "JSN0",
            serialize_manifest(
                neuron,
                kind=kind,
                protection=protection,
                tags=tags,
                extra=extra_manifest,
            ),
            compress=False,
        )
        w.add_chunk("nuclear_arrays", "NPZ0", build_nuclear_npz(neuron), compress=True)
        w.add_chunk("torch_state", "TSAV", buf.getvalue(), compress=True)

        metrics = serialize_metrics(neuron)
        if metrics is not None:
            w.add_chunk("metrics", "JSN0", metrics, compress=False)

        if include_accumulator:
            acc_state = serialize_accumulator_state(neuron)
            if acc_state is not None:
                w.add_chunk("accumulator_state", "JSN0", acc_state, compress=True)

        if include_history:
            hist = serialize_history(neuron)
            if hist is not None:
                w.add_chunk("history", "JSN0", hist, compress=False)

        w.finalize(meta)


def load_fold_or_mind(path: str, neuron_class: Any, *, map_location: str = "cpu") -> torch.nn.Module:
    """Restores neuron from torch_state chunk."""

    with FoldReader(path, use_mmap=True) as reader:
        payload = reader.read_torch("torch_state", map_location=map_location)

    cfg_obj = payload.get("config")
    if isinstance(cfg_obj, dict):
        try:
            from pyfolds.core import MPJRDConfig

            cfg_obj = MPJRDConfig(**cfg_obj)
        except Exception:
            pass

    neuron = neuron_class(cfg_obj)
    neuron.load_state_dict(payload["state_dict"])
    return neuron


def peek_fold_or_mind(path: str) -> Dict[str, Any]:
    """Reads container header/index and manifest without loading tensors."""

    with FoldReader(path, use_mmap=True) as reader:
        out = {
            "header": reader.header,
            "chunks": reader.list_chunks(),
            "metadata": reader.index.get("metadata", {}),
        }

        if "manifest" in out["chunks"]:
            out["manifest"] = reader.read_json("manifest", verify=False)

        return out
