"""Container .fold/.mind com chunking, leitura parcial, integridade e ECC opcional.

Dependências opcionais:
- zstandard: compressão/descompressão ZSTD.
- google-crc32c: cálculo CRC32C acelerado (recomendado para integridade).
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import mmap
import os
import platform
import struct
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .ecc import ECCCodec, NoECC, ReedSolomonECC, ecc_from_protection

def _optional_import(module_name: str) -> Any:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


zstd = _optional_import("zstandard")
google_crc32c = _optional_import("google_crc32c")


MAGIC = b"FOLDv1\0\0"
HEADER_FMT = ">8sIQQ"
CHUNK_HDR_FMT = ">4sIQQII"

FLAG_COMP_NONE = 0
FLAG_COMP_ZSTD = 1
MAX_INDEX_SIZE = 100 * 1024 * 1024

# Tabela CRC32C (Castagnoli) para fallback sem dependência externa.
_CRC32C_POLY = 0x82F63B78
_CRC32C_TABLE: List[int] = []
for _i in range(256):
    _crc = _i
    for _ in range(8):
        if _crc & 1:
            _crc = (_crc >> 1) ^ _CRC32C_POLY
        else:
            _crc >>= 1
    _CRC32C_TABLE.append(_crc & 0xFFFFFFFF)


def _crc32c_fallback(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


class FoldSecurityError(RuntimeError):
    """Erro de segurança ao desserializar payload torch."""


def crc32c_u32(data: bytes) -> int:
    if google_crc32c is not None:
        try:
            return int.from_bytes(google_crc32c.value(data).to_bytes(4, "big"), "big")
        except Exception:
            pass

    warnings.warn(
        "google-crc32c não instalado/disponível. Fallback para CRC32C em Python puro.",
        RuntimeWarning,
        stacklevel=2,
    )

    # Fallback de CRC32C (Castagnoli) sem dependências externas.
    # Implementação por bit refletido com polinômio reverso 0x82F63B78.
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x82F63B78
            else:
                crc >>= 1
    return (~crc) & 0xFFFFFFFF


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


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


def _safe_git_hash() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if out.returncode == 0:
        return out.stdout.strip()
    return "unknown"


def _telemetry_snapshot(neuron: Any, max_events: int = 128) -> Dict[str, Any]:
    telemetry = getattr(neuron, "telemetry", None)
    if telemetry is None:
        return {"enabled": False, "events": [], "stats": {}}

    events: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    try:
        stats = telemetry.get_stats()
    except Exception:
        stats = {}

    try:
        raw = telemetry.snapshot()
        if raw:
            events = raw[-max_events:]
    except Exception:
        events = []

    return {"enabled": True, "events": events, "stats": stats}


def _history_snapshot(neuron: Any) -> Optional[Dict[str, List[float]]]:
    stats_acc = getattr(neuron, "stats_acc", None)
    if stats_acc is None:
        return None

    if getattr(stats_acc, "_history_enabled", False) and hasattr(stats_acc, "_history"):
        return {k: list(v) for k, v in stats_acc._history.items()}
    return None


def _build_nuclear_npz(neuron: Any) -> bytes:
    payload: Dict[str, np.ndarray] = {}
    for key in ("N", "I", "W", "protection"):
        try:
            tensor = getattr(neuron, key, None)
            if tensor is not None and hasattr(tensor, "detach"):
                payload[key] = tensor.detach().cpu().numpy()
        except Exception as exc:
            warnings.warn(f"Falha ao serializar tensor '{key}': {exc}", RuntimeWarning)

    for key in ("theta", "r_hat"):
        tensor = getattr(neuron, key, None)
        if tensor is not None:
            try:
                payload[key] = np.array([float(tensor.item())], dtype=np.float32)
            except Exception:
                pass

    if not payload:
        return b""

    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    return buf.getvalue()


def read_nuclear_arrays(path: str, use_mmap: bool = True, verify: bool = True) -> Dict[str, np.ndarray]:
    """Lê apenas o chunk `nuclear_arrays` para análises científicas parciais."""

    with FoldReader(path, use_mmap=use_mmap) as reader:
        raw = reader.read_chunk_bytes("nuclear_arrays", verify=verify)
    with np.load(io.BytesIO(raw)) as data:
        return {k: data[k] for k in data.files}


class FoldWriter:
    """Writer do container .fold/.mind."""

    def __init__(
        self,
        path: str,
        compress: str = "zstd",
        zstd_level: int = 3,
        ecc: Optional[ECCCodec] = None,
    ):
        self.path = path
        self.compress = compress
        self.zstd_level = zstd_level
        self.ecc = ecc or NoECC()
        self._chunks: List[Dict[str, Any]] = []
        self._f = None

    def __enter__(self) -> "FoldWriter":
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "wb")
        header_len = struct.calcsize(HEADER_FMT)
        self._f.write(struct.pack(HEADER_FMT, MAGIC, header_len, 0, 0))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._f:
            self._f.close()

    def _compress(self, raw: bytes) -> Tuple[bytes, int]:
        if self.compress == "zstd":
            if zstd is None:
                raise RuntimeError("zstandard não instalado. Use compress='none' ou instale zstandard")
            compressor = zstd.ZstdCompressor(level=self.zstd_level)
            return compressor.compress(raw), FLAG_COMP_ZSTD
        return raw, FLAG_COMP_NONE

    def add_chunk(self, name: str, ctype4: str, payload: bytes) -> None:
        """Adiciona chunk com compressão, integridade e ECC opcional.

        Ordem de escrita:
            1. compressão (se habilitada)
            2. checksum (CRC32C/SHA256) do conteúdo comprimido
            3. codificação ECC do conteúdo comprimido
            4. persistência: [header | comp | ecc]

        Na leitura a ordem é invertida: leitura -> ECC -> verificação -> descompressão.
        """
        if len(ctype4) != 4:
            raise ValueError("ctype precisa ter 4 chars")

        comp, flags = self._compress(payload)
        ecc_result = self.ecc.encode(comp)

        crc = crc32c_u32(comp)
        sha = sha256_hex(comp)
        offset = self._f.tell()

        header = struct.pack(
            CHUNK_HDR_FMT,
            ctype4.encode("ascii"),
            flags,
            len(payload),
            len(comp),
            crc,
            len(ecc_result.ecc_bytes),
        )
        self._f.write(header)
        self._f.write(comp)
        if ecc_result.ecc_bytes:
            self._f.write(ecc_result.ecc_bytes)

        self._chunks.append(
            {
                "name": name,
                "ctype": ctype4,
                "flags": flags,
                "offset": offset,
                "header_len": struct.calcsize(CHUNK_HDR_FMT),
                "comp_len": len(comp),
                "uncomp_len": len(payload),
                "crc32c": crc,
                "sha256": sha,
                "ecc_algo": ecc_result.ecc_algo,
                "ecc_len": len(ecc_result.ecc_bytes),
            }
        )

    def finalize(self, metadata: Dict[str, Any]) -> None:
        chunk_hashes = {chunk["name"]: chunk["sha256"] for chunk in self._chunks}
        metadata = dict(metadata)
        metadata["chunk_hashes"] = chunk_hashes
        manifest_source = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        metadata["manifest_hash"] = sha256_hex(manifest_source)

        index = {
            "format": "fold",
            "version": "1.2.0",
            "created_at_unix": time.time(),
            "metadata": metadata,
            "chunks": self._chunks,
        }
        index_bytes = _json_bytes(index)

        phase = "capturar index_off"
        try:
            index_off = self._f.tell()

            phase = "escrever index"
            self._f.write(index_bytes)

            phase = "persistir index"
            self._f.flush()
            os.fsync(self._f.fileno())

            phase = "reposicionar para header"
            self._f.seek(0)

            phase = "reescrever header"
            header_len = struct.calcsize(HEADER_FMT)
            self._f.write(struct.pack(HEADER_FMT, MAGIC, header_len, index_off, len(index_bytes)))

            phase = "persistir header"
            self._f.flush()
            os.fsync(self._f.fileno())
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao finalizar arquivo fold na fase '{phase}' para '{self.path}': {exc}"
            ) from exc


class FoldReader:
    """Reader do container .fold/.mind com suporte a mmap."""

    def __init__(self, path: str, use_mmap: bool = True):
        self.path = path
        self.use_mmap = use_mmap
        self._f = None
        self._mm = None
        self.header: Dict[str, Any] = {}
        self.index: Dict[str, Any] = {}

    def __enter__(self) -> "FoldReader":
        self._f = open(self.path, "rb")
        if self.use_mmap:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._read_header_and_index()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        close_errors: List[str] = []
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception as mm_exc:
                msg = f"Erro ao fechar mmap de '{self.path}': {mm_exc}"
                close_errors.append(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            finally:
                self._mm = None
        if self._f:
            try:
                self._f.close()
            except Exception as file_exc:
                msg = f"Erro ao fechar arquivo '{self.path}': {file_exc}"
                close_errors.append(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            finally:
                self._f = None

        if close_errors and exc_type is None:
            raise RuntimeError("; ".join(close_errors))

    def _read_at(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError(f"offset e length devem ser >= 0, obtido offset={offset}, length={length}")

        if self._mm is not None:
            file_size = len(self._mm)
            if offset > file_size:
                raise EOFError(f"offset {offset} >= tamanho do arquivo {file_size}")
            if offset + length > file_size:
                raise EOFError(
                    "Tentativa de ler além do arquivo: "
                    f"offset={offset}, length={length}, tamanho={file_size}"
                )
            return bytes(self._mm[offset:offset + length])
        self._f.seek(offset)
        data = self._f.read(length)
        if len(data) < length:
            raise EOFError(
                "Fim de arquivo inesperado: "
                f"esperado {length} bytes, obtido {len(data)}"
            )
        return data

    def _read_header_and_index(self) -> None:
        header_size = struct.calcsize(HEADER_FMT)

        try:
            raw = self._read_at(0, header_size)
        except (OSError, IOError, EOFError) as exc:
            raise ValueError(f"Arquivo .fold/.mind inacessível ou truncado: {exc}") from exc

        if len(raw) < header_size:
            raise ValueError(
                f"Arquivo truncado: esperado {header_size} bytes no header, obtido {len(raw)}"
            )

        try:
            magic, hlen, index_off, index_len = struct.unpack(HEADER_FMT, raw)
        except struct.error as exc:
            raise ValueError(f"Falha ao interpretar header .fold/.mind: {exc}") from exc

        if magic != MAGIC:
            expected = MAGIC.decode("latin1", errors="replace")
            got = magic.decode("latin1", errors="replace")
            raise ValueError(
                "Arquivo .fold/.mind inválido. "
                f"Magic esperado: {expected!r}, obtido: {got!r}."
            )

        if hlen != header_size:
            raise ValueError(
                f"Header inconsistente: tamanho informado {hlen}, esperado {header_size}"
            )

        if index_off < hlen:
            raise ValueError(
                f"Index offset inválido: {index_off} < tamanho do header {hlen}"
            )

        if index_len > MAX_INDEX_SIZE:
            raise ValueError(
                f"Index muito grande: {index_len} bytes (máximo permitido {MAX_INDEX_SIZE})"
            )

        self.header = {"header_len": hlen, "index_off": index_off, "index_len": index_len}
        try:
            index_raw = self._read_at(index_off, index_len)
        except EOFError as exc:
            raise ValueError(
                f"Index truncado: não foi possível ler {index_len} bytes no offset {index_off}"
            ) from exc

        try:
            self.index = json.loads(index_raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(f"Index contém encoding inválido: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Index JSON corrompido: {exc}") from exc

    def list_chunks(self) -> List[str]:
        return [chunk["name"] for chunk in self.index.get("chunks", [])]

    def _decompress(self, comp: bytes, flags: int) -> bytes:
        if flags == FLAG_COMP_ZSTD:
            if zstd is None:
                raise RuntimeError("zstandard não instalado para leitura")
            dec = zstd.ZstdDecompressor()
            return dec.decompress(comp)
        return comp

    def _ecc_codec(self, algo: str) -> ECCCodec:
        if algo == "none":
            return NoECC()
        if algo.startswith("rs(") and algo.endswith(")"):
            symbols = int(algo[3:-1])
            return ReedSolomonECC(symbols=symbols)
        raise RuntimeError(f"Codec ECC não suportado: {algo}")

    def read_chunk_bytes(self, name: str, verify: bool = True) -> bytes:
        chunk = next((c for c in self.index.get("chunks", []) if c["name"] == name), None)
        if chunk is None:
            raise KeyError(f"Chunk '{name}' não existe")

        offset = chunk["offset"]
        hdr_size = struct.calcsize(CHUNK_HDR_FMT)
        hdr = self._read_at(offset, hdr_size)
        _ctype, flags, uncomp_len, comp_len, crc, ecc_len = struct.unpack(CHUNK_HDR_FMT, hdr)

        comp_off = offset + hdr_size
        comp = self._read_at(comp_off, comp_len)
        ecc_bytes = self._read_at(comp_off + comp_len, ecc_len) if ecc_len else b""

        ecc_algo = chunk.get("ecc_algo", "none")
        if ecc_algo != "none" and ecc_bytes:
            codec = self._ecc_codec(ecc_algo)
            comp = codec.decode(comp, ecc_bytes)

        if verify:
            if crc32c_u32(comp) != crc:
                raise RuntimeError(f"CRC32C inválido no chunk '{name}'")
            if sha256_hex(comp) != chunk["sha256"]:
                raise RuntimeError(f"SHA256 inválido no chunk '{name}'")

            metadata = self.index.get("metadata", {})
            expected_chunk_hash = metadata.get("chunk_hashes", {}).get(name)
            if expected_chunk_hash and expected_chunk_hash != chunk["sha256"]:
                raise RuntimeError(
                    f"Hash hierárquico inválido no chunk '{name}': "
                    "metadado e índice divergem"
                )

            expected_manifest = metadata.get("manifest_hash")
            if expected_manifest:
                manifest_data = {k: v for k, v in metadata.items() if k != "manifest_hash"}
                actual_manifest = sha256_hex(
                    json.dumps(manifest_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
                )
                if actual_manifest != expected_manifest:
                    warnings.warn(
                        "Manifest hash divergente: metadados podem estar corrompidos.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        raw = self._decompress(comp, flags)
        if len(raw) != uncomp_len:
            raise RuntimeError(f"Tamanho inconsistente no chunk '{name}'")
        return raw

    def read_json(self, name: str, verify: bool = True) -> Dict[str, Any]:
        return json.loads(self.read_chunk_bytes(name, verify=verify).decode("utf-8"))

    def read_torch(self, name: str, map_location: str = "cpu", verify: bool = True) -> Any:
        payload = self.read_chunk_bytes(name, verify=verify)
        kwargs: Dict[str, Any] = {"map_location": map_location}
        try:
            kwargs["weights_only"] = True
            return torch.load(io.BytesIO(payload), **kwargs)
        except Exception as exc:
            raise FoldSecurityError(
                "Falha ao carregar chunk torch em modo seguro (weights_only=True). "
                "Se o arquivo for confiável, use load_fold_or_mind(..., trusted_torch_payload=True)."
            ) from exc


class _TrustedFoldReader(FoldReader):
    """Reader dedicado para ambientes de confiança explícita."""

    def read_torch(self, name: str, map_location: str = "cpu", verify: bool = True) -> Any:
        payload = self.read_chunk_bytes(name, verify=verify)
        return torch.load(io.BytesIO(payload), map_location=map_location)


def _expression_summary(neuron: Any) -> Dict[str, Any]:
    def _tensor_scalar(attr: str) -> Optional[float]:
        tensor = getattr(neuron, attr, None)
        if tensor is None:
            return None
        try:
            return float(tensor.item())
        except Exception:
            return None

    def _mean(attr: str) -> Optional[float]:
        tensor = getattr(neuron, attr, None)
        if tensor is None:
            return None
        try:
            return float(tensor.detach().float().mean().item())
        except Exception:
            return None

    mode = getattr(getattr(neuron, "mode", None), "value", None)
    step = int(getattr(getattr(neuron, "step_id", None), "item", lambda: 0)())

    return {
        "mode": mode,
        "step_id": step,
        "theta": _tensor_scalar("theta"),
        "r_hat": _tensor_scalar("r_hat"),
        "mean_N": _mean("N"),
        "mean_I": _mean("I"),
        "mean_u": _mean("u"),
        "mean_R": _mean("R"),
    }


def _reproducibility_metadata() -> Dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "git_hash": _safe_git_hash(),
        "torch_initial_seed": int(torch.initial_seed()),
    }


def save_fold_or_mind(
    neuron: Any,
    path: str,
    tags: Optional[Dict[str, str]] = None,
    include_history: bool = True,
    include_telemetry: bool = True,
    include_nuclear_arrays: bool = True,
    compress: str = "zstd",
    ecc: Optional[ECCCodec] = None,
    protection: str = "off",
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> None:
    """Salva neurônio MPJRD em container .fold/.mind único e extensível."""

    selected_ecc = ecc if ecc is not None else ecc_from_protection(protection)

    cfg = _cfg_to_dict(getattr(neuron, "cfg", None))
    mode = getattr(getattr(neuron, "mode", None), "value", None)
    step = int(getattr(getattr(neuron, "step_id", None), "item", lambda: 0)())

    torch_payload = {
        "state_dict": neuron.state_dict(),
        "config": cfg,
        "mode": mode,
        "step_id": step,
    }
    torch_buffer = io.BytesIO()
    torch.save(torch_payload, torch_buffer)

    expression = _expression_summary(neuron)
    metrics = None
    if hasattr(neuron, "get_metrics"):
        try:
            metrics = neuron.get_metrics()
        except Exception:
            metrics = None

    history = _history_snapshot(neuron) if include_history else None
    telemetry = _telemetry_snapshot(neuron) if include_telemetry else None
    nuclear_arrays = _build_nuclear_npz(neuron) if include_nuclear_arrays else b""

    manifest = {
        "format": "fold",
        "version": "1.2.0",
        "created_at_unix": time.time(),
        "model_type": neuron.__class__.__name__,
        "tags": tags or {},
        "expression": expression,
        "reproducibility": _reproducibility_metadata(),
        "routing": {
            "resume_training": "torch_state",
            "audit": ["llm_manifest", "metrics", "history", "telemetry", "nuclear_arrays"],
        },
        "chunks_expected": [
            "torch_state",
            "llm_manifest",
            "metrics",
            "history",
            "telemetry",
            "nuclear_arrays",
        ],
    }
    if extra_manifest:
        manifest["extra"] = extra_manifest

    metadata = {
        "model_type": neuron.__class__.__name__,
        "tags": tags or {},
        "expression": expression,
        "reproducibility": manifest["reproducibility"],
        "protection": protection,
    }

    tmp_path = f"{path}.tmp"
    with FoldWriter(tmp_path, compress=compress, ecc=selected_ecc) as writer:
        writer.add_chunk("torch_state", "TSAV", torch_buffer.getvalue())
        writer.add_chunk("llm_manifest", "JSON", _json_bytes(manifest))
        if metrics is not None:
            writer.add_chunk("metrics", "JSON", _json_bytes(metrics))
        if history is not None:
            writer.add_chunk("history", "JSON", _json_bytes(history))
        if telemetry is not None:
            writer.add_chunk("telemetry", "JSON", _json_bytes(telemetry))
        if nuclear_arrays:
            writer.add_chunk("nuclear_arrays", "NPZ0", nuclear_arrays)
        writer.finalize(metadata=metadata)

    os.replace(tmp_path, path)


def peek_fold_or_mind(path: str, use_mmap: bool = True) -> Dict[str, Any]:
    """Inspeção rápida (header + índice + manifesto), sem carregar state_dict."""

    with FoldReader(path, use_mmap=use_mmap) as reader:
        output = {
            "header": reader.header,
            "metadata": reader.index.get("metadata", {}),
            "chunks": reader.list_chunks(),
        }
        if "llm_manifest" in output["chunks"]:
            output["llm_manifest"] = reader.read_json("llm_manifest")
        output["is_mind"] = is_mind_chunks(output["chunks"])
        return output


def peek_mind(path: str, use_mmap: bool = True) -> Dict[str, Any]:
    """Alias semântico para fluxos .mind."""

    return peek_fold_or_mind(path, use_mmap=use_mmap)


def load_fold_or_mind(
    path: str,
    neuron_class: Any,
    map_location: str = "cpu",
    trusted_torch_payload: bool = False,
) -> Any:
    """Carrega neurônio pelo chunk `torch_state` com validação por chunk."""

    reader_class = _TrustedFoldReader if trusted_torch_payload else FoldReader
    with reader_class(path, use_mmap=True) as reader:
        payload = reader.read_torch("torch_state", map_location=map_location)

    cfg = payload.get("config")
    if isinstance(cfg, dict):
        from ..core.config import MPJRDConfig

        cfg = MPJRDConfig.from_dict(cfg)

    neuron = neuron_class(cfg)
    neuron.load_state_dict(payload["state_dict"])
    return neuron


def is_mind(path: str) -> bool:
    """Regra de branding: arquivo com chunks IA explícitos é `.mind`."""

    with FoldReader(path, use_mmap=True) as reader:
        return is_mind_chunks(reader.list_chunks())


def is_mind_chunks(chunks: List[str]) -> bool:
    return any(name in set(chunks) for name in ("ai_graph", "ai_vectors"))
