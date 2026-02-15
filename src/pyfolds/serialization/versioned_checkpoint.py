"""Checkpoint versionado com metadados de reprodutibilidade."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class VersionedCheckpoint:
    """Salva/recupera estado do modelo com metadados e hash de integridade."""

    def __init__(self, model: torch.nn.Module, version: str):
        self.model = model
        self.version = version

    def _cfg_dict(self) -> Dict[str, Any]:
        cfg = getattr(self.model, "cfg", None)
        if cfg is None:
            return {}
        if hasattr(cfg, "to_dict"):
            return cfg.to_dict()
        if is_dataclass(cfg):
            return asdict(cfg)
        return dict(cfg.__dict__)

    def _git_hash(self) -> str:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    def _metadata(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "version": self.version,
            "git_hash": self._git_hash(),
            "config": self._cfg_dict(),
        }
        if extra:
            metadata.update(extra)
        return metadata

    def _compute_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key].detach().cpu().contiguous()
            hasher.update(key.encode())
            hasher.update(str(tensor.dtype).encode())
            hasher.update(str(tuple(tensor.shape)).encode())
            hasher.update(tensor.numpy().tobytes())
        return hasher.hexdigest()

    def save(self, path: str, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Salva checkpoint e retorna payload persistido."""
        state = self.model.state_dict()
        ckpt = {
            "model_state": state,
            "metadata": self._metadata(extra_metadata),
            "integrity_hash": self._compute_hash(state),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        return ckpt

    @staticmethod
    def load(path: str, model: Optional[torch.nn.Module] = None, map_location: str = "cpu") -> Dict[str, Any]:
        """Carrega checkpoint, valida hash e opcionalmente restaura modelo."""
        ckpt = torch.load(path, map_location=map_location)
        model_state = ckpt["model_state"]

        verifier = VersionedCheckpoint(model=model or torch.nn.Identity(), version="verify")
        expected = verifier._compute_hash(model_state)
        if expected != ckpt.get("integrity_hash"):
            raise ValueError("Falha na verificação de integridade do checkpoint")

        if model is not None:
            model.load_state_dict(model_state)

        return ckpt
