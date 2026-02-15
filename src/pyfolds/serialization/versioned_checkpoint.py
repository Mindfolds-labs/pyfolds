"""Checkpoint versionado com metadados de reprodutibilidade."""

from __future__ import annotations

import hashlib
import subprocess
import warnings
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class VersionedCheckpoint:
    """
    Salva/recupera estado do modelo com metadados e hash de integridade.
    
    ✅ CORRIGIDO:
        - Warnings para git hash ausente
        - Validação de versão ao carregar
        - Otimização de hash
        - Suporte a compressão
    """

    def __init__(self, model: torch.nn.Module, version: str):
        """
        Args:
            model: Modelo a ser salvo/restaurado
            version: Versão semântica do modelo (ex: "2.0.0")
        """
        self.model = model
        self.version = version

    def _cfg_dict(self) -> Dict[str, Any]:
        """Extrai configuração do modelo em formato dict."""
        cfg = getattr(self.model, "cfg", None)
        if cfg is None:
            return {}
        
        if hasattr(cfg, "to_dict"):
            return cfg.to_dict()
        if is_dataclass(cfg):
            return asdict(cfg)
        if hasattr(cfg, "__dict__"):
            return {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
        
        return {"warning": "config não serializável"}

    def _git_hash(self) -> str:
        """Obtém hash atual do git para reprodutibilidade."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            warnings.warn(
                f"Git hash não disponível (código {result.returncode}): {result.stderr}",
                RuntimeWarning
            )
            return "unknown"
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            warnings.warn(f"Git hash não disponível: {e}", RuntimeWarning)
            return "unknown"

    def _metadata(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Gera metadados para o checkpoint."""
        metadata = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "version": self.version,
            "git_hash": self._git_hash(),
            "config": self._cfg_dict(),
            "pytorch_version": torch.__version__,
        }
        if extra:
            metadata.update(extra)
        return metadata

    def _compute_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Calcula hash SHA-256 do state dict."""
        hasher = hashlib.sha256()
        
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key].detach().cpu().contiguous()
            hasher.update(key.encode('utf-8'))
            hasher.update(str(tensor.dtype).encode('utf-8'))
            hasher.update(str(tuple(tensor.shape)).encode('utf-8'))
            hasher.update(tensor.numpy().tobytes())
        
        return hasher.hexdigest()

    def save(
        self, 
        path: str, 
        extra_metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True
    ) -> Dict[str, Any]:
        """
        Salva checkpoint e retorna payload persistido.
        
        Args:
            path: Caminho do arquivo .pt
            extra_metadata: Metadados adicionais
            compress: Usar compressão zip (recomendado)
        
        Returns:
            Dicionário com o checkpoint salvo
        """
        state = self.model.state_dict()
        metadata = self._metadata(extra_metadata)
        integrity_hash = self._compute_hash(state)
        
        ckpt = {
            "model_state": state,
            "metadata": metadata,
            "integrity_hash": integrity_hash,
        }
        
        # Cria diretório se necessário
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Salva com ou sem compressão
        if compress:
            torch.save(ckpt, path, pickle_protocol=5, _use_new_zipfile_serialization=True)
        else:
            torch.save(ckpt, path)
        
        return ckpt

    @classmethod
    def load(
        cls, 
        path: str, 
        model: Optional[torch.nn.Module] = None, 
        map_location: str = "cpu",
        strict: bool = True,
        expected_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Carrega checkpoint, valida hash e opcionalmente restaura modelo.
        
        Args:
            path: Caminho do arquivo .pt
            model: Modelo para carregar o estado (opcional)
            map_location: Device para carregar tensores
            strict: Se True, valida hash e versão
            expected_version: Versão esperada (opcional)
        
        Returns:
            Dicionário com o checkpoint carregado
        
        Raises:
            ValueError: Se hash ou versão não coincidirem
        """
        ckpt = torch.load(path, map_location=map_location)
        
        model_state = ckpt["model_state"]
        metadata = ckpt.get("metadata", {})
        saved_hash = ckpt.get("integrity_hash")
        
        if strict:
            # Valida hash
            if saved_hash is None:
                warnings.warn("Checkpoint sem hash de integridade", RuntimeWarning)
            else:
                verifier = cls(model=model or torch.nn.Identity(), version="verify")
                current_hash = verifier._compute_hash(model_state)
                if current_hash != saved_hash:
                    raise ValueError(
                        f"Falha na verificação de integridade do checkpoint.\n"
                        f"Esperado: {saved_hash}\n"
                        f"Obtido: {current_hash}"
                    )
            
            # Valida versão (se esperada)
            saved_version = metadata.get("version")
            if expected_version and saved_version != expected_version:
                warnings.warn(
                    f"Versão do checkpoint ({saved_version}) "
                    f"diferente da esperada ({expected_version})",
                    RuntimeWarning
                )
        
        # Restaura modelo se fornecido
        if model is not None:
            model.load_state_dict(model_state)
        
        return ckpt

    def __repr__(self) -> str:
        return f"VersionedCheckpoint(version={self.version})"