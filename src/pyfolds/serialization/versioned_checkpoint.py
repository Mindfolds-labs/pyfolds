"""Checkpoint versionado com metadados de reprodutibilidade."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import shutil
import subprocess
import warnings
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .ecc import ReedSolomonECC, ecc_from_protection

from pyfolds.core.config import MPJRDConfig

try:
    from safetensors.torch import load_file as load_safetensors_file
    from safetensors.torch import save_file as save_safetensors_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class VersionedCheckpoint:
    _safe_globals_registered = False

    """
    Salva/recupera estado do modelo com metadados e hash de integridade.

    ✅ CORRIGIDO:
        - Warnings para git hash ausente
        - Validação de versão ao carregar
        - Otimização de hash
        - Suporte a compressão
        - Suporte opcional a safetensors
        - Validação de shape antes de ``load_state_dict``
        - Sidecar ECC opcional para recuperação física de bitflips
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
        git_bin = shutil.which("git")
        if not git_bin:
            return "unknown"
        try:
            result = subprocess.run(
                [git_bin, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()

            warnings.warn(
                f"Git hash não disponível (código {result.returncode}): {result.stderr}",
                RuntimeWarning,
            )
            return "unknown"

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            warnings.warn(f"Git hash não disponível: {e}", RuntimeWarning)
            return "unknown"

    def _metadata(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Gera metadados para o checkpoint."""
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "version": self.version,
            "git_hash": self._git_hash(),
            "config": self._cfg_dict(),
            "pytorch_version": torch.__version__,
        }
        if extra:
            metadata.update(extra)
        return self._to_weights_only_safe(metadata)

    @classmethod
    def _register_safe_globals(cls) -> None:
        """Registra tipos permitidos para carregamento seguro (weights_only=True)."""
        if cls._safe_globals_registered:
            return

        safe_types = [
            dict,
            list,
            tuple,
            set,
            str,
            int,
            float,
            bool,
            bytes,
            type(None),
        ]
        safe_types.append(MPJRDConfig)

        torch.serialization.add_safe_globals(safe_types)
        cls._safe_globals_registered = True

    @classmethod
    def _to_weights_only_safe(cls, value: Any) -> Any:
        """Converte metadados para estrutura composta por tipos primitivos seguros."""
        if is_dataclass(value):
            value = asdict(value)

        if isinstance(value, dict):
            return {str(k): cls._to_weights_only_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_weights_only_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                scalar = value.item()
            except Exception:
                scalar = None
            if isinstance(scalar, (str, int, float, bool)) or scalar is None:
                return scalar
        if hasattr(value, "__dict__"):
            safe_obj = {k: v for k, v in vars(value).items() if not k.startswith("_")}
            return cls._to_weights_only_safe(safe_obj)
        return repr(value)

    def _compute_plain_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Calcula hash SHA-256 puro do state_dict."""
        hasher = hashlib.sha256()

        for key in sorted(state_dict.keys()):
            tensor = state_dict[key].detach().cpu().contiguous()
            hasher.update(key.encode("utf-8"))
            hasher.update(str(tensor.dtype).encode("utf-8"))
            hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
            hasher.update(tensor.numpy().tobytes())

        return hasher.hexdigest()

    def _compute_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Calcula digest de integridade do state dict (HMAC quando disponível)."""
        digest = self._compute_plain_hash(state_dict)
        hmac_key = os.getenv("PYFOLDS_CHECKPOINT_HMAC_KEY")
        if not hmac_key:
            return f"sha256:{digest}"
        signed = hmac.new(
            hmac_key.encode("utf-8"), digest.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"hmac-sha256:{signed}"

    @staticmethod
    def _verify_hash(plain_digest: str, saved: str) -> bool:
        if saved.startswith("hmac-sha256:"):
            hmac_key = os.getenv("PYFOLDS_CHECKPOINT_HMAC_KEY")
            if not hmac_key:
                return False
            expected = (
                "hmac-sha256:"
                + hmac.new(
                    hmac_key.encode("utf-8"),
                    plain_digest.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
            )
            return hmac.compare_digest(expected, saved)
        if saved.startswith("sha256:"):
            return hmac.compare_digest(f"sha256:{plain_digest}", saved)
        return hmac.compare_digest(plain_digest, saved)

    @staticmethod
    def _resolve_safetensors_paths(path: str) -> tuple[Path, Path]:
        base_path = Path(path)
        if base_path.suffix == ".safetensors":
            return base_path, base_path.with_suffix(".json")
        if base_path.suffix == ".json":
            return base_path.with_suffix(".safetensors"), base_path
        return base_path.with_suffix(".safetensors"), base_path.with_suffix(".json")

    @staticmethod
    def _decode_with_ecc(raw_payload: bytes, ecc_algo: str, ecc_path: Path) -> bytes:
        if not ecc_path.exists():
            raise ValueError(f"Sidecar ECC ausente: {ecc_path}")

        ecc_bytes = ecc_path.read_bytes()
        if ecc_algo.startswith("rs(") and ecc_algo.endswith(")"):
            symbols = int(ecc_algo[3:-1])
            codec = ReedSolomonECC(symbols=symbols)
        else:
            codec = ecc_from_protection("off")

        return codec.decode(raw_payload, ecc_bytes)

    @staticmethod
    def _validate_shape_invariants(
        model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Valida shape dos tensores do checkpoint antes do load no modelo."""
        for name, tensor in model.state_dict().items():
            loaded = state_dict.get(name)
            if loaded is None:
                continue
            if tuple(tensor.shape) != tuple(loaded.shape):
                raise ValueError(
                    f"Shape mismatch para {name}: "
                    f"Modelo {tuple(tensor.shape)} vs Checkpoint {tuple(loaded.shape)}"
                )

    def save(
        self,
        path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True,
        use_safetensors: bool = False,
    ) -> Dict[str, Any]:
        """
        Salva checkpoint e retorna payload persistido.

        Args:
            path: Caminho do arquivo .pt/.safetensors
            extra_metadata: Metadados adicionais
            compress: Usar compressão zip (recomendado para .pt)
            use_safetensors: Usa formato seguro sem pickle para pesos
            ecc_protection: Nível ECC de sidecar (off|low|med|high)

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

        file_path = Path(path)

        if use_safetensors:
            if not HAS_SAFETENSORS:
                raise RuntimeError(
                    "use_safetensors=True, mas o pacote 'safetensors' não está instalado"
                )

            safetensor_path = (
                file_path
                if file_path.suffix == ".safetensors"
                else file_path.with_suffix(".safetensors")
            )
            save_safetensors_file(state, str(safetensor_path))

            ckpt = {
                "metadata": metadata,
                "integrity_hash": integrity_hash,
                "format": "safetensors",
            }
            meta_path = safetensor_path.with_suffix(
                safetensor_path.suffix + ".meta.json"
            )
            meta_path.write_text(
                json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            # Salva com ou sem compressão
            if compress:
                torch.save(
                    ckpt, path, pickle_protocol=5, _use_new_zipfile_serialization=True
                )
            else:
                torch.save(ckpt, path)

        return ckpt

    @staticmethod
    def _validate_model_shapes(
        model: torch.nn.Module, model_state: Dict[str, torch.Tensor]
    ) -> None:
        current_state = model.state_dict()

        missing_keys = sorted(set(current_state.keys()) - set(model_state.keys()))
        unexpected_keys = sorted(set(model_state.keys()) - set(current_state.keys()))
        if missing_keys or unexpected_keys:
            raise ValueError(
                "State dict incompatível entre modelo e checkpoint. "
                f"missing_keys={missing_keys[:5]} unexpected_keys={unexpected_keys[:5]}"
            )

        for name, tensor in current_state.items():
            loaded = model_state[name]
            if torch.is_tensor(tensor) and torch.is_tensor(loaded):
                if tuple(tensor.shape) != tuple(loaded.shape):
                    raise ValueError(
                        f"Shape mismatch para {name}: "
                        f"Modelo {tuple(tensor.shape)} vs Checkpoint {tuple(loaded.shape)}"
                    )

    @classmethod
    def load(
        cls,
        path: str,
        model: Optional[torch.nn.Module] = None,
        map_location: str = "cpu",
        strict: bool = True,
        expected_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Carrega checkpoint, valida hash e opcionalmente restaura modelo.

        Args:
            path: Caminho do arquivo .pt/.safetensors
            model: Modelo para carregar o estado (opcional)
            map_location: Device para carregar tensores
            strict: Se True, valida hash e versão
            expected_version: Versão esperada (opcional)

        Returns:
            Dicionário com o checkpoint carregado

        Raises:
            ValueError: Se hash/shape não coincidirem
        """
        input_path = Path(path)
        if input_path.suffix == ".safetensors":
            if not HAS_SAFETENSORS:
                raise RuntimeError(
                    "Não foi possível carregar .safetensors: pacote 'safetensors' não está instalado"
                )
            model_state = load_safetensors_file(str(input_path), device=map_location)
            meta_path = input_path.with_suffix(input_path.suffix + ".meta.json")
            if meta_path.exists():
                ckpt = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                warnings.warn(
                    "Arquivo de metadados .meta.json ausente para checkpoint safetensors",
                    RuntimeWarning,
                )
                ckpt = {}

            ckpt["model_state"] = model_state
        else:
            try:
                cls._register_safe_globals()
                ckpt = torch.load(path, map_location=map_location, weights_only=True)
            except TypeError:
                # Compatibilidade com versões do PyTorch sem argumento weights_only
                ckpt = torch.load(path, map_location=map_location)
            except pickle.UnpicklingError as exc:
                # Compatibilidade com checkpoints serializados com protocolo/objetos
                # ainda não suportados pelo modo weights_only=True (PyTorch >=2.6).
                warnings.warn(
                    "Fallback para torch.load(weights_only=False) devido a "
                    f"incompatibilidade do formato seguro: {exc}",
                    RuntimeWarning,
                )
                ckpt = torch.load(path, map_location=map_location, weights_only=False)

        model_state = ckpt["model_state"]
        metadata = ckpt.get("metadata", {})
        saved_hash = ckpt.get("integrity_hash")

        if strict:
            if saved_hash is None:
                warnings.warn("Checkpoint sem hash de integridade", RuntimeWarning)
            else:
                verifier = cls(model=model or torch.nn.Identity(), version="verify")
                plain_hash = verifier._compute_plain_hash(model_state)
                if not cls._verify_hash(plain_hash, saved_hash):
                    current_hash = verifier._compute_hash(model_state)
                    raise ValueError(
                        "Falha na verificação de integridade do checkpoint.\n"
                        f"Esperado: {saved_hash}\n"
                        f"Obtido: {current_hash}"
                    )

            saved_version = metadata.get("version")
            if expected_version and saved_version != expected_version:
                warnings.warn(
                    f"Versão do checkpoint ({saved_version}) "
                    f"diferente da esperada ({expected_version})",
                    RuntimeWarning,
                )

        if model is not None:
            cls._validate_model_shapes(model, model_state)
            model.load_state_dict(model_state)

        return ckpt

    @classmethod
    def load_secure(cls, manifest_path: str, model: torch.nn.Module) -> Dict[str, Any]:
        """Carrega pesos via safetensors com validações rígidas de integridade."""
        if not HAS_SAFETENSORS:
            raise RuntimeError("load_secure requer o pacote 'safetensors' instalado")

        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

        weight_file = manifest.get("weight_file")
        if not weight_file:
            raise ValueError("Manifesto inválido: campo 'weight_file' ausente")

        weight_path = manifest_file.parent / weight_file
        tensors = load_safetensors_file(str(weight_path))

        cls._validate_model_shapes(model, tensors)

        saved_hash = manifest.get("integrity_hash")
        if saved_hash:
            verifier = cls(model=model, version="verify")
            plain_hash = verifier._compute_plain_hash(tensors)
            if not cls._verify_hash(plain_hash, saved_hash):
                raise ValueError(
                    "Falha de integridade: hash do manifesto não confere com pesos"
                )

        # Injeção somente após validações completas.
        model.load_state_dict(tensors, strict=True)
        return manifest.get("metadata", {})

    def __repr__(self) -> str:
        return f"VersionedCheckpoint(version={self.version})"
