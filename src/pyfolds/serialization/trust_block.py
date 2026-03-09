from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from .foldio import _sign_payload_ed25519, _verify_payload_signature_ed25519, sha256_hex


TRUST_BLOCK_VERSION = 1


@dataclass(frozen=True)
class TrustBlock:
    version: int
    metadata_hash: str
    index_hash: str
    signature_algo: str
    signature: str
    key_id: str = "default"

    def to_payload_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "metadata_hash": self.metadata_hash,
            "index_hash": self.index_hash,
            "signature_algo": self.signature_algo,
            "key_id": self.key_id,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            {**self.to_payload_dict(), "signature": self.signature},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "TrustBlock":
        try:
            parsed = json.loads(data.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"TrustBlock inválido/corrompido: {exc}") from exc
        version = int(parsed.get("version", -1))
        if version != TRUST_BLOCK_VERSION:
            raise ValueError(f"Versão de TrustBlock não suportada: {version}")
        required = ("metadata_hash", "index_hash", "signature_algo", "signature", "key_id")
        missing = [k for k in required if k not in parsed]
        if missing:
            raise ValueError(f"TrustBlock incompleto: campos ausentes {missing}")
        return cls(
            version=version,
            metadata_hash=str(parsed["metadata_hash"]),
            index_hash=str(parsed["index_hash"]),
            signature_algo=str(parsed["signature_algo"]),
            signature=str(parsed["signature"]),
            key_id=str(parsed["key_id"]),
        )

    @classmethod
    def build(cls, metadata: Dict[str, Any], index: Dict[str, Any], private_key_pem: str, key_id: str = "default") -> "TrustBlock":
        payload = {
            "version": TRUST_BLOCK_VERSION,
            "metadata_hash": sha256_hex(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")),
            "index_hash": sha256_hex(json.dumps(index, sort_keys=True, separators=(",", ":")).encode("utf-8")),
            "signature_algo": "ed25519",
            "key_id": key_id,
        }
        signature = _sign_payload_ed25519(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            private_key_pem,
        )
        return cls(signature=signature, **payload)

    def verify(self, public_key_pem: str) -> bool:
        if self.signature_algo != "ed25519":
            raise ValueError(f"Algoritmo de assinatura não suportado: {self.signature_algo}")
        payload = json.dumps(self.to_payload_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return _verify_payload_signature_ed25519(payload, self.signature, public_key_pem)


def verify_header(data: bytes, public_key_pem: str) -> bool:
    return TrustBlock.from_bytes(data).verify(public_key_pem)
