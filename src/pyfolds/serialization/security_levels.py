from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Union


class SecurityLevel(str, Enum):
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    PARANOID = "paranoid"

    @classmethod
    def parse(cls, value: Union[str, "SecurityLevel"]) -> "SecurityLevel":
        if isinstance(value, SecurityLevel):
            return value
        normalized = str(value).strip().lower()
        try:
            return SecurityLevel(normalized)
        except ValueError as exc:
            raise ValueError(f"Nível de segurança inválido: {value!r}") from exc


@dataclass(frozen=True)
class SecurityConfig:
    level: SecurityLevel
    crc32c: bool = True
    sha256: bool = True
    ecc: bool = False
    signature: bool = False
    trust_block: bool = False
    merkle: bool = False
    encryption: bool = False
    provenance: bool = False
    sharding: bool = False
    recovery: bool = False


@lru_cache(maxsize=None)
def _get_security_config_cached(level_value: str) -> SecurityConfig:
    parsed = SecurityLevel.parse(level_value)
    if parsed is SecurityLevel.BASIC:
        return SecurityConfig(level=parsed)
    if parsed is SecurityLevel.STANDARD:
        return SecurityConfig(level=parsed, ecc=True, signature=True)
    if parsed is SecurityLevel.HIGH:
        return SecurityConfig(
            level=parsed,
            ecc=True,
            signature=True,
            trust_block=True,
            merkle=True,
            encryption=True,
        )
    return SecurityConfig(
        level=parsed,
        ecc=True,
        signature=True,
        trust_block=True,
        merkle=True,
        encryption=True,
        provenance=True,
        sharding=True,
        recovery=True,
    )


def get_security_config(level: Union[str, SecurityLevel]) -> SecurityConfig:
    return _get_security_config_cached(SecurityLevel.parse(level).value)
