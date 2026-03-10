"""Módulos de serialização e checkpointing."""

from .strategies import JSONStrategy, SerializationStrategy, TorchCheckpointStrategy, ZstdFoldStrategy
from .foldio import FoldReader, FoldWriter, load_fold_or_mind, peek_fold_or_mind, save_fold_or_mind
from .versioned_checkpoint import VersionedCheckpoint
from .ecc import ECCCodec, ECCProtector, ECCResult, NoECC, ReedSolomonECC, ecc_from_protection
from .security_levels import SecurityConfig, SecurityLevel, get_security_config
from .trust_block import TrustBlock, verify_header
from .merkle_fast import FastMerkleTree
from .encryption_fast import FastEncryptor
from .provenance_light import LightProvenance, LightProvenanceEntry
from .sharding_raid import RAIDSharding
from .async_io import AsyncFoldReader, AsyncFoldWriter
from .recovery import (
    attempt_chunk_repair,
    generate_recovery_report,
    locate_corrupted_chunks,
    reconstruct_from_shards,
    scan_fold_integrity,
    verify_trust_chain,
)
from .foldio import (
    FoldSecurityError,
    is_mind,
    peek_mind,
    read_nuclear_arrays,
)

__all__ = [
    "TorchCheckpointStrategy",
    "JSONStrategy",
    "ZstdFoldStrategy",
    "SerializationStrategy",
    "VersionedCheckpoint",
    "ECCCodec",
    "ECCProtector",
    "ECCResult",
    "NoECC",
    "ReedSolomonECC",
    "ecc_from_protection",
    "FoldReader",
    "FoldWriter",
    "FoldSecurityError",
    "save_fold_or_mind",
    "peek_fold_or_mind",
    "peek_mind",
    "read_nuclear_arrays",
    "load_fold_or_mind",
    "is_mind",
    "SecurityConfig",
    "SecurityLevel",
    "get_security_config",
    "TrustBlock",
    "verify_header",
    "FastMerkleTree",
    "FastEncryptor",
    "LightProvenance",
    "LightProvenanceEntry",
    "RAIDSharding",
    "AsyncFoldReader",
    "AsyncFoldWriter",
    "scan_fold_integrity",
    "locate_corrupted_chunks",
    "attempt_chunk_repair",
    "reconstruct_from_shards",
    "verify_trust_chain",
    "generate_recovery_report",
]
