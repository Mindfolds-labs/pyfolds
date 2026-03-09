from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def _hash(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=32).digest()


def _leaf_digest(chunk_meta: bytes) -> bytes:
    return _hash(b"L" + chunk_meta)


@dataclass(frozen=True)
class MerkleProofItem:
    sibling: bytes
    is_left: bool


class FastMerkleTree:
    def __init__(self, leaves: Iterable[bytes]):
        self.leaf_data = list(leaves)
        if not self.leaf_data:
            raise ValueError("Merkle tree requer ao menos uma folha")
        self.leaf_hashes = [_leaf_digest(x) for x in self.leaf_data]
        self.levels: List[List[bytes]] = [self.leaf_hashes]
        cur = self.leaf_hashes
        while len(cur) > 1:
            nxt: List[bytes] = []
            for i in range(0, len(cur), 2):
                left = cur[i]
                right = cur[i + 1] if i + 1 < len(cur) else cur[i]
                nxt.append(_hash(b"N" + left + right))
            self.levels.append(nxt)
            cur = nxt

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def get_proof(self, index: int) -> Sequence[MerkleProofItem]:
        if index < 0 or index >= len(self.leaf_hashes):
            raise IndexError("index fora do range")
        proof: List[MerkleProofItem] = []
        cur_idx = index
        for level in self.levels[:-1]:
            sib = cur_idx ^ 1
            if sib >= len(level):
                sib = cur_idx
            proof.append(MerkleProofItem(sibling=level[sib], is_left=(sib < cur_idx)))
            cur_idx //= 2
        return proof

    @staticmethod
    def verify(chunk_meta: bytes, proof: Sequence[MerkleProofItem], root: bytes, index: int) -> bool:
        h = _leaf_digest(chunk_meta)
        cur_idx = index
        for p in proof:
            if p.is_left:
                h = _hash(b"N" + p.sibling + h)
            else:
                h = _hash(b"N" + h + p.sibling)
            cur_idx //= 2
        return h == root
