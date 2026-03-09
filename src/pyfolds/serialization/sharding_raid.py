from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RAIDSharding:
    data_shards: int = 4
    parity_shards: int = 1

    def split(self, data: bytes) -> List[bytes]:
        if self.data_shards < 2:
            raise ValueError("data_shards deve ser >= 2")
        shard_size = math.ceil(len(data) / self.data_shards)
        padded = data + (b"\x00" * (shard_size * self.data_shards - len(data)))
        shards = [padded[i * shard_size : (i + 1) * shard_size] for i in range(self.data_shards)]
        parity = bytearray(shard_size)
        for shard in shards:
            for i, b in enumerate(shard):
                parity[i] ^= b
        return shards + [bytes(parity)]

    def reconstruct(self, available_shards: Sequence[bytes], available_indices: Sequence[int]) -> bytes:
        total = self.data_shards + self.parity_shards
        if len(available_shards) != len(available_indices):
            raise ValueError("shards e índices devem ter o mesmo tamanho")
        if len(available_shards) < self.data_shards:
            raise ValueError("Shards insuficientes para reconstrução")
        by_idx = {idx: shard for idx, shard in zip(available_indices, available_shards)}
        shard_size = len(available_shards[0])
        if any(len(s) != shard_size for s in available_shards):
            raise ValueError("Todos os shards devem ter mesmo tamanho")

        if len(by_idx) == total:
            data = b"".join(by_idx[i] for i in range(self.data_shards))
            return data.rstrip(b"\x00")

        missing = [i for i in range(total) if i not in by_idx]
        if len(missing) > 1:
            raise ValueError("Implementação atual suporta reconstrução de apenas 1 shard perdido")
        miss = missing[0]
        rec = bytearray(shard_size)
        for i in range(total):
            if i == miss:
                continue
            shard = by_idx[i]
            for j, b in enumerate(shard):
                rec[j] ^= b
        by_idx[miss] = bytes(rec)
        data = b"".join(by_idx[i] for i in range(self.data_shards))
        return data.rstrip(b"\x00")
