from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class EpochMetrics:
    epoch: int
    loss: float
    train_acc_pct: float
    test_acc_pct: float
    spike_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
