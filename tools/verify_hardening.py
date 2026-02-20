from datetime import UTC, datetime

import torch

from pyfolds import NeuronConfig
from pyfolds.serialization.versioned_checkpoint import VersionedCheckpoint


def check_integrity() -> None:
    print("--- Verificando Blindagem PyFolds ---")

    now = datetime.now(UTC)
    print(f"[OK] Timezone UTC compatível com Python 3.12: {now.isoformat()}")

    VersionedCheckpoint._register_safe_globals()
    print("[OK] Registro de segurança de tipos (Safe Globals) validado.")

    cfg = NeuronConfig(n_dendrites=1, n_synapses_per_dendrite=1, device="cpu")
    assert cfg.n_dendrites == 1
    print(f"[OK] Configuração básica validada com torch {torch.__version__}.")


if __name__ == "__main__":
    check_integrity()
