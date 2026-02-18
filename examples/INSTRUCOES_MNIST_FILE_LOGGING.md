# Instruções — MNIST com PyFolds (log em arquivo)

## Imports recomendados
```python
from pyfolds import MPJRDConfig, MPJRDLayer, LearningMode, VersionedCheckpoint
from pyfolds.advanced import MPJRDNeuronAdvanced
```

## Comandos principais
```bash
# instalar em modo editable
python -m pip install -e .

# executar treino exemplo
python examples/mnist_file_logging.py

# rodar teste de integração do exemplo
pytest -q tests/integration/test_mnist_file_logging.py
```

## O que o exemplo valida
- Compatibilidade de imports `pyfolds` (core + advanced).
- Pipeline de treino completo com entrada adaptada para o formato da camada MPJRD.
- Fallback automático para dataset sintético quando MNIST não puder ser baixado.
- Geração de log em arquivo (`logs/pyfolds_*.log`).
- Salvamento de checkpoint versionado via `VersionedCheckpoint`.

## Como salvar e recarregar modelo (compatível com Folds)
```python
# salvar
checkpoint = VersionedCheckpoint(model, version="2.0.0")
checkpoint.save("outputs/meu_modelo.pt", extra_metadata={"run": "mnist"})

# recarregar
payload = VersionedCheckpoint.load(
    "outputs/meu_modelo.pt",
    model=model,
    map_location="cpu",
    strict=True,
    expected_version="2.0.0",
)
```

## Boas práticas
- Sempre usar o mesmo `device` para neurônio/camada e entrada.
- Garantir shape de entrada válido para `MPJRDLayer`: `[B,N,D,S]`, `[B,N,D]` ou `[B,D,S]`.
- Manter batchs pequenos em testes para execução rápida e reproduzível.
