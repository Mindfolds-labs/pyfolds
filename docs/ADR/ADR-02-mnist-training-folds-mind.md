# ADR-02: Pipeline de treinamento MNIST com persistência Folds/Mind e suporte a resume

- **Status:** Proposto
- **Data:** 2026-02-19

## Decisão

Padronizar treino MNIST com:
- backends de persistência intercambiáveis (`folds`, `mind`)
- checkpoint (`checkpoint.pt`) com `model_state`, `optimizer_state`, `epoch`
- métricas essenciais (`loss`, `acc_pct`, `spike_rate`) e resumo final
- dois modelos (`ModelPy`, `ModelWave`)
- execução por scripts locais (`train_mnist_folds.py`, `train_mnist_mind.py`)

## Contrato do modelo

- `forward(x, state=None) -> (logits, out)`
- `init_state(batch_size, device) -> state`
- `detach_state(state) -> state`
- `get_config() -> dict`

`out` deve incluir `state`, `spike_rate`, `v_mean` e `spikes`.

## Observabilidade e operação

- `console` desligado por padrão; saída principal em `runs/<run-id>/train.log`.
- `metrics.jsonl` e `summary.json` são sempre gerados.
- Em falha: `logger.exception(...)` + `ADR-ERR-<timestamp>.md` + issue draft em `docs/issues/`.
- Execução PowerShell empacota crash bundle com logs e artefatos para retrabalho.
