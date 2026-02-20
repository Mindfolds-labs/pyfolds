# FREEZE v2.1.1 — Core Hardening Report

## Escopo executado

1. **Ajuste temporal (Python 3.12)**
   - Verificação de `utcnow()` no core endurecido: sem ocorrências.
   - Logging estruturado e timestamp do dispatcher usando timezone UTC explícito.

2. **Segurança de pesos (PyTorch 2.6+)**
   - Registro explícito de globais seguras para `weights_only=True`:
     - `NeuronConfig`
     - `dict`
     - `list`
     - `str`
     - `float`
   - Mantida lista ampliada de tipos seguros já existente para compatibilidade retroativa.

3. **Ponto de saída agnóstico (MindDispatcher)**
   - Consolidada API `capture_event(...)` para exportação canônica (`origin`, `layer`, `ts`, `payload`).
   - Preservada compatibilidade de contrato legado via `prepare_payload(...)` para não quebrar consumidores atuais.

4. **Silenciamento de ruído em hooks de camada**
   - O processamento do forward da camada permanece encapsulado por `warnings.catch_warnings()` para suprimir `UserWarning` de saída em `dict`.

## Verificação de estresse

Comando executado:

```bash
PYTHONPATH=src pytest tests/ -v
```

Resultado:

- **320 passed**
- **1 skipped**
- **0 failed**
- **8 warnings**

## Observação de integridade

A meta de `0 warnings` **não foi atingida** no estado atual do repositório, porém os avisos remanescentes não vieram do dispatcher nem do warning de hooks em `layer.py`. Eles se concentram em:

- fallback de `torch.load(weights_only=False)` em cenários de checkpoint legado;
- warning de protocolo pickle do PyTorch;
- um warning induzido por teste de robustez de fechamento de mmap;
- warnings de depreciação cobertos por testes de superfície pública.

## Conclusão

O Core foi endurecido com ponte de exportação agnóstica e sem regressão funcional na suíte completa.
