# Hardening Final v2.1.1

## Escopo executado
- Validação de temporalidade UTC no código-fonte (`datetime.now(UTC)` / `datetime.now(datetime.UTC)`).
- Blindagem de checkpoint para `weights_only=True` com registro de globals seguros.
- Isolamento de exportação externa via dispatcher agnóstico (`pyfolds.bridge.dispatcher`).
- Supressão direcionada de ruído de warning de hooks de dicionário em camadas.
- Execução completa da suíte de testes.

## Comandos executados
```bash
rg "utcnow\(" -n
PYTHONPATH=src pytest tests/ -v
```

## Resultado da validação
- Testes: **320 passed, 1 skipped, 0 failed**.
- Warnings emitidos pela suíte: **8** (redução substancial frente ao baseline histórico de 58 warnings reportado no hardening anterior).

### Observações sobre warnings remanescentes
1. 2 warnings de compatibilidade de protocolo pickle no `torch.load`.
2. 2 warnings de fallback explícito para `weights_only=False` em arquivo legado.
3. 1 warning induzido por teste de erro forçado de `mmap.close`.
4. 3 warnings de depreciação de aliases v1 na superfície pública.

Esses warnings remanescentes são controlados/esperados no contexto atual dos testes e não geraram falhas.
