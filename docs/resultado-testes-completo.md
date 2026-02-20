# Resultado da suíte completa de testes — PyFolds

Data: 2026-02-20  
Ambiente: Python 3.12.12 / pytest 9.0.2

## Objetivo
Executar a suíte completa do projeto para identificar possíveis falhas após a correção anterior.

## Comandos executados

1. Tentativa inicial com cobertura (alvo equivalente ao `make test`):

```bash
PYTHONPATH=src pytest tests/ -v --cov=pyfolds --cov-report=term-missing
```

**Resultado:** não executou por ausência do plugin de cobertura (`pytest-cov`) no ambiente atual, com erro de argumentos `--cov` não reconhecidos.

2. Execução completa da suíte (sem cobertura):

```bash
PYTHONPATH=src pytest tests/ -v
```

## Resultado final da suíte completa

- **Total coletado:** 317 testes
- **Passaram:** 316
- **Falharam:** 0
- **Pulados:** 1
- **Warnings:** 58
- **Tempo total:** 38.26s

Resumo do pytest:

```text
================= 316 passed, 1 skipped, 58 warnings in 38.26s =================
```

## Observações relevantes

- Não foram identificadas falhas de teste na suíte completa.
- Foram emitidos warnings (ex.: deprecações e avisos de serialização/torch), mas sem impacto de falha nesta execução.
- Se desejado, o próximo passo pode ser habilitar `pytest-cov` no ambiente para restaurar a execução com cobertura.
