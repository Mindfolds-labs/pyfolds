# ADR-046 — Sanity check periódico de integridade de pesos

- **Status:** Ativo
- **Data:** 2026-02-20
- **Decisores:** Engenharia de Runtime, Segurança de Modelo e Confiabilidade
- **Contexto:** Como continuidade da blindagem da release `2.0.2` (ADR-045), foi priorizada uma verificação periódica de integridade para detecção precoce de corrupção silenciosa de parâmetros em execuções longas.

## Contexto

A cadeia atual já protege checkpoints em repouso (HMAC SHA-256 + `compare_digest`, `safetensors`, sidecar de metadados e ECC opcional). Entretanto, a equipe identificou no backlog especializado um risco residual: corrupção em memória durante run de longa duração.

## Decisão

Adicionar monitoramento periódico de hash SHA-256 do `state_dict` em runtime, com as seguintes regras:

1. **Novo monitor de integridade:** `ModelIntegrityMonitor` em `pyfolds.monitoring`.
2. **Baseline explícito ou lazy:** baseline via `set_baseline()` ou inicialização automática no primeiro `check_integrity()`.
3. **Checagem por intervalo:** `check_every_n_steps` para reduzir overhead em cenários de alto throughput.
4. **Detecção determinística de drift:** retorno explícito de `integrity_ok`, `expected_hash` e `current_hash`.

## Consequências

### Positivas
- Capacidade de detecção precoce de alterações inesperadas de pesos em runtime.
- Integração simples com pipelines existentes de monitoramento e telemetria.
- Mantém consistência com estratégia de hardening já adotada para checkpoints.

### Trade-offs
- A checagem é baseada em leitura de estado para CPU para cálculo do digest, podendo adicionar overhead se configurada com frequência alta.
- Em cenário de treino ativo, a verificação deve ser usada com baseline revalidado por etapa lógica (senão mudanças legítimas de otimização aparecerão como drift).

## Implementação vinculada

- `src/pyfolds/monitoring/health.py`
- `src/pyfolds/monitoring/__init__.py`
- `src/pyfolds/__init__.py`
- `tests/unit/core/test_monitoring_and_checkpoint.py`

## Referências

- `docs/governance/adr/ADR-045-checkpoints-seguros-com-safetensors-flush-lazy-e-shape-validation.md`
- `CHANGELOG.md`
