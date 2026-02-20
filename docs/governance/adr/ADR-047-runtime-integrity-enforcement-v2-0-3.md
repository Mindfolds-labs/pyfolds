# ADR-047 — Runtime Integrity Enforcement v2.0.3

- **Status:** Ativo
- **Data:** 2026-02-20
- **Decisores:** Engenharia Core, Segurança de Runtime, Governança de Release
- **Contexto:** Evolução do hardening iniciado em ADR-045/ADR-046 para fechar lacunas de integridade em memória, throughput de telemetria e carga segura de pesos.

## Decisão

Implementar o pacote final de hardening da release `2.0.3` com três eixos técnicos:

1. **Monitor de integridade em runtime (VRAM):** `WeightIntegrityMonitor` com hash determinístico periódico do `state_dict`.
2. **I/O de telemetria com buffer:** `BufferedJSONLinesSink` para reduzir churn de escrita em cenários de alta frequência.
3. **Carregamento seguro orientado a manifesto:** `VersionedCheckpoint.load_secure(...)` com validação de chaves, shapes e hash antes da injeção de estado no modelo.

## Justificativa

- Bitflips e corrupção transitória de hardware podem passar despercebidos em runs longos.
- Escritas síncronas evento-a-evento degradam throughput de treino.
- Fluxos de carga devem falhar rápido quando manifesto/pesos divergirem em integridade ou shape.

## Consequências

### Positivas
- Melhora da tolerância a falhas de runtime em produção.
- Menor overhead de disco para telemetria intensa.
- Carga de checkpoint mais previsível e auditável.

### Trade-offs
- Hash periódico adiciona custo de snapshot para CPU.
- Buffer de telemetria demanda `flush/close` disciplinado.

## Rastreabilidade

- `src/pyfolds/monitoring/health.py`
- `src/pyfolds/telemetry/sinks.py`
- `src/pyfolds/serialization/versioned_checkpoint.py`
- `tests/unit/core/test_monitoring_and_checkpoint.py`
- `tests/unit/telemetry/test_sinks.py`
