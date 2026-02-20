# ISSUE-012 — Validação de assinatura digital opcional e overhead de telemetria

## Objetivo
Validar dois pontos operacionais críticos:
1. assinatura digital `.fold/.mind` quando `cryptography` está instalada;
2. overhead de telemetria no `forward` com ring buffer ativo.

## Evidências executadas
- `python -m pip install cryptography`
- `pytest tests/unit/serialization/test_foldio.py::test_fold_signature_roundtrip_if_cryptography_available -v`
- `pytest tests/performance/test_telemetry_overhead.py -v -s -m "slow and performance"`
- benchmark adicional em Python (CPU) para extração de métricas de média/p95/overhead.

## Resultados
- Teste de assinatura digital: **PASS** (`1 passed`).
- Teste de overhead de telemetria: **PASS** (`1 passed`).
- Métricas medidas (CPU, `telemetry_profile="heavy"`):
  - `base_mean_ms=1.147574`
  - `telem_mean_ms=1.217941`
  - `overhead_ms=0.070367`
  - `overhead_pct=6.132`
  - `base_p95_ms=1.722571`
  - `telem_p95_ms=1.848684`

## Análise técnica
- A assinatura digital está funcional no cenário com dependência opcional presente, preservando o desenho de hardening para integridade de pesos.
- O overhead observado de telemetria pesada em CPU (~6.1%) é baixo para perfil de inspeção detalhada e está muito abaixo do guard-rail do teste de performance.
- Não foi detectada regressão funcional no pipeline de `forward` com telemetria ativa.

## Conclusão
Issue concluída com validação de segurança e desempenho no escopo solicitado.
