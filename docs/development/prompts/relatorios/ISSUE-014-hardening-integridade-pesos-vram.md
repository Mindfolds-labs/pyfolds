# Relatório — ISSUE-014: Hardening de integridade de pesos (VRAM)

## Resumo

Foi implementado um monitor periódico de integridade por hash SHA-256 para detectar alterações inesperadas no `state_dict` durante execução longa.

## Entregas

- Novo `WeightIntegrityMonitor` no módulo de monitoramento.
- Export público do novo monitor em `pyfolds.monitoring`.
- Testes unitários cobrindo:
  - detecção de alteração de pesos;
  - respeito ao intervalo de checagem.
- ADR-046 publicada com contexto, decisão e trade-offs.

## Evidências de validação

- `pytest tests/unit/core/test_health_monitor.py`

## Resultado

Entrega concluída com cobertura unitária focada e rastreabilidade documental para a linha de hardening da versão 2.0.2.
