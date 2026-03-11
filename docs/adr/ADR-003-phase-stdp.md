# ADR-003 — STDP orientado por fase

## Status
Accepted

## Contexto
Plasticidade dependente de timing exigia integração com estado de fase.

## Decisão
Usar estratégia de poda programada por fase (`pruning_strategy=phase_scheduled`) com gate periódico.

## Justificativa
Integra timing global (fase) com refinamento sináptico sem loops Python intensivos.

## Impactos e trade-offs
- Pró: regra vetorizada e observável.
- Contra: heurística de fase simplificada.

## Relação com literatura científica
Conceitualmente alinhado com theta-phase modulation em janelas de plasticidade.

## Limitações
Não implementa janela STDP bi-exponencial completa por sinapse baseada em spikes reais.

## Próximos passos
Acoplar traces STDP existentes com gate de fase contínuo.
