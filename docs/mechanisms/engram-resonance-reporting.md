# Relatório de ressonância por engrama

## Objetivo do mecanismo
Expor métricas de ressonância e top engrams para inspeção.

## Base científica resumida
Memória distribuída pode ser observada por padrões de reativação e importância.

> **Nota de escopo científico**: este módulo usa uma aproximação computacional experimental inspirada em literatura de memória e replay. O relatório **não** implica equivalência biológica completa.

## Tradução computacional adotada
`NoeticCore.collect_engram_report()` combina cache de ressonância runtime e ranking de engrams.

Além do relatório base (`resonance_by_dendrite`, `mean_resonance`, `max_resonance`), a camada Noetic adiciona:
- `engram_count`
- `top_engrams` com `signature`, `concept`, `importance`, `access_count` e `consolidated`

Isso amplia a telemetria para inspeção de priorização de memória durante ciclos de uso e sono.

## Arquivos do código afetados
- `src/pyfolds/advanced/noetic_model.py`

## Flags de ativação/desativação
Sem flag dedicada para o *report* em si; depende de existência do banco de engrams.

Flags experimentais relacionadas ao contexto dos dados observados:
- `replay_batch_size`: controla volume de replay por ciclo de sono.
- `consolidate_pruning_after_replay`: quando ativada no neurônio base, permite consolidar poda após replay offline.
- `pruning_strategy="replay_consolidated"`: estratégia experimental de poda orientada por replay/consolidação.

## Riscos de implementação
Ranking por importância/acesso é heurístico.

## Estratégia de teste
Validação funcional em integração Noetic (futuro teste dedicado).

## Critérios de observabilidade/debug
`collect_engram_report(top_k=...)`.

## Comportamento offline (consolidação/replay)
Durante `sleep()`, o fluxo Noetic executa replay e consolidação do banco de engrams, e o relatório posterior pode refletir mudanças em `consolidated` e no ranking por acesso/importância.

## Baseline safety
Com flags experimentais desligadas (ex.: `consolidate_pruning_after_replay=False` e estratégias não orientadas por replay), o caminho estável permanece: coleta de ressonância + ranking básico, sem consolidação adicional induzida por replay.
