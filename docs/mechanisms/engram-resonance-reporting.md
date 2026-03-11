# Relatório de ressonância por engrama

## Objetivo do mecanismo
Expor métricas de ressonância e top engrams para inspeção.

## Base científica resumida
Memória distribuída pode ser observada por padrões de reativação e importância.

## Tradução computacional adotada
`NoeticCore.collect_engram_report()` combina cache de ressonância runtime e ranking de engrams.

## Arquivos do código afetados
- `src/pyfolds/advanced/noetic_model.py`

## Flags de ativação/desativação
Sem flag dedicada; depende de existência do banco de engrams.

## Riscos de implementação
Ranking por importância/acesso é heurístico.

## Estratégia de teste
Validação funcional em integração Noetic (futuro teste dedicado).

## Critérios de observabilidade/debug
`collect_engram_report(top_k=...)`.
