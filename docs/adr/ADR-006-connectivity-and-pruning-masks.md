# ADR-006 — Máscaras de conectividade e poda

## Status
Accepted

## Contexto
Faltava separação clara entre estrutura conectiva e poda dinâmica.

## Decisão
Introduzir:
- `connectivity_mask` (buffer persistente estrutural)
- `pruning_mask` (buffer persistente de refinamento)

Aplicar máscara efetiva vetorizada no cálculo de potencial dendrítico.

## Justificativa
Materializa estrutura e pruning de forma inspecionável, serializável e compatível com `.to(device)`.

## Impactos e trade-offs
- Pró: evita lógica implícita de “zerar ativação”.
- Contra: adiciona custo leve de multiplicação de máscara.

## Relação com literatura científica
Compatível com noções de synaptic pruning/circuit refinement em nível abstrato computacional.

## Limitações
A regra de poda atual é limiar de magnitude, não regra biológica completa.

## Próximos passos
Adicionar critérios de estabilidade, uso e competição local para poda.
