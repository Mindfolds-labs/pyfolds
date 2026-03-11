# ADR-002 — Coordenadas espaciais de neurônios e clusters

## Status
Accepted

## Contexto
A inspeção por região/cluster exigia estrutura espacial estável para conectividade/poda.

## Decisão
Representar conectividade estrutural por máscara 2D (`connectivity_mask`) por dendrito/sinapse.

## Justificativa
A máscara persiste no `state_dict` e mantém identidade estrutural do circuito.

## Impactos e trade-offs
- Pró: inspeção vetorizada por região.
- Contra: granularidade limitada ao grid dendrito×sinapse.

## Relação com literatura científica
Alinha com abstrações de topologia funcional e refinamento de circuito.

## Limitações
Não modela distância física real entre neurônios.

## Próximos passos
Incluir metadados de cluster anatômico e projeções cruzadas.
