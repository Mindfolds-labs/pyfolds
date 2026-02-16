# ADR-007 — Testes de corrupção e recuperação segura

## Status
Accepted

## Contexto
Para operação robusta, o parser precisa falhar com mensagens acionáveis diante de truncamento, offsets inválidos e checksums divergentes.

## Decisão
Padronizar suíte de robustez do formato:

- cenários mandatórios: truncamento, índice inválido, chunk inválido e hash divergente;
- mensagens de erro claras para diagnóstico operacional;
- estratégia de degradação segura (falha explícita sem comportamento indefinido).

## Consequências
### Positivas
- Melhor resposta a incidentes de dados corrompidos.
- Redução de risco de leitura parcialmente inválida.

### Trade-offs
- Aumento no custo de manutenção dos testes.
