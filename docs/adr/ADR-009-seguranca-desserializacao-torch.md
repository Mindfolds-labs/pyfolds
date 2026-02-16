# ADR-009 — Segurança na desserialização Torch

## Status
Accepted

## Contexto
`torch.load` pode carregar objetos arbitrários se usado sem restrições, elevando risco em arquivos não confiáveis.

## Decisão
Adotar `torch.load(..., weights_only=True)` por padrão na leitura de payload Torch, com modo explícito/trusted para casos controlados.

## Consequências
### Positivas
- redução de superfície de ataque em carga de checkpoints.

### Trade-offs
- possíveis limitações para checkpoints antigos que dependam de objetos fora de pesos/estados.

## Dependências
- ADR-003
- ADR-008
