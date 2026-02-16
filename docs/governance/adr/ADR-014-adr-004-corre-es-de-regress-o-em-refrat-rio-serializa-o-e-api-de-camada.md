# ADR-004 — Correções de regressão em refratário, serialização e API de camada

## Status
Accepted

## Contexto
A suíte de testes identificou regressões que afetavam consistência comportamental e robustez de serialização.

## Decisão
1. Refratário relativo sem bloqueio direto (bloqueio apenas na janela absoluta).
2. Fallback de checksum corrigido para CRC32C (Castagnoli) quando não houver `google-crc32c`.
3. Remoção do atributo legado `neuron_class` em favor de `neuron_cls`.
4. Ajustes em testes ECC para não depender exclusivamente de `reedsolo`.

## Consequências
- comportamento alinhado aos contratos documentados;
- menor fragilidade da suíte em ambientes variáveis;
- possível impacto para integrações externas dependentes de API legada.

## Dependências
- ADR-003.
