# ADR-008 — Estabilidade de API e semântica refratária

## Status
Accepted

## Contexto
Falhas anteriores mostraram risco em API legada ambígua e semântica refratária divergente.

## Decisão
Consolidar API oficial (`neuron_cls`) e preservar semântica documentada do refratário relativo (sem bloqueio direto fora da janela absoluta).

## Consequências
- + previsibilidade de contrato para usuários e testes.
- - código externo legado pode requerer ajustes.
