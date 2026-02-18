# ADR-038 — Auditoria contínua do formato `.fold/.mind` com governança de execução (ISSUE-036)

## Status
Aceito

## Contexto
Houve solicitação de validação completa do fluxo de persistência `.fold/.mind`, incluindo:
- verificação de integridade e segurança de serialização/desserialização;
- revisão de possíveis erros lógicos;
- criação de trilha formal de governança com ISSUE/EXEC/HUB.

Também foi identificado que o fluxo de validação de assinatura digital precisava padronizar exceções para manter contrato de segurança em alto nível.

## Decisão
1. Manter o modelo `.fold/.mind` atual como baseline arquitetural (header + chunks + índice + hashes + ECC opcional).
2. Padronizar falhas na validação de assinatura digital para `FoldSecurityError` no caminho de `load_fold_or_mind`.
3. Instituir, para auditorias similares, o fluxo mínimo obrigatório:
   - executar suíte focada de serialização/corrupção;
   - registrar ISSUE + EXEC;
   - atualizar `execution_queue.csv`;
   - sincronizar `HUB_CONTROLE.md`.

## Alternativas consideradas
1. **Não alterar tratamento de exceção de assinatura**
   - Prós: menor alteração de código.
   - Contras: vazamento de exceções de baixo nível, pior DX e governança de erro.

2. **Padronizar em `FoldSecurityError` (decisão adotada)**
   - Prós: contrato consistente para consumidores da API.
   - Contras: pequena mudança comportamental para integrações que dependiam de exceções internas.

## Consequências
- A auditoria do formato `.fold/.mind` fica reproduzível e rastreável.
- O fluxo de erro de assinatura passa a ser semanticamente coerente com a camada de segurança da serialização.
- O processo operacional (ISSUE/EXEC/HUB) permanece alinhado com controles de governança do repositório.
