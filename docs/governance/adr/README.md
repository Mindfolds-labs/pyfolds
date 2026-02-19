# ADRs (Architecture Decision Records)

Este diretório (`docs/governance/adr/`) é a **única fonte oficial** para ADRs do projeto.

## Estrutura

```text
/docs/governance/adr/
├── ADR-*.md                  # ADRs ativos/canônicos
├── INDEX.md                  # índice oficial com status
└── legado/
    ├── 0001-*.md             # documentos migrados de docs/adr/
    ├── 0040-*.md             # documentos migrados de docs/adr/
    └── ADR-*-*.md            # ADRs superseded e históricos
```

> `docs/adr/` foi descontinuado como fonte de ADR. Novos ADRs e manutenção devem ocorrer somente em `docs/governance/adr/`.

## Política ADR

### 1) Numeração
- ADRs canônicos usam o padrão `ADR-XXX-<slug>.md` com `XXX` em 3 dígitos.
- O próximo número é sempre sequencial ao maior ADR canônico ativo.
- Não reutilizar número já publicado.

### 2) Status e ciclo de vida
- **Ativo**: decisão vigente e referência oficial.
- **Legado**: documento histórico migrado, sem efeito normativo atual.
- **Superseded**: ADR substituído por outro ADR canônico mais recente.

### 3) Arquivamento
- Documentos legados ou superseded devem ser movidos para `docs/governance/adr/legado/`.
- ADR superseded deve conter nota no cabeçalho indicando:
  - ADR canônico que o substitui;
  - data da supersessão;
  - motivo resumido.

### 4) Fluxo de revisão
1. Propor ADR em arquivo novo com numeração sequencial.
2. Revisar tecnicamente e registrar status no próprio ADR.
3. Atualizar `INDEX.md` na mesma alteração (árvore, status e link).
4. Em caso de substituição, mover ADR antigo para `legado/` e marcar como superseded.
5. Aprovar via PR com rastreabilidade da decisão.
