# ADR 0040: Conclusão do ciclo de ISSUE e foco em execução

- **Status:** Aceito
- **Data:** 2026-02-19
- **Decisores:** Maintainers PyFOLDS
- **Relacionados:** [ADR 0001](./0001-import-contract-and-release-readiness.md)

## Contexto

O diretório `docs/development/prompts/` acumulou um histórico extenso de artefatos `ISSUE-*` e `EXEC-*`.
Para o estágio atual do projeto, as demandas de governança associadas às `ISSUE-*` já foram executadas
ou serão consolidadas por um fluxo único de finalização.

Isso gerou ruído operacional: novas tarefas passavam a repetir etapas de criação de ISSUE sem ganho real
para o objetivo imediato de entrega.

## Decisão

1. **Encerrar o ciclo ativo de criação de novas `ISSUE-*`** no fluxo de prompts de desenvolvimento.
2. **Manter os arquivos `ISSUE-*` existentes como histórico**, sem remoção automática.
3. **Focar o fluxo operacional em execução/finalização**, privilegiando `EXEC-*`, evidências e validação.
4. Atualizar os READMEs de `docs/development/prompts/` para refletir este estado.

## Consequências

### Positivas

- Menos overhead documental para tarefas de manutenção.
- Fluxo mais simples para execução técnica e fechamento.
- Preservação de rastreabilidade histórica sem perda de contexto.

### Negativas

- Reduz a granularidade formal de abertura de novos itens por ISSUE.
- Exige disciplina para documentar escopo diretamente nos artefatos de execução.

## Plano de implementação

- Atualizar `docs/development/prompts/README.md` com estado “somente execução/finalização”.
- Atualizar `docs/development/prompts/relatorios/README.md` com status de arquivo histórico.
- Não remover arquivos legados nesta mudança.

## Critérios de aceite

- [x] ADR 0040 criado em `docs/adr/`.
- [x] Referências em `docs/development/prompts/*` alinhadas com a decisão.
- [x] Nenhuma exclusão destrutiva de histórico nesta etapa.
