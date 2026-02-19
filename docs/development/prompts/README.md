# 📁 Portal de Prompts Operacionais

Guia oficial do fluxo **humano → IA** para execução e finalização técnica.

> **Status atual (ADR 0040):** o ciclo de abertura de novas `ISSUE-*` foi concluído para a fase atual.  
> Os arquivos `ISSUE-*` existentes permanecem como histórico e referência.

## 🎯 Objetivo
Garantir que cada execução tenha:
1. artefato técnico de execução (`EXEC-*`),
2. evidências de validação,
3. sincronização dos documentos de controle aplicáveis.

## 🔄 Fluxo oficial (fase atual)
1. **ANALISAR (humano):** valida escopo da demanda em andamento.
2. **EXECUTAR (IA):** implementa e registra evidências técnicas.
3. **FINALIZAR (humano):** revisa evidências e aprova PR.

## ✅ Diretriz de governança

- **Não abrir novas `ISSUE-*` por padrão nesta fase.**
- Usar `ISSUE-*` legadas apenas para consulta histórica.
- Priorizar documentação em `EXEC-*` e nos artefatos de validação.

## ✅ Prompt padrão para ANALISAR (humano)
```markdown
ANÁLISE DA EXECUÇÃO

Checklist:
- [ ] escopo técnico claro
- [ ] riscos e dependências identificados
- [ ] critérios de aceite verificáveis
- [ ] validações obrigatórias definidas

Status:
- [ ] APROVADA para execução
- [ ] REPROVADA com ajustes
```

## 🚀 Prompt padrão para EXECUTAR (IA)
```markdown
Executar demanda aprovada e registrar evidências técnicas.

Passos:
1) Aplicar apenas o escopo definido.
2) Atualizar/criar EXEC correspondente.
3) Rodar validações necessárias.
4) Sincronizar documentos de controle aplicáveis.
5) Commit + PR.
```

## 🔗 Referências
- [Relatórios](./relatorios/README.md)
- [execution_queue.csv](../execution_queue.csv)
- [HUB_CONTROLE.md](../HUB_CONTROLE.md)
- [ADR 0040](../../governance/adr/legado/0040-conclusao-do-ciclo-issue-e-foco-em-execucao.md)
