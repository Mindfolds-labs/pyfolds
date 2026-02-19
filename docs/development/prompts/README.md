# 📁 Portal de Prompts Operacionais

Guia oficial do fluxo **humano → IA** para execução e finalização técnica.

> **Status atual (ADR 0041):** a abertura de novas `ISSUE-*` passa a seguir política de fases
> (ativa, freeze e legado), sem bloqueio absoluto permanente.

## 🎯 Objetivo
Garantir que cada execução tenha:
1. artefato técnico de execução (`EXEC-*`),
2. evidências de validação,
3. sincronização dos documentos de controle aplicáveis.

## 🔄 Fluxo oficial (por fase)
1. **ANALISAR (humano):** valida escopo da demanda em andamento.
2. **EXECUTAR (IA):** implementa e registra evidências técnicas.
3. **FINALIZAR (humano):** revisa evidências e aprova PR.

## ✅ Diretriz de governança

- Abertura de `ISSUE-*` depende da fase vigente no workflow integrado.
- Em **fase ativa**, novas issues são permitidas.
- Em **fase freeze**, somente correções críticas podem gerar nova issue.
- Em **fase legado**, `ISSUE-*` existentes são apenas consulta histórica.
- Priorizar documentação em `EXEC-*` e nos artefatos de validação em todas as fases.

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
- [ADR 0041](../../adr/0041-modelo-de-fases-ciclo-continuo-e-legado.md)
