# ISSUE-024: revisao-estetica-hub-controle

## Metadados
- id: ISSUE-024
- tipo: DOCS
- titulo: Revisão estética do HUB_CONTROLE com cards sincronizados por CSV
- criado_em: 2026-02-17
- owner: Codex
- status: DONE

## 1. Objetivo
Evoluir o `HUB_CONTROLE.md` para um layout visual mais legível e confiável, com geração automática da seção de cards de atividades a partir de `execution_queue.csv`, eliminando divergência manual entre tabela e detalhamento.

## 2. Escopo

### 2.1 Inclui:
- Atualização de `tools/sync_hub.py` para gerar tabela resumida e cards detalhados com estilos por status.
- Geração automática de links para relatório e execução usando `ID` e slug de tema (com fallback e descoberta de arquivo existente).
- Ajuste da seção de workflow/sincronização do HUB para refletir que os cards também são sincronizados.
- Registro da issue no CSV e sincronização do HUB no mesmo commit.

### 2.2 Exclui:
- Alterações no esquema/colunas de `execution_queue.csv`.
- Expansão de governança fora do bloco da fila/cards e instruções de sincronização.
- Mudanças de produto fora da documentação e script de sincronização.

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-024-revisao-estetica-hub-controle.md`
- `docs/development/prompts/execucoes/EXEC-024-revisao-estetica-hub-controle.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`
- `tools/sync_hub.py`

## 4. Riscos
- Risco: GitHub pode renderizar callouts de forma diferente em ambientes fora do padrão.
  Mitigação: uso de Markdown/CommonMark + fallback textual sem depender de HTML/CSS inline.
- Risco: IDs históricos duplicados podem apontar para artefatos ambíguos.
  Mitigação: descoberta determinística por prefixo + fallback previsível baseado em slug.

## 5. Critérios de Aceite
- ISSUE em conformidade com `tools/validate_issue_format.py`
- EXEC correspondente criada com passos técnicos e validações
- Linha da ISSUE-024 adicionada ao `execution_queue.csv`
- `python tools/sync_hub.py` executado e `HUB_CONTROLE.md` atualizado no mesmo commit
- Cards gerados automaticamente com diferenciação visual por status e links funcionais

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-024"
tipo: "DOCS"
titulo: "Revisão estética do HUB_CONTROLE com cards sincronizados por CSV"

passos_obrigatorios:
  - "Atualizar tools/sync_hub.py para gerar tabela e cards"
  - "Criar ISSUE-024-revisao-estetica-hub-controle.md"
  - "Criar EXEC-024-revisao-estetica-hub-controle.md"
  - "Registrar ISSUE no execution_queue.csv"
  - "Rodar python tools/sync_hub.py"
  - "Validar python tools/sync_hub.py --check"

validacao:
  - "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-024-revisao-estetica-hub-controle.md"
  - "python tools/sync_hub.py --check"
  - "python tools/check_issue_links.py docs/development/prompts/relatorios"
```
