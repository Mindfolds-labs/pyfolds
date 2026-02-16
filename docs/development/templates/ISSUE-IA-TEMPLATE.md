# ISSUE-[NNN]: [Título Descritivo com Verbo no Infinitivo]

## Metadados

| Campo | Valor |
|-------|-------|
| **Data** | YYYY-MM-DD |
| **Autor** | [Nome ou Papel] |
| **Issue de Origem** | ISSUE-[NNN] ou N/A |
| **Normas de Referência** | [Normas aplicáveis] |

## 1. Objetivo
[Descrever em 2-3 frases o propósito da issue e o resultado esperado.]

## 2. Escopo

### 2.1 Inclui:
- ✅ [Item 1]
- ✅ [Item 2]

### 2.2 Exclui:
- ❌ [Item 1]
- ❌ [Item 2]

## 3. Artefatos Gerados

| Artefato | Localização | Descrição | Formato |
|----------|-------------|-----------|---------|
| [Nome] | [caminho/arquivo.ext] | [descrição] | [.md/.py/.csv] |

## 4. Riscos

| ID | Risco | Probabilidade | Impacto | Mitigação |
|----|-------|---------------|---------|-----------|
| R01 | [descrição] | [Alta/Média/Baixa] | [Alto/Médio/Baixo] | [ação] |

## 5. Critérios de Aceite
- [ ] [Critério verificável 1]
- [ ] [Critério verificável 2]

## 6. PROMPT:EXECUTAR

```yaml
fase: [NOME_DA_FASE]
prioridade: [CRITICA/ALTA/MEDIA/BAIXA]
responsavel: [CODEX/HUMANO/DUPLA]
dependente: [ISSUE-XXX, ISSUE-YYY]

acoes_imediatas:
  - task: "Descrição da ação"
    output: "caminho/do/arquivo/saida.ext"
    prazo: "Xh"
    comando: "comando opcional"

validacao_automatica:
  - tipo: "formato/links/testes"
    ferramenta: "tools/validator.py"
    criterio: "descrição do critério"

pos_execucao:
  - atualizar: "execution_queue.csv"
  - sincronizar: "HUB"
  - notificar: "canais aplicáveis"
```
