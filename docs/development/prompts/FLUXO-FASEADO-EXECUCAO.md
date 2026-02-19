# Fluxo Faseado de Execução de Prompts

## Objetivo
Definir um fluxo operacional repetível para execução de issues/prompts com checkpoints de validação, rollback controlado e padrão de commit/PR por fase.

## Sequência faseada (Fase 0 a Fase N)

### Fase 0 — Preparação e baseline
**Objetivo:** alinhar escopo, registrar estado inicial e garantir ambiente reproduzível.

**Ações principais:**
1. Ler a issue/prompt e extrair critérios de aceite.
2. Sincronizar branch e registrar baseline técnico.
3. Capturar evidências iniciais (status, testes e contexto).

**Comandos de validação da fase:**
```bash
git status --short
git rev-parse --abbrev-ref HEAD
python -V
```

**Formato de commit esperado (quando aplicável):**
- `chore(phase-0): registrar baseline e escopo da execução`

**Formato de PR esperado por fase (se PR incremental):**
- Título: `phase-0: baseline e preparo da execução`
- Corpo: contexto, escopo, riscos e evidências coletadas.

---

### Fase 1 — Planejamento técnico
**Objetivo:** decompor a entrega em etapas testáveis, com riscos e estratégia de validação.

**Ações principais:**
1. Definir plano em tarefas atômicas.
2. Mapear arquivos-alvo e impactos.
3. Selecionar comandos de teste por escopo.

**Comandos de validação da fase:**
```bash
rg -n "TODO|FIXME" docs/development/prompts || true
```

**Formato de commit esperado (quando aplicável):**
- `docs(phase-1): registrar plano técnico e estratégia de validação`

**Formato de PR esperado por fase (se PR incremental):**
- Título: `phase-1: planejamento técnico da execução`
- Corpo: plano por tarefas, matriz de risco, definição de pronto.

---

### Fase 2 — Implementação incremental
**Objetivo:** aplicar mudanças mínimas e verificáveis por incremento.

**Ações principais:**
1. Implementar alterações por bloco lógico.
2. Evitar mudanças não relacionadas.
3. Garantir rastreabilidade entre mudança e requisito.

**Comandos de validação da fase:**
```bash
git diff -- docs/development
```

**Formato de commit esperado (obrigatório):**
- `type(phase-2): resumo curto da alteração`
- Tipos recomendados: `docs`, `fix`, `refactor`, `test`, `chore`.

**Formato de PR esperado por fase (se PR incremental):**
- Título: `phase-2: implementação incremental`
- Corpo: o que mudou, por quê, impactos e limitações.

---

### Fase 3 — Validação integrada
**Objetivo:** comprovar consistência funcional e documental da entrega.

**Ações principais:**
1. Executar testes/checks aplicáveis ao escopo.
2. Registrar resultados e falhas.
3. Corrigir desvios antes de seguir.

**Comandos de validação da fase:**
```bash
pytest -q || true
python -m compileall src || true
```

> Use `|| true` apenas quando o objetivo for coletar diagnóstico sem interromper a trilha de investigação. Remova em pipelines obrigatórios.

**Formato de commit esperado (quando aplicável):**
- `test(phase-3): registrar validações e ajustes pós-check`

**Formato de PR esperado por fase (se PR incremental):**
- Título: `phase-3: validação integrada`
- Corpo: comandos, resultado (pass/fail), evidências e pendências.

---

### Fase 4 — Consolidação e governança
**Objetivo:** finalizar entrega com histórico limpo e PR auditável.

**Ações principais:**
1. Revisar diffs finais.
2. Garantir mensagem de commit e corpo de PR no padrão.
3. Atualizar referências cruzadas de documentação.

**Comandos de validação da fase:**
```bash
git status --short
rg -n "FLUXO-FASEADO-EXECUCAO" docs/development/prompts.md
```

**Formato de commit esperado (obrigatório):**
- `docs(phase-4): consolidar fluxo e referências de descoberta`

**Formato de PR esperado por fase (PR final):**
- Título: `phase-4: consolidação final da execução`
- Corpo mínimo:
  - Contexto e objetivo.
  - Lista de mudanças por fase.
  - Comandos executados com status.
  - Política de rollback aplicada.
  - Riscos remanescentes.

---

### Fase N — Repetição controlada por incremento
Repita o ciclo `Fase 1 → Fase 4` para cada novo incremento até cumprir todos os critérios de aceite.

**Regra de saída da Fase N:**
- Todos os critérios de aceite atendidos.
- Nenhuma pendência crítica aberta.
- Evidências de validação anexadas.

## Política de rollback

### Princípios
1. **Rollback mínimo:** reverter apenas o conjunto de commits da fase problemática.
2. **Rollback rastreável:** sempre registrar causa, impacto e comando usado.
3. **Rollback seguro:** nunca reescrever histórico de branch compartilhada sem alinhamento prévio.

### Estratégia recomendada
- Falha antes do merge: `git revert <commit>` (preferível para manter trilha auditável).
- Falha local ainda não publicada: `git reset --hard <hash-seguro>` (uso restrito).
- Falha parcial de fase: abrir commit corretivo com prefixo:
  - `revert(phase-X): desfazer alteração por <motivo>`
  - `fix(phase-X): correção pós-rollback`

### Checklist pós-rollback
```bash
git status --short
git log --oneline -n 10
pytest -q || true
```

## Formato de commit/PR esperado por fase (resumo rápido)

| Fase | Commit (padrão) | PR (título base) |
|---|---|---|
| 0 | `chore(phase-0): ...` | `phase-0: baseline e preparo da execução` |
| 1 | `docs(phase-1): ...` | `phase-1: planejamento técnico da execução` |
| 2 | `type(phase-2): ...` | `phase-2: implementação incremental` |
| 3 | `test(phase-3): ...` | `phase-3: validação integrada` |
| 4 | `docs(phase-4): ...` | `phase-4: consolidação final da execução` |
| N | repetir 1→4 | repetir 1→4 |

## Próximo comando
> Execute exatamente o comando abaixo para iniciar uma nova execução de forma idempotente (sem efeitos colaterais se já estiver em estado limpo):

```bash
git status --short && git rev-parse --abbrev-ref HEAD
```
