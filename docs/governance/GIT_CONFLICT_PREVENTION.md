# Prevenção de Conflitos Git (Canônico)

> **Status:** Ativo  
> **Escopo:** Contribuição técnica e governança operacional  
> **Aplica-se a:** todo o repositório, com ênfase em `docs/development/execution_queue.csv` e `docs/development/HUB_CONTROLE.md`

## 1. Objetivo
Definir um padrão único para reduzir conflitos de merge/rebase, preservar rastreabilidade e manter a sincronização entre fila operacional (CSV) e HUB.

## 2. Estratégia de branches

### 2.1 Modelo oficial
- Branch estável: `main`.
- Branches de trabalho: `feature/<tema-curto>`, `fix/<tema-curto>`, `docs/<tema-curto>`, `chore/<tema-curto>`.
- Branches devem ser **curtas e efêmeras** (ideal: 1 escopo lógico por PR).

### 2.2 Regras operacionais
1. Sempre criar branch a partir de `main` atualizada.
2. Evitar branches long-lived para reduzir divergência.
3. Limitar PRs com escopo misto (código + governança + docs) quando não houver necessidade técnica.

## 3. Proteção de branch (`main`)
Configurar as seguintes proteções:
- Bloquear push direto em `main`.
- Exigir pull request para merge.
- Exigir checks obrigatórios de CI (incluindo validação de fila/HUB).
- Exigir branch atualizada antes do merge quando houver conflitos.
- Exigir ao menos 1 aprovação de review.

## 4. Regra de integração (rebase/merge)

### 4.1 Durante desenvolvimento
- Recomendado: `git fetch origin && git rebase origin/main` antes de abrir/atualizar PR.
- Resolver conflitos localmente com foco em:
  - preservar o estado mais recente de governança;
  - manter integridade do CSV (`execution_queue.csv`);
  - manter HUB sincronizado.

### 4.2 No merge da PR
- Estratégia preferencial: **Squash merge** para manter histórico limpo por entrega.
- Só fazer merge após todos os checks obrigatórios passarem.

## 5. Protocolo para conflitos em `execution_queue.csv` e `HUB_CONTROLE.md`

### 5.1 Fonte da verdade
- `docs/development/execution_queue.csv` é a fonte primária.
- `docs/development/HUB_CONTROLE.md` é derivado e deve ser sincronizado por `tools/sync_hub.py`.

### 5.2 Resolução de conflito no CSV
1. Manter ambos os registros quando IDs forem distintos.
2. Em caso de mesmo ID duplicado, consolidar em uma única linha com o estado mais recente.
3. Preservar histórico no campo de artefatos/links quando aplicável.
4. Validar duplicidade de IDs antes de concluir: `python tools/check_queue_governance.py`.

### 5.3 Resolução de conflito no HUB
1. Nunca resolver bloco de fila manualmente quando houver dúvida.
2. Após resolver o CSV, executar `python tools/sync_hub.py`.
3. Confirmar consistência com `python tools/sync_hub.py --check`.
4. Se `execution_queue.csv` mudar, o HUB correspondente deve mudar no mesmo PR.

## 6. Gates obrigatórios pré-merge (governança)
Antes de mergear qualquer PR que toque governança:
1. `python tools/check_queue_governance.py`
2. `python tools/sync_hub.py --check`
3. Garantir que não há edição concorrente de CSV sem atualização do HUB.

## 7. Automação em CI
A CI deve bloquear merge quando:
- houver IDs duplicados em `docs/development/execution_queue.csv`;
- houver alteração de `execution_queue.csv` sem alteração de `HUB_CONTROLE.md` no mesmo PR;
- o HUB estiver fora de sincronização com o CSV.

## 8. Referências
- `CONTRIBUTING.md`
- `docs/development/CONTRIBUTING.md`
- `docs/development/HUB_CONTROLE.md`
- `docs/development/release_process.md`
