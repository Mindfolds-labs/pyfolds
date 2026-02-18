# ISSUE-036: auditoria fold/mind, integridade lógica e governança operacional

## Metadados
- id: ISSUE-036
- tipo: GOVERNANCE
- titulo: Auditoria do formato `.fold/.mind` com validação técnica, correção de lógica de segurança e prompt operacional para Codex
- criado_em: 2026-02-18
- owner: Codex
- status: DONE

## 1. Objetivo
Executar uma auditoria prática do formato `.fold/.mind`, confirmar integridade do fluxo de serialização/desserialização, ajustar pontos de lógica de segurança quando necessário e entregar um prompt canônico para execução por Codex, com rastreabilidade completa de governança.

## 2. Escopo

### 2.1 Inclui:
- Revisão e ajuste de tratamento de erro na verificação de assinatura digital em `foldio.py`.
- Atualização de testes unitários para refletir semântica de erro de segurança.
- Criação de ADR de decisão arquitetural (ADR-038).
- Geração de EXEC correspondente.
- Atualização de `execution_queue.csv` e sincronização de `HUB_CONTROLE.md`.
- Definição de prompt operacional pronto para Codex (auditoria `.fold/.mind`).

### 2.2 Exclui:
- Reescrita ampla do formato binário `.fold/.mind`.
- Mudanças de API pública além de hardening pontual de erro.
- Alterações de produto fora do escopo de serialização/governança.

## 3. Artefatos Gerados
- `src/pyfolds/serialization/foldio.py`
- `tests/unit/serialization/test_foldio.py`
- `docs/governance/adr/ADR-038-governanca-fold-mind-auditoria-integridade-e-prompt-operacional.md`
- `docs/governance/adr/INDEX.md`
- `docs/development/prompts/relatorios/ISSUE-036-auditoria-fold-mind-integridade-governanca.md`
- `docs/development/prompts/execucoes/EXEC-036-auditoria-fold-mind-integridade-governanca.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

## 4. Riscos
- Risco: divergência entre análise textual e comportamento real em runtime.
  Mitigação: validação obrigatória com `pytest` + `py_compile`.

- Risco: falha de interpretação de erro criptográfico por exceções genéricas.
  Mitigação: normalizar para `FoldSecurityError` no fluxo de verificação.

- Risco: entrega incompleta de governança (sem CSV/HUB sincronizados).
  Mitigação: executar `tools/sync_hub.py` e `tools/sync_hub.py --check`.

## 5. Critérios de Aceite
- ISSUE em conformidade com `tools/validate_issue_format.py`.
- Verificação de assinatura com chave inválida retorna erro de segurança explícito.
- Testes unitários de serialização executados com sucesso.
- ADR-038 criado e indexado.
- EXEC-036 criado.
- `execution_queue.csv` atualizado para ISSUE-036.
- `HUB_CONTROLE.md` sincronizado no mesmo ciclo.

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-036"
tipo: "GOVERNANCE"
titulo: "Auditar formato .fold/.mind, validar integridade e consolidar governança"

passos_obrigatorios:
  - "Revisar src/pyfolds/serialization/foldio.py e identificar possíveis falhas de lógica em validação de assinatura e integridade"
  - "Ajustar comportamento para erro de segurança explícito quando verificação criptográfica falhar"
  - "Atualizar/adequar testes em tests/unit/serialization/test_foldio.py"
  - "Executar: PYTHONPATH=src pytest -q tests/unit/serialization/test_foldio.py"
  - "Executar: PYTHONPATH=src python -m py_compile src/pyfolds/serialization/foldio.py tests/unit/serialization/test_foldio.py"
  - "Criar ADR-038 com decisão de governança do ritual de auditoria .fold/.mind"
  - "Criar EXEC-036 com evidências"
  - "Atualizar docs/development/execution_queue.csv com ISSUE-036"
  - "Rodar python tools/sync_hub.py"
  - "Garantir alteração de docs/development/HUB_CONTROLE.md no mesmo commit"

validacao:
  - "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-036-auditoria-fold-mind-integridade-governanca.md"
  - "python tools/sync_hub.py --check"
  - "python tools/check_issue_links.py docs/development/prompts/relatorios"
```

---

## Apêndice A — Prompt pronto para uso direto no Codex
```text
Objetivo: auditar e validar o formato .fold/.mind do pyfolds com evidência executável.

Passos:
1) Revisar src/pyfolds/serialization/foldio.py para confirmar:
   - validações de header/index,
   - integridade CRC32C/SHA256 por chunk,
   - segurança de torch.load(weights_only=True),
   - comportamento de assinatura digital (quando habilitada).
2) Corrigir lógica que retorne exceção genérica em caminho de segurança; padronizar erro para FoldSecurityError.
3) Executar:
   - PYTHONPATH=src pytest -q tests/unit/serialization/test_foldio.py
   - PYTHONPATH=src python -m py_compile src/pyfolds/serialization/foldio.py tests/unit/serialization/test_foldio.py
4) Produzir resumo com:
   - pontos fortes atuais do formato fold/mind,
   - riscos e recomendações,
   - confirmação de ausência/presença de erro lógico encontrado.
5) Atualizar governança da execução:
   - ISSUE/EXEC correspondentes,
   - execution_queue.csv,
   - HUB_CONTROLE.md sincronizado.

Entrega: commit único com código + testes + documentação de governança.
```
