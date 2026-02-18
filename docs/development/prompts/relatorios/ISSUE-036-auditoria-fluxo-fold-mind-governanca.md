# ISSUE-036: auditoria do fluxo fold-mind e governanca operacional

## Metadados
- id: ISSUE-036
- tipo: GOVERNANCE
- titulo: Auditoria do formato `.fold/.mind`, validação da lógica e consolidação de governança (ADR + EXEC + HUB)
- criado_em: 2026-02-18
- owner: Codex
- status: DONE

## 1. Objetivo
Validar tecnicamente o modelo de arquivo `.fold/.mind`, checar possíveis erros lógicos na serialização/desserialização e consolidar um pacote completo de governança com relatório, execução e ADR.

## 2. Escopo

### 2.1 Inclui:
- Revisão do módulo `src/pyfolds/serialization/foldio.py`.
- Execução de testes focados de serialização, corrupção e leitura concorrente.
- Ajuste corretivo de tratamento de erro em validação de assinatura digital.
- Criação de relatório ISSUE-036 e execução EXEC-036 com prompt pronto para Codex.
- Criação de ADR-038 para formalizar decisões do ciclo.
- Atualização de `execution_queue.csv`, sincronização de `HUB_CONTROLE.md` e validações de governança.

### 2.2 Exclui:
- Reescrita integral do formato binário `.fold/.mind`.
- Mudanças de API pública fora do fluxo de segurança/auditoria do `foldio`.

## 3. Artefatos Gerados
- `src/pyfolds/serialization/foldio.py`
- `tests/unit/serialization/test_foldio.py`
- `docs/governance/adr/ADR-038-auditoria-fold-mind-governanca-execucao.md`
- `docs/governance/adr/INDEX.md`
- `docs/development/prompts/relatorios/ISSUE-036-auditoria-fluxo-fold-mind-governanca.md`
- `docs/development/prompts/execucoes/EXEC-036-auditoria-fluxo-fold-mind-governanca.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

## 4. Riscos
- Risco: indisponibilidade da dependência `cryptography` pode limitar cobertura de assinatura em ambiente mínimo.
  Mitigação: testes de assinatura usam `pytest.importorskip`; fluxos principais permanecem operacionais sem a dependência.
- Risco: ajustes de exceção em assinatura alterarem tipo de erro observado por integrações externas.
  Mitigação: padronização em `FoldSecurityError` para manter contrato semântico de segurança.

## 5. Critérios de Aceite
- ISSUE em conformidade com `tools/validate_issue_format.py`.
- EXEC com passos executados e validações registradas.
- Registro no `execution_queue.csv`.
- `python tools/sync_hub.py` executado.
- `HUB_CONTROLE.md` alterado no mesmo commit.
- Testes focados de serialização/corrupção executados sem regressão.

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-036"
tipo: "GOVERNANCE"
titulo: "Auditar fluxo fold/mind, validar lógica e consolidar ADR+EXEC+HUB"

passos_obrigatorios:
  - "Revisar src/pyfolds/serialization/foldio.py e validar header/chunks/integridade"
  - "Executar testes focados: foldio + corrupção + concorrência"
  - "Padronizar erros de assinatura para FoldSecurityError"
  - "Criar ISSUE-036 e EXEC-036 com trilha completa"
  - "Criar ADR-038 e atualizar docs/governance/adr/INDEX.md"
  - "Registrar ISSUE-036 em docs/development/execution_queue.csv"
  - "Rodar python tools/sync_hub.py"
  - "Garantir alteração de docs/development/HUB_CONTROLE.md no mesmo commit"

validacao:
  - "PYTHONPATH=src pytest -q tests/unit/serialization/test_foldio.py tests/test_fold_corruption.py tests/test_corruption_detection.py tests/test_concurrent_reads.py"
  - "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-036-auditoria-fluxo-fold-mind-governanca.md"
  - "python tools/sync_hub.py --check"
  - "python tools/check_issue_links.py docs/development/prompts/relatorios"
```

---

## Apêndice A — Relatório técnico (modelo ISSUE-003)

### A.1 Sumário executivo
- O formato `.fold/.mind` segue robusto para integridade (CRC32C + SHA256 + ECC opcional) e segurança de desserialização (`weights_only=True`).
- Não foram observadas falhas críticas de lógica no pipeline principal de serialização/desserialização.
- Foi aplicado ajuste corretivo para encapsular falhas de validação de assinatura em `FoldSecurityError`, reduzindo vazamento de exceções de baixo nível.

### A.2 Diagnóstico técnico
- Header/index: validações de magic/header_len/index_off/index_len mantêm barreira contra leitura fora de faixa.
- Chunks: checks de limites (`MAX_CHUNK_SIZE`) e verificação hierárquica (`chunk_hashes`, `manifest_hash`) estão consistentes.
- Segurança: assinatura digital opcional permanece desacoplada (dependência opcional), mantendo compatibilidade em ambientes mínimos.

### A.3 Evidências executadas
- `PYTHONPATH=src pytest -q tests/unit/serialization/test_foldio.py tests/test_fold_corruption.py tests/test_corruption_detection.py tests/test_concurrent_reads.py`
- `PYTHONPATH=src python -m py_compile src/pyfolds/serialization/foldio.py tests/unit/serialization/test_foldio.py`

### A.4 Conclusão
A implementação `.fold/.mind` está tecnicamente adequada para persistência auditável. O ciclo ISSUE-036 formaliza a governança de execução e deixa prompt operacional pronto para reuso por Codex em novas auditorias.
