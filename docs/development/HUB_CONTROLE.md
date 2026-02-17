- ⏳ Planejada → `#e2e3e5` (fundo) | `#6c757d` (borda esquerda)
- ❌ Bloqueada → `#f8d7da` (fundo) | `#dc3545` (borda esquerda)

## 5. Fluxo Padrão para Novas Issues
1. Registrar issue em `execution_queue.csv` e sincronizar a tabela acima com `python tools/sync_hub.py`.
2. Verificar se há ADR aplicável.
3. Criar próximo ADR sequencial (`ADR-XXX-*`) quando necessário.
4. Executar mudanças em branch dedicada.
5. Confirmar link de relatório no card da issue e atualizar este HUB e os índices de governança.

## 6. Checklist de Fechamento
- [ ] Links internos validados.
- [ ] Índices atualizados (`docs/index.md`, `docs/README.md`, `docs/governance/adr/INDEX.md` quando aplicável).
- [ ] Rastreabilidade de artefatos atualizada na tabela.
- [ ] Conformidade com diretrizes IEEE/ISO revisada.

## 7. Referências
- ISO/IEC 12207 — Software Life Cycle Processes.
- IEEE 828 — Software Configuration Management Plans.
- IEEE 730 — Software Quality Assurance.

## 8. Como atualizar a fila manualmente

```bash
python tools/sync_hub.py
python tools/sync_hub.py --check
```

> Nota GitHub Actions: para o workflow de sincronização abrir PR automaticamente, habilite
> **Settings > Actions > General > Workflow permissions > Allow GitHub Actions to create and approve pull requests**.


### 4.23 ISSUE-023
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-023** — Auditoria corretiva de estabilidade runtime e consistência cross-módulo  
*Código / Testes / Governança*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-17

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-023-auditoria-corretiva-estabilidade-runtime.md)
📦 [Ver execução](./prompts/execucoes/EXEC-023-auditoria-corretiva-estabilidade-runtime.md)

</div>
