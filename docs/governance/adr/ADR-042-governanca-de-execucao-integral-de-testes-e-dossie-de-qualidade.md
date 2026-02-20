# ADR-042 — Governança de execução integral de testes e dossiê de qualidade

## Status
Aceito

## Data
2026-02-20

## Contexto
Foi solicitada uma execução integral da suíte de testes com produção de evidências completas (logs, métricas, issue operacional e relatório final). O repositório já possui fluxo de governança para issues/ADR, mas não havia um artefato canônico consolidando um **dossiê único de execução** para auditoria técnica rápida.

A execução completa da suíte (`pytest tests -v`) produziu falhas regressivas e indicadores de qualidade relevantes para priorização imediata.

## Decisão
A partir desta ADR, toda execução ampla de validação deve seguir o padrão:

1. **Executar suíte completa com rastreabilidade**
   - Registrar logs completos em `outputs/test_logs/`.
   - Exportar JUnit XML para consumo por automações.

2. **Abrir issue operacional vinculada**
   - Registrar falhas e hipóteses com checklist objetivo.
   - Ligar issue aos artefatos de evidência.

3. **Publicar relatório final consolidado**
   - Incluir métricas de aprovação, falhas, skips, duração e pontos lentos.
   - Incluir plano de ação por severidade.

4. **Atualizar governança**
   - Atualizar `execution_queue.csv` com estado da execução.
   - Referenciar esta ADR no relatório e na issue.

## Consequências
### Positivas
- Melhora de auditabilidade e reprodutibilidade de validação.
- Redução de perda de contexto entre execução técnica e decisão de priorização.
- Facilita integração com pipelines de CI/CD e SLOs de qualidade.

### Negativas
- Aumento de custo operacional documental em execuções pontuais.
- Necessidade de disciplina para manter logs e relatórios sincronizados.

## Riscos
- Desalinhamento entre log bruto e resumo executivo.
- Crescimento de volume de artefatos sem curadoria.

## Mitigações
- Padronizar nomenclatura dos logs e incluir sumário no relatório final.
- Validar links e estrutura de issue antes de fechamento.

## Links cruzados
- Issue operacional: `ISSUE-010`.
- Relatório consolidado: `docs/RELATORIO_FINAL_EXECUCAO_TESTES_ISSUE-010.md`.
- Evidências: `outputs/test_logs/`.
