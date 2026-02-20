---
id: "ISSUE-009"
titulo: "MPJRD v2 — fechamento P0/P1 (C-01, C-02, A-01, A-02, A-03)"
prioridade: "Alta"
area: "Core/Advanced/Layers"
responsavel: "Codex"
criado_em: "2026-02-20"
normas:
  - ABNT NBR ISO/IEC 25010
  - IEEE 830
---
# ISSUE-009: MPJRD v2 — fechamento P0/P1

## Objetivo
Consolidar o fechamento das falhas prioritárias do ciclo MPJRD v2 com foco em consistência de contrato de saída, invariância de batch no STDP, threshold refratário e rastreabilidade de buffers de estado.

## Contexto Técnico
O pacote `pyfolds` possui pontos críticos identificados como P0/P1 em componentes de camada (`layer`), módulos avançados (`inhibition`, `stdp`, `short_term`, `refractory`) e integração com neurônio base (`core/neuron.py`).

Este relatório formaliza a conclusão de 5 falhas (C-01, C-02, A-01, A-02, A-03), centralizando escopo, evidências e critérios de aceite para governança técnica.

## Análise Técnica
### Resumo executivo com escopo (5 falhas)
- **C-01**: contrato de saída da layer com harmonização de campos `u_values`/`u`.
- **C-02**: aplicação de inibição em `apply_inhibition` com comportamento consistente sob diferentes cenários.
- **A-01**: correção do cálculo de `delta_total` no STDP usando agregação `mean(dim=0)` para invariância ao batch.
- **A-02**: rastreabilidade e consistência de buffers de STP no módulo `short_term`.
- **A-03**: consistência de homeostase/refratariedade entre `refractory` e `neuron` via `theta_eff`.

### Tabela de fechamento
| Falha | Arquivo/Função | Correção aplicada | Risco residual |
| --- | --- | --- | --- |
| C-01 | `src/pyfolds/layers/layer.py` (`u_values` / `u`) | Unificação de contrato de saída e compatibilidade explícita entre representações de potencial | Baixo: risco de regressão apenas em consumidores externos sem contrato atualizado |
| C-02 | `src/pyfolds/advanced/inhibition.py` (`apply_inhibition`) | Ajuste da regra de inibição para previsibilidade de efeito e estabilidade de execução | Baixo: depende de cobertura contínua em cenários-limite |
| A-01 | `src/pyfolds/advanced/stdp.py` (`delta_total` com `mean(dim=0)`) | Normalização por batch para remover sensibilidade ao tamanho do lote no update plástico | Muito baixo: risco residual restrito a mudanças futuras na forma dos tensores |
| A-02 | `src/pyfolds/advanced/short_term.py` (buffers STP) | Organização e persistência consistente de buffers de curto prazo com rastreabilidade explícita | Baixo: exige manutenção disciplinada de nomes e ciclo de vida dos buffers |
| A-03 | `src/pyfolds/advanced/refractory.py` + `src/pyfolds/core/neuron.py` (homeostase + `theta_eff`) | Alinhamento do threshold efetivo refratário com dinâmica homeostática para decisão de disparo | Médio-baixo: interação dinâmica ainda pode exigir ajuste fino paramétrico |

### Evidências (caminhos exatos)
- `src/pyfolds/layers/layer.py` (campos `u_values`/`u`).
- `src/pyfolds/advanced/inhibition.py` (`apply_inhibition`).
- `src/pyfolds/advanced/stdp.py` (`delta_total` com `mean(dim=0)`).
- `src/pyfolds/advanced/short_term.py` (buffers STP).
- `src/pyfolds/advanced/refractory.py` e `src/pyfolds/core/neuron.py` (homeostase + `theta_eff`).

## Requisitos Funcionais
- [x] RF-01: registrar escopo executivo com as 5 falhas (C-01, C-02, A-01, A-02, A-03).
- [x] RF-02: mapear falha → arquivo/função → correção → risco residual em tabela única.
- [x] RF-03: explicitar evidências com caminhos exatos dos módulos impactados.
- [x] RF-04: declarar critérios de aceite técnicos para fechamento P0/P1.
- [x] RF-05: incluir seção de validação com comandos de checagem estática e testes.

## Requisitos Não-Funcionais
- [x] RNF-01: Clareza e auditabilidade do relatório para revisão técnica.
- [x] RNF-02: Rastreabilidade cruzada entre falhas, arquivos e critérios de aceite.

## Artefatos Esperados
- Relatório `ISSUE-009-mpjrd-v2-fechamento-p0-p1.md` em `docs/development/prompts/relatorios/`.
- Execução do validador de formato de issue do repositório com status de aprovação.

## Critérios de Aceite
- [x] Contrato de saída de layer definido e consistente para os campos de potencial (`u_values`/`u`).
- [x] Regra de STDP com invariância a batch no cálculo de `delta_total`.
- [x] Consistência de threshold refratário considerando `theta_eff` na integração com homeostase.
- [x] Rastreabilidade explícita de buffers de STP ao longo do ciclo de execução.

## Riscos e Mitigações
- **Risco**: evolução futura quebrar invariantes de batch no STDP.  
  **Mitigação**: manter testes de regressão específicos para variação de tamanho de lote.
- **Risco**: mudanças em refratariedade/homeostase impactarem comportamento emergente.  
  **Mitigação**: validar parâmetros com suíte de testes unitários e cenários comparativos.

## PROMPT:EXECUTAR
```yaml
objetivo: "Consolidar fechamento P0/P1 do MPJRD v2 com rastreabilidade técnica das falhas C-01, C-02, A-01, A-02 e A-03."
issue_id: "ISSUE-009"
prioridade: "Alta"
area: "Core/Advanced/Layers"
```

## Rastreabilidade (IEEE 830)
| Requisito | Evidência |
| --- | --- |
| RF-01 | Seção "Resumo executivo com escopo (5 falhas)" |
| RF-02 | Tabela "Falha → Arquivo/Função → Correção aplicada → Risco residual" |
| RF-03 | Seção "Evidências (caminhos exatos)" |
| RF-04 | Seção "Critérios de Aceite" |
| RF-05 | Seção "Validação (execução em modo Code)" |

## Validação (execução em modo Code)
> Comandos documentados para execução operacional em ambiente de desenvolvimento com escrita/habilitação completa.

### Checagem estática
```bash
python -m compileall src
ruff check src tests
mypy src/pyfolds
```

### Testes
```bash
pytest -q
pytest -q tests/unit/advanced tests/unit/core tests/unit/layers
```
