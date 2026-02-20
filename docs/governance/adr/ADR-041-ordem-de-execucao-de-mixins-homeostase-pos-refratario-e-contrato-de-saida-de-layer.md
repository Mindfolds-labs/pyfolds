# ADR-041 — Ordem de execução de mixins, homeostase pós-refratário e contrato de saída de layer

## Status
Aceito

## Data
2026-02-20

## Contexto
Incidentes recentes em produção e regressões funcionais (C-01, C-02, A-01, A-02, A-03) evidenciaram inconsistências de contrato e de ordem de processamento entre `MPJRDLayer` e mixins avançados.

Principais efeitos observados:
- divergência de decisão de disparo quando diferentes caminhos consumiam `theta` vs `theta_eff`;
- atualização homeostática baseada em spike intermediário (pré-máscara refratária), causando drift de controle;
- inconsistência de interface de saída (`u` vs `u_values`, `theta` vs `thetas`) entre consumidores;
- variação de escala em atualização STDP em cenários com batch heterogêneo;
- risco de perda de rastreabilidade de estado STP por reatribuições que quebram buffers registrados.

Em produção, isso se refletiu em comportamento não determinístico entre execuções equivalentes, maior dificuldade de depuração em pipelines com inibição/refratário e aumento de custo operacional para manter compatibilidade retroativa.

## Decisão
Fica definido o seguinte contrato arquitetural e de execução:

1. **Contrato de saída de camada (`MPJRDLayer.forward`)**
   - `MPJRDLayer.forward` deve sempre expor potencial somático por batch em `u_values`.
   - Deve manter compatibilidade legada com alias `u` apontando para o mesmo tensor lógico.

2. **Normalização STDP em batch**
   - Atualizações STDP em modo batch devem ser normalizadas por média no eixo do batch (`mean(dim=0)`), evitando dependência espúria do tamanho do batch.

3. **Homeostase pós-refratário**
   - Em neurônios com refratário, a homeostase deve consumir o spike final pós-máscara refratária (resultado efetivo de disparo), não o spike intermediário.

4. **Referência de threshold para disparo**
   - `theta_eff` é a referência compartilhada para decisão de disparo nos caminhos que combinam camada base + mixins.
   - `theta` permanece como estado/base de threshold, não como critério final quando houver ajustes dinâmicos (ex.: refratário relativo).

5. **Persistência de estado STP**
   - Estados de STP devem permanecer como buffers registrados (`register_buffer`), com atualização in-place.
   - É vedada reatribuição que substitua o objeto do buffer e rompa registro, serialização e/device transfer.

## Consequências
### Positivas
- Contrato de saída estável e explícito para consumidores de camada.
- Menor ambiguidade na integração entre mixins (inibição/refratário/homeostase/plasticidade).
- Melhor previsibilidade numérica em treinamento batch.
- Maior robustez de serialização/checkpoint para estados STP.

### Negativas
- Consumidores antigos que dependem de campos legados podem exigir adaptação gradual.
- Pode haver mudança de métricas históricas devido ao alinhamento da semântica de disparo/homeostase.

## Riscos
- Regressão silenciosa em integrações externas que assumem apenas `theta` ou apenas `u`.
- Diferenças em curvas de convergência após a normalização STDP por `mean(dim=0)`.
- Falsa sensação de compatibilidade se aliases legados forem removidos sem depreciação formal.

## Rollback
Em caso de regressão operacional severa:
1. Reativar temporariamente compatibilidade estrita com campos legados (`u`, `thetas`) via feature flag.
2. Manter `theta_eff` disponível, mas restaurar caminho anterior apenas nos consumidores afetados.
3. Publicar hotfix com janela curta e telemetria para comparação A/B.
4. Reabrir revisão desta ADR com evidências de impacto e plano de correção incremental.

## Plano de migração e compatibilidade
1. **Saída de potencial somático**
   - Novo contrato primário: `u_values`.
   - Compatibilidade: manter `u` como alias durante janela de transição.
   - Ação para consumidores: migrar leituras para `u_values`.

2. **Threshold de disparo**
   - Novo contrato de decisão: `theta_eff`.
   - Compatibilidade: aceitar `theta` como fallback quando `theta_eff` não estiver presente.
   - Compatibilidade ampliada: `thetas` tratado como legado somente leitura.
   - Ação para consumidores: priorizar `theta_eff`; usar `theta` apenas para observabilidade de estado base.

3. **Depreciação formal**
   - Documentar em changelog a depreciação progressiva de `thetas` e de uso exclusivo de `u`.
   - Definir data de remoção após ciclo mínimo de versões compatíveis.

## Links cruzados
### Issue consolidada
- [ISSUE-003 — Inconsistência matemática entre ONLINE e BATCH no termo pré-sináptico](../quality/issues/ISSUE-003-batch-hebbian-pre-rate-normalization.md)

### Módulos afetados
- [`src/pyfolds/layers/layer.py`](../../../src/pyfolds/layers/layer.py)
- [`src/pyfolds/advanced/refractory.py`](../../../src/pyfolds/advanced/refractory.py)
- [`src/pyfolds/advanced/inhibition.py`](../../../src/pyfolds/advanced/inhibition.py)
- [`src/pyfolds/core/neuron.py`](../../../src/pyfolds/core/neuron.py)
