# Issues Propostas (Revisão Técnico-Científica)

## Issue 1 — Garantir localidade sináptica na atualização dendrítica
**Tipo:** bug matemático/algorítmico  
**Severidade:** alta

**Problema:** atualização passada para cada sinapse com vetor completo de `pre_rate`, reduzindo localidade da regra de Hebb/three-factor.

**Critério de aceite:**
- Cada sinapse `i` recebe apenas `pre_rate[i]`.
- Teste unitário falha no comportamento antigo e passa no novo.

## Issue 2 — Corrigir assinaturas lazy de eventos de telemetria
**Tipo:** bug de sintaxe/import  
**Severidade:** crítica

**Problema:** assinatura inválida em funções lazy de eventos bloqueava import do pacote.

**Critério de aceite:**
- `import pyfolds` executa sem `SyntaxError`.
- Teste de smoke de import no pipeline.

## Issue 3 — Formalizar estados STP (`u`, `R`) e responsabilidade de módulo
**Tipo:** dívida técnica de arquitetura  
**Severidade:** média

**Problema:** classes core e advanced compartilham semântica de STP sem fronteira explícita.

**Critério de aceite:**
- Documento de arquitetura definindo ownership de `u`/`R`.
- Testes cobrindo leitura de estado e evolução temporal.


## Issue 4 — Revisar normalização de `pre_rate` no modo BATCH
**Tipo:** ajuste matemático/modelagem
**Severidade:** média

**Problema:** em `apply_plasticity`, `pre_rate` é dividido por `n_active`, transformando taxa absoluta em distribuição relativa e reduzindo a magnitude hebbiana conforme densidade de entrada aumenta.

**Critério de aceite:**
- Experimento controlado (entradas esparsas vs densas) mostrando impacto em convergência de `N_mean` e `spike_rate`.
- Se mantida a regra atual, justificar explicitamente no guia teórico.

## Issue 5 — Validar STDP contra curva canônica Δw(Δt)
**Tipo:** validação científica
**Severidade:** média

**Problema:** implementação atual do mixin STDP é pós-condicionada por spike do passo corrente; pode divergir da formulação pair-based clássica.

**Critério de aceite:**
- Teste dedicado para Δt>0 e Δt<0.
- Documentação explícita do regime adotado (pair-based vs trace-gated).

## Issue 6 — Estabilizar infraestrutura de logging em testes
**Tipo:** confiabilidade de experimento
**Severidade:** média

**Problema:** suíte atual reporta falhas em singleton, TRACE, file handler e module levels.

**Critério de aceite:**
- Bloco `TestLogging` passa de forma determinística em execução limpa.
- Sem regressão nos níveis de log dos módulos.
