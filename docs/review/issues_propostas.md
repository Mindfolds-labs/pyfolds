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
