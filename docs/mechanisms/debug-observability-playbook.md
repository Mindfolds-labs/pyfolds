# Playbook de depuração e observabilidade

## Objetivo do mecanismo
Padronizar coleta de snapshots para diagnóstico rápido.

## Base científica resumida
Instrumentação robusta melhora interpretação de dinâmica neural simulada.

## Tradução computacional adotada
Funções de snapshot retornam tensores clonados para inspeção sem mutação acidental.

## Arquivos do código afetados
- `src/pyfolds/core/neuron.py`

## Flags de ativação/desativação
`audit_mode`, frequência de coleta definida externamente.

## Riscos de implementação
Overhead quando usado em alta frequência.

## Estratégia de teste
Teste unitário cobrindo chaves esperadas dos relatórios.

## Critérios de observabilidade/debug
Checklist: conectividade efetiva, poda, fase, ressonância.
