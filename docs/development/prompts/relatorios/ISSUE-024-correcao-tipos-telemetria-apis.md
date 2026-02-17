# ISSUE-024 — Relatório estruturado de correções (core/layers/telemetry)

## Resumo executivo
Aplicadas correções de tipagem, consistência de API e padronização de exports públicos em módulos centrais do `pyfolds`, com foco nos itens críticos e altos reportados.

## Escopo corrigido
1. `src/pyfolds/core/neuron_v2.py`
   - Ajuste do tipo de retorno de `forward` para suportar `Tensor | float | str`.
   - Docstring de retorno detalhada para eliminar ambiguidade.
2. `src/pyfolds/core/synapse.py`
   - Inclusão de `__all__` explícito.
   - Expansão de docstring de `W` com lei de Bartol.
3. `src/pyfolds/layers/layer.py`
   - Padronização das chaves de saída da camada para plural semântico:
     - `spike_rates`, `theta_values`, `r_hat_values`.
   - Mantidos aliases legados (`rates`, `thetas`, `r_hats`) para compatibilidade.
4. `src/pyfolds/core/homeostasis.py`
   - Fortalecimento de contrato de `update` com validação de tipo e entradas inválidas (NaN/Inf).
   - Melhoria de docstring e semântica de retorno.
5. `src/pyfolds/telemetry/controller.py`
   - Remoção de magic strings com `TelemetryProfile` (`Enum`).
   - `TelemetryConfig` agora normaliza string para enum em `__post_init__`.
6. `src/pyfolds/telemetry/__init__.py`
   - Exportação explícita de `TelemetryProfile`.
7. `src/pyfolds/core/neuron.py`
   - Compatibilização com `TelemetryProfile` no bootstrap de telemetria.
8. `src/pyfolds/utils/logging.py`
   - Inclusão de `__all__` para API pública explícita.

## Card de execução
- **Card ID:** CARD-024
- **Tipo:** Correção técnica transversal
- **Severidade agregada:** Alta (inclui itens críticos de tipagem/API)
- **Status:** Concluído
- **Risco residual:** Baixo (mantidos aliases e normalização retrocompatível)

## Evidências de validação
- `python -m compileall src`
- `pytest -q`

## Observações
- Não foi introduzido `try/except` em importação de módulos para manter conformidade com diretriz de estilo de engenharia vigente.
