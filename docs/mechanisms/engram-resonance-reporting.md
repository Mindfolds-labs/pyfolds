# Engram Resonance Reporting

## Objetivo
Relatar memória/ressonância para rastrear formação, recuperação e consolidação de engrams.

## Variáveis
- **Entrada:** padrões de consulta, contexto temporal e banco de engrams.
- **Controle:** `experimental_engram_enabled`, `enable_engram_resonance_telemetry`.
- **Saída:** ranking de engrams, estatísticas de memória e sinais de ressonância.

## Fluxo
1. Criar/consultar engrams com base no padrão recebido.
2. Agregar métricas de ressonância e contagem de memória.
3. Expor relatório para análise offline e telemetria.

## Custo computacional
Busca/ordenação dependem do número de engrams ativos; custo cresce com cardinalidade do banco.

## Integração
- `NoeticCore.collect_engram_report` (`src/pyfolds/advanced/noetic_model.py`).
- `EngramBank.search_by_resonance` e `EngramBank.get_stats` (`src/pyfolds/advanced/engram.py`).
- `MPJRDConfig.experimental_engram_enabled` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Experimental`.
- **Justificativa:** dependente de toggles experimentais e sem protocolo final de calibração científica.
