# Relatório Técnico de Testes e Benchmark

## Escopo e objetivo
Este documento consolida a execução dos testes do projeto, os gargalos encontrados, e sugestões de melhoria com foco em robustez matemática, coerência teórico-implementacional e estabilidade de treinamento.

## Ambiente de execução
- Python 3.12.12
- `pytest 9.0.2`
- `torch 2.10.0+cu128`
- Restrições de rede/proxy impediram instalação de dependências opcionais (`psutil`, `torchvision`) durante esta sessão.

## Correções objetivas aplicadas antes da validação
1. **Correção de assinatura inválida em funções lazy de telemetria**
   - Arquivo: `src/pyfolds/telemetry/events.py`
   - Problema: `SyntaxError: parameter without a default follows parameter with a default`.
   - Ação: reordenação de parâmetros (`payload_fn` antes de `neuron_id`) para respeitar regra sintática do Python.
2. **Export das funções lazy no módulo de telemetria**
   - Arquivo: `src/pyfolds/telemetry/__init__.py`
   - Problema: `ImportError` ao importar `forward_event_lazy`, `commit_event_lazy`, `sleep_event_lazy`.
   - Ação: inclusão nos imports e `__all__`.

## Benchmark dos testes executados

| Comando | Resultado | Tempo | Observação |
|---|---:|---:|---|
| `pytest tests/ -v --maxfail=1 --durations=10` (antes da correção) | falha | 3.577s | bloqueado por `SyntaxError` em `telemetry/events.py` |
| `pytest tests/ -v --durations=10` (após correção de sintaxe) | falha | 4.318s | coleta interrompida por ausência de `psutil` em `tests/performance/test_memory_usage.py` |
| `pytest tests/ -v --ignore=tests/performance/test_memory_usage.py --durations=10` | falha parcial | 4.393s | suíte executada; **69 passed, 20 failed, 30 errors, 3 skipped** |
| `python -m compileall -q src tests` | sucesso | ~0.50s | sem erros de sintaxe após correções |
| `python test_install.py` | falha | 20.289s | bloqueado por proxy ao instalar `torch` no venv de teste |

## Resultado consolidado (estado atual)
- O projeto **não está em estado verde** na suíte de testes.
- Principais classes de falha observadas:
  1. Dependência ausente no ambiente: `psutil` (teste de performance).
  2. Erros de configuração/imutabilidade (`FrozenInstanceError`) em componentes core e advanced.
  3. Divergências entre API esperada pelos testes e implementação atual (ex.: tipos/enums/logging/telemetry).

## Revisão técnica (engenharia + coerência matemática)

### 1) Estabilidade, domínio e convergência
- O caminho de `forward` do neurônio usa operações com domínio controlado:
  - `spikes = (u >= theta).float()` (limiar explícito);
  - `R` saturado em `[-1, 1]` para modos externos/endógenos;
  - `post_rate` e `pre_rate` com `clamp([0,1])` em `apply_plasticity`.
- Esses limites são consistentes com práticas para evitar explosão numérica em regras tipo Hebb/neuromoduladas.
- **Risco prático detectado:** falhas massivas de teste em componentes de estado sugerem inconsistência de ciclo de atualização (estado mutável vs classes congeladas), o que pode invalidar a dinâmica de aprendizado na prática.

### 2) Coerência teórica vs implementação
- A pipeline implementa: integração dendrítica → competição WTA → disparo somático → homeostase → neuromodulação → plasticidade.
- Esse fluxo é coerente com arquiteturas bioinspiradas e com estratégias de estabilização por homeostase.
- Porém, os erros de execução impedem validar experimentalmente se a convergência esperada é atingida nas configurações de treino.

### 3) Parâmetros críticos e impacto
- `activity_threshold`: define máscaras de pré-atividade e altera diretamente o ganho efetivo de atualização sináptica.
- `target_spike_rate` e `theta`: governam regime de disparo e estabilidade homeostática.
- `n_max` e saturação (`N == n_max`): impactam capacidade plástica residual.
- `neuromod_mode` (`external/capacity/surprise`): muda o sinal de reforço e pode inverter tendência de consolidação/poda.

## Issues explicativas (priorização sugerida)

### Issue 1 — Dependências de teste incompletas
**Sintoma:** erro de coleta por `ModuleNotFoundError: psutil`.
**Impacto:** impede validação automatizada completa.
**Ação proposta:** adicionar `psutil` como dependência de dev/performance ou condicionar import com `pytest.importorskip`.

### Issue 2 — Conflito de imutabilidade no core
**Sintoma:** múltiplos `FrozenInstanceError: cannot assign to field '_torch'`.
**Impacto:** falha estrutural de inicialização/estado; compromete treino e atualização de parâmetros.
**Ação proposta:** revisar classes `dataclass(frozen=True)` que precisam mutar buffers/estado interno durante inicialização e aprendizado.

### Issue 3 — Divergência de API pública vs testes
**Sintoma:** falhas em `LearningMode`, `ConnectionType`, `ModeConfig`, logging e telemetry.
**Impacto:** interface inconsistente, risco de regressão e quebra de integração.
**Ação proposta:** alinhar contrato público documentado com testes (ou atualizar testes se contrato mudou por decisão arquitetural formal).

## Melhorias possíveis (fundamentadas)
1. **Definir invariantes matemáticos testáveis** (monotonicidade/saturação/faixas de domínio) para cada etapa de plasticidade.
2. **Adicionar testes de propriedade** (property-based) para garantir `R ∈ [-1,1]`, `pre_rate/post_rate ∈ [0,1]`, limites de saturação e ausência de NaNs.
3. **Segregar testes por perfil** (`unit`, `integration`, `performance`) com skip condicional por dependência, evitando falha de coleta global.
4. **Criar benchmark reprodutível** com ambiente fixo (arquivo de lock + CI matrix CPU/GPU) e métricas mínimas: tempo por forward, throughput batch e uso de memória.
5. **Formalizar critérios de convergência** (ex.: erro estacionário de `r_hat` vs `target_spike_rate`) para avaliar estabilidade de aprendizado.

## Referências científicas úteis para calibrar validação
- Turrigiano, G. (2012). *Homeostatic synaptic plasticity: local and global mechanisms for stabilizing neuronal function.*
- Zenke, F., Gerstner, W. (2017). *Hebbian plasticity requires compensatory processes on multiple timescales.*
- Gerstner, W., Kistler, W. et al. *Neuronal Dynamics* (Cambridge University Press).

## Conclusão executiva
- A execução revelou problemas estruturais que impedem afirmar ausência de erro no sistema neste estado.
- Foi aplicada uma correção objetiva e segura para destravar parsing/import da telemetria.
- Recomenda-se tratar as issues priorizadas acima antes de qualquer análise de convergência em treino de longo horizonte.
