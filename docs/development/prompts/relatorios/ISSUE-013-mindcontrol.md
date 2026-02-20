# ISSUE-013 — MindControl: Motor de Injeção e Mutação em Tempo Real

## Objetivo
Implementar um motor externo em malha fechada para mutar hiperparâmetros do neurônio durante execução contínua, aproveitando eventos de telemetria (`commit`) sem bloquear o `forward`.

## Escopo entregue
- Novo módulo `pyfolds.monitoring.mindcontrol` com:
  - `MindControl` para registro de neurônios e injeção assíncrona.
  - `MutationCommand` para comandos de mutação.
  - `MindControlSink` para integração com telemetria via sink.
- `MPJRDNeuron` atualizado com fila lock-free (`SimpleQueue`) para aplicar mutações no início de `forward/apply_plasticity`.
- Atualização segura de tensores (`theta`, `r_hat`) com `torch.no_grad()` e `copy_()`.
- Atualização dinâmica de `MPJRDConfig` (dataclass imutável) via `with_runtime_update`.
- Teste de integração para mutação abrupta de `activity_threshold` em runtime sem NaN.

## Critérios de segurança atendidos
- Thread/graph safety: mutações tensoriais in-place fora do grafo.
- Latência baixa: enfileiramento sem lock no caminho crítico de execução.
- Acoplamento por telemetria: decisão disparada no evento `commit`.

## Riscos e mitigação
- **Risco:** mutações inválidas em campos da config.
  - **Mitigação:** `with_runtime_update` reutiliza validações de `MPJRDConfig` e só aplica campos reconhecidos.
- **Risco:** inconsistência de referência de config nos submódulos.
  - **Mitigação:** `_refresh_config_references` propaga a nova configuração para módulos com atributo `cfg`.


## Governança (ADR)
- ADR criada: `docs/governance/adr/ADR-044-mindcontrol-cerebro-externo-e-mutacao-runtime-segura.md`.
