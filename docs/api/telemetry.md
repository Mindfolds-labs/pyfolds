# API — Telemetry

## Componentes

- `TelemetryController`
- `TelemetryConfig`
- Sinks: `MemorySink`, `ConsoleSink`, `JSONLinesSink`, `DistributorSink`
- Eventos: `forward_event`, `commit_event`, `sleep_event`
- `RingBuffer`

## Fases observáveis

- `forward`
- `commit`
- `sleep`

Objetivo: rastrear comportamento do neurônio sem instrumentação intrusiva no loop de treino.
