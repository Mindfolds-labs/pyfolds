# Telemetry Architecture

Fluxo: `emit -> RingBufferThreadSafe -> worker thread -> exporters`.

- `emit` é O(1) e não faz IO.
- Buffer circular com descarte controlado e métrica de dropped events.
- Worker faz flush em lotes, isolando falhas por exporter.
- Shutdown executa flush final e close.

Limitações: em Python a implementação é thread-safe com lock curto, não lock-free real.
