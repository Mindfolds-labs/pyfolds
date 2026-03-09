# Migração de Telemetria

Antes: `TelemetryController` com sinks síncronos.

Depois: `TelemetryCollector` assíncrono com exporters desacoplados.

Compatibilidade:
- APIs legadas continuam disponíveis em `pyfolds.telemetry`.
