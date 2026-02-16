# Guide — Telemetry

A telemetria expõe eventos de ciclo de vida do neurônio:

- `forward`: atividade e estado durante inferência/treino.
- `commit`: aplicação de plasticidade acumulada.
- `sleep`: consolidação offline.

Com `MemorySink`/`JSONLinesSink`, é possível criar auditoria e observabilidade reproduzível.
