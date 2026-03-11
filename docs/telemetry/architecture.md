# Telemetry Architecture

Arquitetura atual (implementada): `emit -> Sink (MemorySink/JSONLines/etc.)`.

- O `emit` delega para o sink configurado no controller (`src/pyfolds/telemetry/controller.py`).
- Os sinks concretos vivem em `src/pyfolds/telemetry/sinks.py`.
- No caso de `MemorySink`, o armazenamento é um `RingBuffer` definido em `src/pyfolds/telemetry/ringbuffer.py`.
- Outros sinks (por exemplo `JSONLines`) realizam persistência conforme sua implementação, sem thread worker global no caminho atual.

## Roadmap (visão futura, não comportamento atual)

Para evolução, pode existir um pipeline assíncrono explícito, por exemplo com worker dedicado para flush em lote e isolamento por exporter. Isso **não** descreve o comportamento atual; é apenas direção de roadmap.

## Limitações e latência

- O impacto de latência depende do perfil/sink ativo (ex.: memória tende a menor custo; escrita em disco pode variar conforme ambiente).
- O parâmetro `sample_every` reduz volume de eventos processados, trocando granularidade por menor overhead.
- Em perfis de maior instrumentação, avaliar o custo de emissão no caminho crítico e ajustar `sample_every` conforme necessidade.
