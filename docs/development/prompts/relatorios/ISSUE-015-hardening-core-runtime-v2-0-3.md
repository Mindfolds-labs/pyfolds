---
id: "ISSUE-015"
titulo: "Hardening final do Core: integridade runtime, telemetria bufferizada e carga segura v2.0.3"
prioridade: "Alta"
area: "Segurança/Runtime/Telemetria"
fase: "concluida"
adr_vinculada: "ADR-047"
---

# ISSUE-015: Hardening final do Core v2.0.3

## Objetivo
Consolidar as alterações finais de hardening para a release `2.0.3` no Core:
- monitor de integridade de pesos em runtime;
- sink JSONL bufferizado;
- carregamento seguro com manifesto + validação de hash/shape.

## Critérios de aceite
- [x] `WeightIntegrityMonitor` disponível na API pública.
- [x] `BufferedJSONLinesSink` implementado e testado.
- [x] `VersionedCheckpoint.load_secure` implementado com validações rígidas.
- [x] Versão `2.0.3` atualizada em metadados de release.
- [x] Fila/HUB atualizados com rastreabilidade documental.
