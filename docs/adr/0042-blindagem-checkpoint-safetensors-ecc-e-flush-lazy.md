# ADR 0042: Blindagem de checkpoint com safetensors, ECC e flush lazy de logs

- **Status:** Aceito
- **Data:** 2026-02-20
- **Decisores:** Maintainers PyFolds
- **Relacionados:** [ADR 0001](./0001-import-contract-and-release-readiness.md), [ADR 0041](./0041-modelo-de-fases-ciclo-continuo-e-legado.md)

## Contexto

A auditoria técnica identificou três vetores de melhoria imediata:

1. **Segurança de serialização:** reduzir a superfície de ataque de checkpoints `.pt` dependentes de pickle.
2. **Eficiência de I/O em logs:** evitar reescrita do buffer circular a cada `emit`.
3. **Tolerância a falhas físicas:** habilitar recuperação de corrupção por bitflip em payloads de checkpoint.

Também foi recomendado validar invariantes de shape antes de injetar pesos no modelo.

## Decisão

1. Introduzir suporte nativo a checkpoints blindados em **safetensors** no `VersionedCheckpoint`.
2. Adotar sidecar de metadados `.json` com hash de integridade e metadados versionados.
3. Adotar sidecar ECC (`.safetensors.ecc`) opcional com Reed-Solomon para recuperação por chunk de payload.
4. Validar shapes (`model.state_dict()` vs checkpoint) antes de `load_state_dict`.
5. Evoluir `CircularBufferFileHandler` com **flush temporizado** e flush imediato em eventos `ERROR+`.

## Consequências

### Positivas

- Menor risco de exploração por desserialização insegura ao usar `safetensors`.
- Redução de overhead de I/O em logging contínuo.
- Maior robustez a corrupção física de dados em armazenamento instável.
- Falhas de compatibilidade estrutural (shape) passam a falhar cedo e com mensagem explícita.

### Negativas

- Novo contrato de arquivos sidecar (`.json`, `.ecc`) para o fluxo safetensors.
- Introduz parâmetros adicionais de API (`use_safetensors`, `ecc_protection`, `circular_flush_interval_sec`).

## Plano de implementação

- Atualizar `VersionedCheckpoint` para salvar/carregar `.safetensors` + `.json` + `.ecc`.
- Reutilizar codec ECC existente (`ecc_from_protection` / `ReedSolomonECC`).
- Adicionar utilitário `ECCProtector` para cenários de chunk único fora de `.fold/.mind`.
- Atualizar `CircularBufferFileHandler` com flush lazy configurável.
- Cobrir os fluxos com testes unitários.

## Critérios de aceite

- [x] Checkpoint `safetensors` com metadados versionados e hash de integridade.
- [x] Validação de shape antes de aplicar `load_state_dict`.
- [x] Buffer circular com flush temporizado + flush imediato em `ERROR`.
- [x] Testes unitários cobrindo os novos contratos.
