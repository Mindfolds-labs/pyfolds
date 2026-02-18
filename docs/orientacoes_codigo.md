# Orientações de Código (PyFolds)

Este guia consolida orientações práticas para contribuir no código do projeto, com foco em **imports**, **organização de classes**, **telemetria/monitoramento** e **auditoria de arquitetura**.

## 1) Imports

- Prefira imports explícitos e estáveis por módulo (ex.: `from pyfolds.core.neuron import MPJRDNeuron`).
- Evite dependências circulares entre `core`, `advanced`, `monitoring` e `telemetry`.
- Quando houver import opcional/pesado, mantenha o import local ao método para reduzir acoplamento e custo de inicialização.
- Não capture erros de import sem necessidade; trate problemas de ambiente no setup/dependências.

## 2) Organização de classes

- Classe base em `core` deve concentrar invariantes e sincronização compartilhada.
- Variações experimentais (ex.: V2) devem herdar comportamento crítico da base (locks, semântica de contador, validação de device e contratos de retorno).
- Métodos de métrica (`get_metrics`) devem expor chaves estáveis; consumidores (health checks) devem prever fallback compatível para versões antigas.
- Tipagem (`TypedDict`, `Protocol`) deve refletir o retorno real em runtime para evitar divergência entre documentação e uso.

## 3) Telemetria e monitoramento

- Incremento de `step_id` deve ocorrer sob lock quando houver execução concorrente.
- Emissores devem respeitar `should_emit(step_id)` para manter custo previsível.
- Health checks devem usar métricas efetivamente produzidas por `get_metrics()`.
- Evite defaults que ocultem falhas silenciosamente (ex.: chave inexistente sempre resultando em estado "healthy").

## 4) Reprodutibilidade

- Inicializações pseudoaleatórias devem usar **um fluxo de RNG consistente** quando pesos e máscara dependem da mesma seed.
- Evite misturar `manual_seed` global com `Generator` separado para partes acopladas da mesma inicialização.
- Preferir `torch.Generator(..., seed=...)` passado explicitamente para APIs com suporte.

## 5) Checklist de auditoria (rápido)

1. **Licença**: `LICENSE`, `pyproject.toml` e `setup.cfg` consistentes.
2. **Contrato de métricas**: chaves retornadas por `get_metrics()` compatíveis com monitoramento.
3. **Concorrência**: contadores e emissão de telemetria protegidos por lock.
4. **Tipagem**: `TypedDict` e enums alinhados ao retorno real.
5. **Determinismo**: inicialização com seed reproduzível.

## 6) Auditoria de arquitetura (sugestão de rotina)

- Executar auditoria de documentação/hub:
  - `python tools/docs_hub_audit.py`
- Validar consistência de API documentada:
  - `python tools/check_api_docs.py`
- Rodar testes focados em áreas críticas alteradas:
  - `pytest tests/unit/core/test_monitoring_and_checkpoint.py tests/unit/core/test_neuron_v2.py tests/unit/advanced/test_inhibition.py tests/unit/telemetry/test_controller.py`

---

Se este guia evoluir, mantenha as regras sincronizadas com ADRs e documentos de governança em `docs/governance/`.
