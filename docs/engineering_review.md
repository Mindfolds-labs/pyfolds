# Revisão de Engenharia — PyFolds (MPJRD)

## 1) Escopo e método

### Escopo analisado
- Núcleo de execução: `src/pyfolds/core`
- Mecanismos avançados (STDP/adaptação/refratário): `src/pyfolds/advanced`
- Integração em camadas: `src/pyfolds/layers` e `src/pyfolds/wave`
- Telemetria e observabilidade: `src/pyfolds/telemetry`
- Persistência e segurança de serialização: `src/pyfolds/serialization`
- Contratos e API de fronteira: `src/pyfolds/contracts`

### Método
Análise estática com foco em **Clean Architecture, modularidade, testabilidade, performance, segurança de execução e consistência científica**.

---

## 2) Diagnóstico arquitetural

## Pontos fortes
1. **Cobertura de testes extensa por domínio** (core, telemetry, advanced, integração), sinalizando maturidade de validação incremental.
2. **Configuração rica e tipada por dataclasses** em `MPJRDConfig`, com parâmetros explícitos para plasticidade, homeostase, dinâmica temporal e segurança operacional.
3. **Preocupação explícita com hardening** na serialização (`safetensors`, limites de header/chunk, verificação criptográfica opcional).
4. **Evolução para contratos científicos** via `ScientificContract` + `ContractEnforcer`, reduzindo risco de regressão semântica no pipeline dendrítico.

## Fragilidades estruturais (alto impacto)
1. **`MPJRDNeuron` concentra responsabilidades demais**: orquestração, runtime queue, telemetria, cache, estado circadiano, contratos, e parte da lógica de controle. Isso cria um ponto único de complexidade e aumenta custo cognitivo/manutenção.
2. **Acoplamento concreto no core**: o neurônio instancia diretamente dependências (`MPJRDDendrite`, `HomeostasisController`, `Neuromodulator`, `StatisticsAccumulator`, etc.), reduzindo substituibilidade de backend e dificultando mocks.
3. **`foldio.py` ainda atua como módulo monolítico** (I/O de container, integridade, crypto hooks, compatibilidade, utilidades de runtime). Apesar de robusto, dificulta evolução isolada e tuning de performance por subdomínio.
4. **Arquitetura em camadas parcialmente implícita**: há boa separação por pastas, mas faltam *boundaries* estritos (ex.: regras formais para import direction entre core/advanced/telemetry/serialization).

---

## 3) Avaliação por critério

## Clean Architecture
- **Estado atual:** organização por pacotes é boa, mas entidades de domínio ainda conhecem detalhes operacionais em excesso.
- **Risco:** domínio científico ficar “contaminado” por concerns de observabilidade e runtime management.
- **Recomendação:** separar `MPJRDNeuron` em:
  - `NeuronKernel` (equações/estado científico)
  - `NeuronRuntimeOrchestrator` (fila, modos, ciclo)
  - `NeuronObservabilityPort` (telemetria/log/auditoria)
  - `NeuronStateRepository` (checkpoint/snapshot)

## Modularidade e testabilidade
- **Estado atual:** suíte de testes ampla, porém fronteiras de injeção ainda limitadas.
- **Risco:** testes unitários precisarem montar objetos “pesados” para validar regras pequenas.
- **Recomendação:** introduzir interfaces/protocolos para componentes do ciclo (`PlasticityEngine`, `HomeostasisEngine`, `TelemetrySinkPort`) e adotar injeção por fábrica no construtor.

## Performance
- **Estado atual:** existem sinais de otimização (cache de pesos, flags de vetorização, benchmark tooling).
- **Risco:** gargalos Python-level em loops de sinapse e overhead de telemetria em perfis agressivos.
- **Recomendação técnica:**
  1. Consolidar caminho hot-path com tensor ops/batching onde não viola semântica local.
  2. Tornar telemetria 100% lazy para payloads caros em perfis `light/heavy`.
  3. Medir custo incremental por mecanismo (STDP, adaptação, circadiano) com benchmark por feature-flag.
  4. Definir budget de latência por `forward` (p50/p95) e budget de memória por neurônio.

## Segurança de execução
- **Estado atual:** bom baseline em serialização segura e limites defensivos.
- **Risco:** concentração de segurança em um módulo único aumenta chance de regressão transversal.
- **Recomendação:** segmentar políticas em componentes explícitos:
  - `PayloadValidator`
  - `IntegrityVerifier`
  - `TrustPolicyEvaluator`
  - `CompatibilityPolicy`
  com testes dedicados por componente + testes de composição.

## Consistência científica
- **Estado atual:** contrato de ordenação de estágios já existe e é um excelente alicerce.
- **Risco:** crescimento de mecanismos opcionais (circadiano/noético/experimental) pode introduzir interações não validadas.
- **Recomendação:** formalizar **invariantes científicas** como testes de propriedade:
  - limites de `theta` e estabilidade homeostática
  - preservação de ordem causal em STDP
  - monotonicidade/limites de variáveis de refratariedade e adaptação
  - estabilidade numérica em regimes extremos (inputs sparsos/densos)

---

## 4) Proposta de refatoração (priorizada)

## Fase 1 — Estrutura (baixo risco, alto retorno)
1. Extrair interfaces de domínio em `contracts/` para mecanismos do núcleo.
2. Quebrar `MPJRDNeuron` em componentes orquestráveis sem alterar API pública.
3. Criar `Architecture Decision Test` garantindo direção de imports (ex.: `core` não depende de `serialization`).

## Fase 2 — Performance dirigida por evidência
1. Criar benchmark matrix por combinação de mecanismos (base, +stdp, +homeostasis, +telemetry).
2. Introduzir “fast path” opcional com vetorização ampliada e fallback seguro.
3. Adicionar métrica de overhead de telemetria por perfil no CI.

## Fase 3 — Segurança e confiabilidade científica
1. Modularizar `foldio` por estratégia (integridade, crypto, chunk codec, compat).
2. Fortalecer testes de mutação/fuzzing para payloads de serialização.
3. Criar suíte de regressão científica (golden traces) para cenários canônicos MPJRD.

## Fase 4 — Governança técnica
1. Definir matriz de estabilidade da API (experimental vs estável).
2. Versionar contratos científicos e de telemetria.
3. Publicar “Engineering Scorecard” por release (perf, cobertura, segurança, invariantes).

---

## 5) Plano de evolução (90 dias)

## 0–30 dias
- Extrair interfaces críticas do core.
- Reduzir responsabilidades de `MPJRDNeuron` em pelo menos 30% (linhas/cohesion split).
- Implantar testes arquiteturais de dependência.

## 31–60 dias
- Entregar benchmark contínuo com baseline público e thresholds.
- Otimizar hot-path com metas de p95 de latência.
- Introduzir property-based tests para invariantes científicas.

## 61–90 dias
- Concluir modularização de serialização e política de confiança.
- Validar backward compatibility com corpus histórico `.fold/.mind`.
- Publicar guia de extensão para novos mecanismos bioinspirados sem acoplamento ao core.

---

## 6) KPIs recomendados

- **Arquitetura:** fan-in/fan-out por módulo, complexidade ciclomática do `core/neuron.py`, número de violações de import-layer.
- **Qualidade:** taxa de flaky tests, mutação score (serialization/core), cobertura de invariantes científicas.
- **Performance:** `forward` p50/p95, memória por neurônio, overhead de telemetria por perfil.
- **Segurança:** taxa de bloqueio de payload malformado, tempo médio de validação de trust block, incidência de fallback inseguro.

---

## 7) Conclusão executiva

O PyFolds já demonstra base técnica robusta para pesquisa neuromórfica, com bons sinais de maturidade em testes e hardening. O principal vetor de evolução agora é **desacoplar o núcleo científico da orquestração operacional**, transformando o ganho funcional atual em **escalabilidade arquitetural**. A prioridade deve ser refatoração orientada a contratos + performance baseada em benchmark + consolidação formal de invariantes científicas.
