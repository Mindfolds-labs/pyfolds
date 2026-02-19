# PROMPT — PyFolds 2.0 Migration (Especificação Operacional)

## Objetivo geral
Padronizar a migração para PyFolds 2.0 em etapas independentes, cada uma com **um único objetivo**, critérios de aceite verificáveis e entregáveis objetivos de teste/documentação.

## Regras de execução desta especificação
- Cada etapa deve ser implementada e validada isoladamente.
- Não misturar objetivos entre etapas.
- Tratar os blocos `config`, `state`, `neuron`, `layers`, `network`, `tf` como **requisitos de interface pública e contratos**, não como transcrição de código.
- Toda mudança deve atualizar teste e documentação correspondentes na mesma etapa.

---

## Etapa 1 — Interface de Configuração (`config`)

**Objetivo único**
- Definir e estabilizar o contrato público de configuração da versão 2.0.

**Arquivos-alvo**
- `src/pyfolds/core/config.py`
- `src/pyfolds/core/__init__.py`
- `tests/unit/core/test_config.py`
- `docs/api/core/config.md` (ou arquivo equivalente de API)

**Requisitos de interface (sem código colado)**
- API explícita para criação/validação de configuração.
- Campos obrigatórios/opcionais claramente declarados.
- Política de defaults documentada e determinística.
- Erros de validação com mensagens acionáveis.

**Critérios de aceite verificáveis**
- Instâncias válidas de configuração são aceitas sem warning inesperado.
- Campos inválidos geram exceção tipada e mensagem previsível.
- Defaults são reproduzíveis entre execuções.
- Documentação lista parâmetros, tipos e regras de validação.

**Saída esperada (testes/documentação)**
- Testes unitários cobrindo casos válidos, inválidos e defaults.
- Seção de documentação com tabela de parâmetros e exemplos mínimos.

---

## Etapa 2 — Interface de Estado (`state`)

**Objetivo único**
- Definir o contrato de estado interno/serializável entre componentes.

**Arquivos-alvo**
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/core/neuron_v2.py`
- `src/pyfolds/serialization/foldio.py`
- `tests/unit/core/test_neuron.py`
- `tests/unit/serialization/test_foldio.py`
- `docs/api/serialization/foldio.md` (ou equivalente)

**Requisitos de interface (sem código colado)**
- Estrutura de estado com chaves estáveis e versionáveis.
- Regras para exportar/importar estado (roundtrip sem perda).
- Compatibilidade entre estado em memória e estado persistido.
- Comportamento definido para estado ausente/corrompido.

**Critérios de aceite verificáveis**
- Roundtrip `save -> load -> compare` preserva equivalência lógica do estado.
- Estado corrompido é rejeitado com erro explícito.
- Alterações de versão são identificáveis por metadado de schema.

**Saída esperada (testes/documentação)**
- Testes de roundtrip e de corrupção de estado.
- Documentação com contrato de schema e garantias de compatibilidade.

---

## Etapa 3 — Interface de Neurônio (`neuron`)

**Objetivo único**
- Consolidar o contrato público de execução e métricas do neurônio.

**Arquivos-alvo**
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/core/neuron_v2.py`
- `src/pyfolds/core/factory.py`
- `tests/unit/core/test_neuron.py`
- `tests/unit/core/test_neuron_v2.py`
- `docs/api/core/neuron.md` (ou equivalente)

**Requisitos de interface (sem código colado)**
- Assinaturas públicas estáveis para inicialização, passo de execução e reset.
- Contrato de entradas/saídas (tipos, shapes e unidade semântica).
- Definição de métricas mínimas obrigatórias (ex.: taxa de disparo/saturação).
- Invariantes de segurança (determinismo em modo controlado, validação de device/shape).

**Critérios de aceite verificáveis**
- Chamadas válidas respeitam tipos e shapes esperados.
- Entrada inválida falha cedo com mensagem diagnóstica.
- Métricas obrigatórias são sempre retornadas no formato documentado.

**Saída esperada (testes/documentação)**
- Testes unitários de API, validação e regressão de métricas.
- Documento de contrato com exemplos de uso e tabela de retornos.

---

## Etapa 4 — Interface de Camadas (`layers`)

**Objetivo único**
- Garantir contrato composicional das camadas sobre neurônios.

**Arquivos-alvo**
- `src/pyfolds/layers/layer.py`
- `src/pyfolds/layers/__init__.py`
- `tests/unit/test_layer_neuron_class.py`
- `docs/api/layers/layer.md` (ou equivalente)

**Requisitos de interface (sem código colado)**
- API para construir camada com múltiplos neurônios e configuração homogênea/heterogênea.
- Contrato de propagação de entrada/saída em lote.
- Regras para sincronização de device/dtype.
- Política para agregação de métricas por camada.

**Critérios de aceite verificáveis**
- Camada preserva consistência de shape entre entrada e saída.
- Device/dtype incompatível é tratado conforme contrato (erro ou coerção documentada).
- Métricas agregadas mantêm formato estável.

**Saída esperada (testes/documentação)**
- Testes de composição, lote, device/dtype e métricas agregadas.
- Página de documentação com contrato de construção e forward.

---

## Etapa 5 — Interface de Rede (`network`)

**Objetivo único**
- Definir contrato de orquestração da rede e conectividade entre camadas.

**Arquivos-alvo**
- `src/pyfolds/network/network.py`
- `src/pyfolds/network/__init__.py`
- `tests/unit/network/test_network_edge_cases.py`
- `docs/api/network/network.md` (ou equivalente)

**Requisitos de interface (sem código colado)**
- API de construção de topologia com validação explícita.
- Contrato de execução ponta-a-ponta (entrada da rede -> saída da rede).
- Regras para pesos/conectividade (shape, domínio, consistência).
- Política de erro para topologia inválida e dados malformados.

**Critérios de aceite verificáveis**
- Topologia válida executa sem erro e retorna shape documentado.
- Topologia inválida falha com exceção tipada e mensagem objetiva.
- Validação de weights/spikes cobre cenários de borda.

**Saída esperada (testes/documentação)**
- Testes de edge cases de topologia e validação de shape.
- Documento com contratos de construção, execução e falhas esperadas.

---

## Etapa 6 — Interface TensorFlow (`tf`)

**Objetivo único**
- Definir camada de compatibilidade TensorFlow com paridade de contrato funcional.

**Arquivos-alvo**
- `src/pyfolds/tf/` (módulos de integração)
- `tests/unit/tf/` (testes de integração/unidade)
- `docs/api/tf/` (documentação de compatibilidade)

**Requisitos de interface (sem código colado)**
- API de entrada/saída equivalente ao contrato principal quando aplicável.
- Mapeamento explícito de tipos/tensores suportados.
- Definição de limites conhecidos (features não suportadas e fallback).
- Estratégia de versionamento de compatibilidade.

**Critérios de aceite verificáveis**
- Fluxo mínimo de inferência em TensorFlow reproduz contrato documentado.
- Diferenças de backend são declaradas e cobertas por teste dedicado.
- Casos não suportados falham de forma explícita e previsível.

**Saída esperada (testes/documentação)**
- Testes de paridade de interface (quando possível) e de limitações conhecidas.
- Matriz de compatibilidade TensorFlow na documentação.

---

## Definition of Done (DoD) — Checklist Fechado

> **Somente considerar concluído quando todos os itens abaixo estiverem marcados como `[x]`.**

- [ ] Cada etapa foi executada com **exatamente um objetivo** e sem escopo cruzado.
- [ ] Todos os arquivos-alvo de cada etapa foram atualizados ou explicitamente justificados como “sem alteração necessária”.
- [ ] Cada etapa possui testes automatizados correspondentes aos critérios de aceite.
- [ ] Todos os critérios de aceite foram validados por evidência objetiva (saída de teste/log).
- [ ] Documentação de API foi atualizada para todas as interfaces alteradas.
- [ ] Não há duplicação de blocos de requisito entre etapas.
- [ ] Todos os blocos (`config/state/neuron/layers/network/tf`) estão descritos como contrato de interface (não código colado).
- [ ] Limitações, incompatibilidades e comportamento de erro estão documentados.
- [ ] A execução completa da suíte relevante de testes finaliza com status verde.
- [ ] O prompt final de execução referencia esta especificação como fonte única de verdade para migração 2.0.
