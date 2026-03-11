# ARCHITECTURE — PyFOLDS MPJRD (C4 + fluxo IEEE)

Este documento define a arquitetura do PyFOLDS sob três perspectivas complementares:

1. **C4** (Context, Container, Component),
2. **Fluxo de execução em estilo IEEE**,
3. **Contratos matemáticos mínimos para estabilidade de implementação**.

---

## 1) Contexto do sistema (C1)

### 1.1 Sistema
**PyFOLDS**: framework de computação neural bioinspirada com foco em processamento dendrítico compartimentalizado.

### 1.2 Atores externos

- **Pesquisador/Engenheiro de ML**: define experimento, hiperparâmetros e protocolo.
- **Pipeline de dados**: entrega sinais em lote/stream (ex.: visão, séries temporais).
- **Observabilidade**: coleta telemetria, logs e sinais de saúde operacional.

### 1.3 Responsabilidades de alto nível

- Integrar sinais em ramos dendríticos com não linearidade local.
- Integrar saídas dendríticas com `DendriticIntegration` (modo configurável) antes da decisão somática.
- Executar homeostase e plasticidade (online/batch/sono).
- Emitir eventos auditáveis para depuração e governança.

---

## 2) Containers do sistema (C2)

| Container | Responsabilidade principal | Módulos-chave |
|---|---|---|
| `pyfolds.core` | Núcleo biofísico e ciclo do neurônio | `config.py`, `synapse.py`, `dendrite.py`, `neuron.py`, `homeostasis.py`, `neuromodulation.py`, `accumulator.py`, `dendrite_integration.py` |
| `pyfolds.network` | Orquestração entre múltiplos neurônios | construção e execução de redes |
| `pyfolds.advanced` | Mecanismos complementares | STDP, refractory, adaptation, inibição, etc. |
| `pyfolds.telemetry` | Observabilidade estruturada | eventos de forward/commit/sleep |
| `pyfolds.serialization` | Persistência e versionamento | checkpoints com metadados de integridade |
| `pyfolds.monitoring` | Saúde operacional | classificação `healthy/degraded/critical` |

---

## 3) Componentes críticos (C3) — `MPJRDNeuron`/`MPJRDNeuronV2`

### 3.1 Fluxo real do neurônio

\[
X \in \mathbb{R}^{B\times D\times S}
\xrightarrow{\text{integração por dendrito}}
V \in \mathbb{R}^{B\times D}
\xrightarrow{\text{DendriticIntegration ou gate cooperativo}}
U \in \mathbb{R}^{B}
\xrightarrow{\text{threshold }\theta_{eff}}
Y \in \{0,1\}^{B}.
\]

Etapas funcionais (runtime):

1. `MPJRDNeuron.forward` integra entradas por ramo (`v_dend`) usando rota vetorizada ou fallback por `MPJRDDendrite`.
2. A integração entre ramos depende de `dendrite_integration_mode`:
   - `nmda_shunting`: `DendriticIntegration` (gate sigmoidal + normalização divisiva);
   - `wta_soft`: soma cooperativa de gates sigmoidais;
   - `wta_hard`: seleção vencedora (legado/ablação).
3. O potencial somático `u` é comparado com `theta_eff` para gerar `spikes`.
4. Com mixins, o ponto de decisão é refinado:
   - `AdaptationMixin`: aplica SFA (`u_eff = u - I_adapt`);
   - `RefractoryMixin`: aplica refratário absoluto/relativo e adia homeostase para pós-refratário.
5. Homeostase/plasticidade: homeostase fora de `INFERENCE`; plasticidade online (`_apply_online_plasticity`) ou batch (`apply_plasticity`) conforme trilha.
6. Telemetria e rastros de auditoria são emitidos no boundary do passo.

`MPJRDNeuronV2` preserva o contrato global de saída, com integração cooperativa por soma de `sigmoid(v_dend - \theta/2)`.

### 3.2 Invariante arquitetural obrigatório

A ordem abaixo **não pode ser quebrada**:

\[
\text{Local Nonlinearity} \prec \text{Competition/Integration} \prec \text{Global Aggregation}.
\]

Esse invariante evita degeneração para um perceptron linearizado.

### 3.3 Implementado vs experimental

| Tema | Implementado (runtime principal) | Experimental / pesquisa |
|---|---|---|
| Integração dendrítica | `MPJRDNeuron` com `DendriticIntegration` (`nmda_shunting`) e caminhos `wta_soft`/`wta_hard` configuráveis | Extensões em `wave` e mecanismos opcionais em `advanced` |
| Variante de neurônio | `MPJRDNeuron` em `src/pyfolds/core/neuron.py` | `MPJRDNeuronV2` e variantes `wave` |
| Competição entre ramos | Não é exclusivamente WTA; depende de `dendrite_integration_mode` | Ablações explícitas entre WTA duro e integração cooperativa |
| Adaptação/Refratário | Mixins `AdaptationMixin` e `RefractoryMixin` integráveis ao pipeline | Novas regras temporais e ajustes de parâmetros |
| Plasticidade/Homeostase | Homeostase ativa fora de inferência e plasticidade por trilha (`ONLINE`/`BATCH`/`SLEEP`) | Gating experimental (fase/canais/wave) por toggles |

Veja também: [Mecanismos experimentais](./mechanisms/experimental_toggles.md) e [fundamentos científicos](./science/SCIENTIFIC_LOGIC.md).

---

## 4) Fluxo de execução no padrão IEEE (visão operacional)

### 4.1 Fases do ciclo

| Fase IEEE | Objetivo | Entradas | Saídas |
|---|---|---|---|
| **Acquisition** | receber lote/sinal | \(X\), metadados de execução | lote validado |
| **Local Processing** | computar \(V\) por ramo | \(X, N, W, I\) | potenciais dendríticos |
| **Selection** | aplicar integração/competição por modo (`nmda_shunting`, `wta_soft`, `wta_hard`) | \(V\) | representação integrada (`gated`/`v_nmda`) |
| **Decision** | gerar spike | \(U, \theta\) | \(Y\), \(U\) |
| **Adaptation** | atualizar estado/plasticidade | \(Y\), taxas, modo | estado ajustado |
| **Observability** | registrar rastros e saúde | estados internos | eventos + status |

### 4.2 Contrato de interface (pseudo-assinatura)

```text
forward(
    X: Tensor[B,D,S],
    mode: {INFERENCE, ONLINE, BATCH, SLEEP},
    neuromod_signal: Optional[Tensor],
) -> ForwardResult
```

### 4.3 Equações de referência (consistentes com algoritmo)

\[
V_{b,d} = \sigma_d\!\left(\sum_{s=1}^{S} \psi(N_{d,s},W_{d,s},I_{d,s})\,X_{b,d,s}\right),
\qquad
U_b = f_{int}(V_b, \theta; \texttt{mode}),
\]
\[
Y_b = \mathbb{1}[U_b\ge\theta_{eff}],
\quad
\texttt{mode} \in \{\texttt{nmda\_shunting},\texttt{wta\_soft},\texttt{wta\_hard}\}.
\]

---

## 5) Padrões de engenharia e robustez

- **Factory Pattern (`pyfolds.core.factory`)**: extensibilidade de tipos neurais por registro.
- **Validação defensiva de entrada (`pyfolds.utils.validation.validate_input`)**: evita propagação silenciosa de shape/tipo inválido.
- **Checkpoint versionado (`pyfolds.serialization.versioned_checkpoint`)**: persistência rastreável com `version`, `git_hash`, `config` e hash de integridade.
- **Health Check (`pyfolds.monitoring.health.NeuronHealthCheck`)**: classificação operacional para observabilidade contínua.

---

## 6) Como manter os documentos legíveis no futuro

1. **Separar teoria de implementação**: manter arquivo matemático (`docs/science/`) e arquivo arquitetural (`docs/ARCHITECTURE.md`) sincronizados, mas com propósitos distintos.
2. **Tabela de símbolos obrigatória** em qualquer documento com equações.
3. **Equação curta por bloco**: evitar blocos longos de LaTeX para reduzir quebra de render no GitHub.
4. **Padronizar fases IEEE** (`Acquisition`, `Local Processing`, `Selection`, `Decision`, `Adaptation`, `Observability`) em todos os fluxos técnicos.
5. **Versionar diagramas-fonte** (Mermaid/PlantUML) junto ao texto, para futura geração automática de páginas.

---

## 7) Referência de fluxo visual

Fluxo detalhado em Mermaid: `docs/architecture/blueprints/sources/dendritic_processing_flow.mmd`.

---

## 8) Contratos finais por trilha (release)

As trilhas operacionais canônicas do projeto são: `INFERENCE`, `ONLINE`, `BATCH` e `SLEEP`.

### 8.1 Regras transversais
- Decisão de disparo deve usar `theta_eff` quando disponível; `theta` permanece como estado/base observável.
- `MPJRDLayer.forward` expõe `u_values` como contrato primário e mantém alias legado `u` durante a janela de compatibilidade.
- Evidências de validação devem ser registradas por trilha em cada release.

### 8.2 Contrato por trilha
| Trilha | Contrato final | Critério de não-regressão |
|---|---|---|
| `INFERENCE` | Forward sem mutação de plasticidade/homeostase | Saída consistente (`spikes`, `u_values`/`u`) |
| `ONLINE` | Atualização local imediata com ordem de mecanismos preservada | Decisão de disparo alinhada em `theta_eff` |
| `BATCH` | Acumulação estatística + atualização por lote | STDP com normalização por `mean(dim=0)` |
| `SLEEP` | Replay offline e consolidação estrutural opcional | Pruning só com `consolidate_pruning_after_replay=True` |

### 8.3 Checklist mínimo de release
- [ ] Executar e registrar validação de `INFERENCE`.
- [ ] Executar e registrar validação de `ONLINE`.
- [ ] Executar e registrar validação de `BATCH`.
- [ ] Executar e registrar validação de `SLEEP`.
- [ ] Confirmar compatibilidade de contrato legado (`u`) e canônico (`u_values`).
- [ ] Confirmar rastreabilidade de evidência (comando, resultado, trilha).

## 9) Links cruzados

- Ciência/algoritmo: [`docs/science/ALGORITHM.md`](./science/ALGORITHM.md) e [`docs/science/SCIENTIFIC_LOGIC.md`](./science/SCIENTIFIC_LOGIC.md).
- Hub de ciência: [`docs/science/README.md`](./science/README.md).
- Mecanismos de runtime e flags: [`docs/mechanisms/README.md`](./mechanisms/README.md) e [`docs/mechanisms/experimental_toggles.md`](./mechanisms/experimental_toggles.md).
- Blueprint de fluxo: [`docs/architecture/blueprints/sources/dendritic_processing_flow.mmd`](./architecture/blueprints/sources/dendritic_processing_flow.mmd).
