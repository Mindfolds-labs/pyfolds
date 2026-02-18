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
- Aplicar competição espacial (WTA) antes da decisão somática.
- Executar homeostase e plasticidade (online/batch/sono).
- Emitir eventos auditáveis para depuração e governança.

---

## 2) Containers do sistema (C2)

| Container | Responsabilidade principal | Módulos-chave |
|---|---|---|
| `pyfolds.core` | Núcleo biofísico e ciclo do neurônio | `config.py`, `synapse.py`, `dendrite.py`, `neuron.py`, `homeostasis.py`, `neuromodulation.py`, `accumulator.py` |
| `pyfolds.network` | Orquestração entre múltiplos neurônios | construção e execução de redes |
| `pyfolds.advanced` | Mecanismos complementares | STDP, refractory, inibição, etc. |
| `pyfolds.telemetry` | Observabilidade estruturada | eventos de forward/commit/sleep |
| `pyfolds.serialization` | Persistência e versionamento | checkpoints com metadados de integridade |
| `pyfolds.monitoring` | Saúde operacional | classificação `healthy/degraded/critical` |

---

## 3) Componentes críticos (C3) — `MPJRDNeuron`

### 3.1 Pipeline interno do `forward`

\[
X \in \mathbb{R}^{B\times D\times S}
\xrightarrow{\text{dendrite integrate}}
V \in \mathbb{R}^{B\times D}
\xrightarrow{\text{WTA}}
G \in \{0,1\}^{B\times D}
\xrightarrow{\text{somatic sum}}
U \in \mathbb{R}^{B}
\xrightarrow{\text{threshold}}
Y \in \{0,1\}^{B}.
\]

Etapas funcionais:

1. Integração local por ramo.
2. Competição WTA (ramo vencedor).
3. Soma somática pós-gating.
4. Disparo por limiar \(\theta\).
5. Homeostase (exceto inferência).
6. Neuromodulação exógena/endógena.
7. Acumulação estatística (modo batch).
8. Emissão de telemetria.

### 3.2 Invariante arquitetural obrigatório

A ordem abaixo **não pode ser quebrada**:

\[
\text{Local Nonlinearity} \prec \text{Competition} \prec \text{Global Aggregation}.
\]

Esse invariante evita degeneração para um perceptron linearizado.

---

## 4) Fluxo de execução no padrão IEEE (visão operacional)

### 4.1 Fases do ciclo

| Fase IEEE | Objetivo | Entradas | Saídas |
|---|---|---|---|
| **Acquisition** | receber lote/sinal | \(X\), metadados de execução | lote validado |
| **Local Processing** | computar \(V\) por ramo | \(X, N, W, I\) | potenciais dendríticos |
| **Selection** | aplicar competição espacial | \(V\) | máscara \(G\) |
| **Decision** | gerar spike | \(G, V, \theta\) | \(Y\), \(U\) |
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
\quad
G_{b,d} = \mathbb{1}\left[d=\arg\max_j V_{b,j}\right],
\]
\[
U_b = \sum_{d=1}^{D} G_{b,d}V_{b,d},
\qquad
Y_b = \mathbb{1}[U_b\ge\theta].
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

Fluxo detalhado em Mermaid: `docs/architecture/blueprints/dendritic_processing_flow.mmd`.
