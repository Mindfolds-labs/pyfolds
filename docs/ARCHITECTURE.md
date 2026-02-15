# ARCHITECTURE — PyFOLDS MPJRD (C4)

Este documento descreve a arquitetura do PyFOLDS usando a visão C4 (Context, Container, Component), com foco no `MPJRDNeuron` e no processamento dendrítico assimétrico.

## 1) Context (C1)

**Sistema:** PyFOLDS (framework de computação neural bioinspirada).

**Atores externos principais:**
- Pesquisador/Engenheiro de ML que define experimentos e hiperparâmetros.
- Pipeline de dados (ex.: MNIST, sinais contínuos, streams sensoriais).
- Stack de observabilidade (telemetria e logs).

**Responsabilidade do sistema:**
- Executar integração dendrítica compartimentalizada.
- Produzir disparo somático com homeostase adaptativa.
- Acumular e aplicar plasticidade (online/batch/sono).

## 2) Container (C2)

### `pyfolds.core`
Núcleo biofísico:
- `config.py`: parâmetros estruturais e dinâmicos (`MPJRDConfig`).
- `synapse.py`: estado sináptico (N, W, I, proteção).
- `dendrite.py`: integração local por ramo dendrítico.
- `neuron.py`: ciclo completo de inferência e aprendizado.
- `homeostasis.py`, `neuromodulation.py`, `accumulator.py`: controle adaptativo e acumulação estatística.

### `pyfolds.network`
Orquestração de múltiplos neurônios e estrutura de rede.

### `pyfolds.advanced`
Mecanismos complementares (STDP, refractory, inhibition, etc.).

### `pyfolds.telemetry`
Eventos estruturados de forward, commit e sono.

## 3) Component (C3) — `MPJRDNeuron`

Pipeline interno do `forward`:
1. **Integração dendrítica local:** cada dendrito recebe `x[:, d, :]` e aplica não linearidade local.
2. **Competição espacial (WTA):** apenas o ramo vencedor contribui para o soma.
3. **Integração somática:** soma dos ramos após gating.
4. **Disparo:** `spike = 1[u >= θ]`.
5. **Homeostase:** atualização de taxa alvo (exceto inferência).
6. **Neuromodulação:** sinal externo ou endógeno.
7. **Acumulação batch:** grava estatísticas para atualização deferida.
8. **Telemetria e logging:** emissão de eventos e rastreabilidade.

## 4) Invariante arquitetural crítico

Para evitar degeneração para perceptron clássico:
- A não linearidade deve ocorrer **dentro de cada dendrito** (compartimentalização).
- A combinação global deve ocorrer **após** a transformação local.

No desenho atual do PyFOLDS, isso é respeitado por:
- `dend(x[:, d, :])` por ramo.
- WTA em `v_dend`.
- Soma somática depois do gating.

## 5) Diagrama técnico

Veja o fluxo em Mermaid em: `docs/diagrams/dendritic_processing_flow.mmd`.
