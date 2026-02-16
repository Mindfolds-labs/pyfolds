# Guide — Neuron Architecture

## Pipeline atual

```mermaid
flowchart LR
    X[Input BxDxS] --> D[Integração por dendrito]
    D --> G[Gate competitivo]
    G --> U[u somático]
    U --> S[Spike por limiar theta]
    S --> P[Plasticidade / Acúmulo]
```

## Nota de evolução

A base atual usa gate competitivo tipo WTA no `forward`. A trilha v2/v3 discute evolução para integração cooperativa como roadmap arquitetural.
