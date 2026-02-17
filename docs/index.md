# PyFolds Documentation Index

> **Document Status:** Stable  
> **Audience:** Researchers, ML Engineers, Systems Engineers, Maintainers  
> **Standard Orientation:** IEEE-style technical documentation (structured, traceable, reproducible)

---

## Abstract

This documentation set defines the technical and scientific baseline for **PyFolds v2.0/v3.0**, a bio-inspired neural computation framework centered on the **MPJRD (Multi-Pathway Joint-Resource Dendritic)** model. The goal is to provide both operational guidance (*how to use*) and mechanistic rationale (*why it works*), with explicit traceability from architecture to API and runtime behavior.

---

## 1. Purpose and Scope

### 1.1 Purpose

This index is the canonical entry point for project documentation. It organizes content to support:

- onboarding and setup,
- architecture and design understanding,
- API-level implementation and integration,
- scientific interpretation of model behavior,
- internal governance and development control.

### 1.2 Scope

The documentation covers:

- core MPJRD computation pipeline,
- advanced mechanisms (STDP, adaptation, inhibition, refractory, short-term dynamics),
- telemetry/observability surfaces,
- scientific logic and theoretical references,
- governance and development workflows.

---

## 2. Audience Navigation

### 2.1 Public (user-facing)
- [`installation.md`](_quickstart/installation.md)
- [`quickstart.md`](_quickstart/quickstart.md)
- [`public/guides/getting_started.md`](public/guides/getting_started.md)
- [`API_REFERENCE.md`](api/API_REFERENCE.md)

### 2.2 Internal (development and governance)
- [`development/DEVELOPMENT.md`](development/DEVELOPMENT.md)
- [`development/HUB_CONTROLE.md`](development/HUB_CONTROLE.md)
- [`development/CONTRIBUTING.md`](development/CONTRIBUTING.md)
- [`governance/MASTER_PLAN.md`](governance/MASTER_PLAN.md)
- [`governance/adr/INDEX.md`](governance/adr/INDEX.md)

---

## 3. Document Architecture (C4-Aligned)

PyFolds documentation is structured in layers, aligned with C4 communication practices:

- **Level 1 — Context:** PyFolds in research/engineering workflows.
- **Level 2 — Containers:** `core`, `advanced`, `layers`, `network`, `telemetry`, `utils`.
- **Level 3 — Components:** Synapse → Dendrite → Soma → Axon.
- **Level 4 — Code:** classes, methods, data contracts, runtime outputs.

---

## 4. Recommended Reading Sequence

For efficient onboarding and technical depth progression:

1. [`installation.md`](_quickstart/installation.md)
2. [`quickstart.md`](_quickstart/quickstart.md)
3. [`public/guides/core_concepts.md`](public/guides/core_concepts.md)
4. [`public/guides/neuron_architecture.md`](public/guides/neuron_architecture.md)
5. [`api/core.md`](api/core.md)
6. [`SCIENTIFIC_LOGIC.md`](science/SCIENTIFIC_LOGIC.md)

---

## 5. Documentation Map

### 5.1 Getting Started
- [`installation.md`](_quickstart/installation.md)
- [`quickstart.md`](_quickstart/quickstart.md)
- [`public/guides/getting_started.md`](public/guides/getting_started.md)

### 5.2 Guides (Conceptual and Operational)
- [`public/guides/core_concepts.md`](public/guides/core_concepts.md)
- [`public/guides/neuron_architecture.md`](public/guides/neuron_architecture.md)
- [`public/guides/engineering_patterns.md`](public/guides/engineering_patterns.md)
- [`public/guides/plasticity.md`](public/guides/plasticity.md)
- [`public/guides/homeostasis.md`](public/guides/homeostasis.md)
- [`public/guides/neuromodulation.md`](public/guides/neuromodulation.md)
- [`public/guides/advanced_mechanisms.md`](public/guides/advanced_mechanisms.md)
- [`public/guides/telemetry.md`](public/guides/telemetry.md)
- [`public/guides/logging.md`](public/guides/logging.md)

### 5.3 API Reference
- [`api/core.md`](api/core.md)
- [`api/network.md`](api/network.md)
- [`api/layers.md`](api/layers.md)
- [`api/advanced.md`](api/advanced.md)
- [`api/utils.md`](api/utils.md)
- [`api/telemetry.md`](api/telemetry.md)

### 5.4 Theory and Scientific Logic
- [`SCIENTIFIC_LOGIC.md`](science/SCIENTIFIC_LOGIC.md)
- [`research/mpjrd/mpjrd_model.md`](research/mpjrd/mpjrd_model.md)
- [`research/mpjrd/three_factor_learning.md`](research/mpjrd/three_factor_learning.md)
- [`research/mpjrd/two_factor_consolidation.md`](research/mpjrd/two_factor_consolidation.md)
- [`research/mpjrd/stdp_mechanism.md`](research/mpjrd/stdp_mechanism.md)

### 5.5 Development and Governance
- [`development/HUB_CONTROLE.md`](development/HUB_CONTROLE.md)
- [`development/TESTING.md`](development/TESTING.md)
- [`development/release_process.md`](development/release_process.md)
- [`governance/QUALITY_ASSURANCE.md`](governance/QUALITY_ASSURANCE.md)
- [`governance/RISK_REGISTER.md`](governance/RISK_REGISTER.md)

---

## 6. Traceability Matrix (Design → Artifact)

| Design Concern | Primary Artifact | Verification Surface |
|---|---|---|
| Engenharia operacional (Factory/Validation/Checkpoint/Health) | `public/guides/engineering_patterns.md` | testes unitários + contratos de uso |
| Core neuron pipeline | `public/guides/neuron_architecture.md`, `api/core/neuron.md` | forward outputs (`u`, `v_dend`, `spikes`) |
| Structural plasticity (`N`, `I`, `W`) | `public/guides/plasticity.md`, `api/core/synapse.md` | state transitions and thresholds |
| Homeostatic stability | `public/guides/homeostasis.md`, `api/core/homeostasis.md` | `theta`, `r_hat`, target rate dynamics |
| Neuromodulatory control | `public/guides/neuromodulation.md`, `api/core/neuromodulation.md` | `R` behavior across modes |
| Batch/sleep consolidation | `api/core/accumulator.md`, theory docs | commit/sleep phase outputs |
| Observability and auditability | `public/guides/telemetry.md`, `api/telemetry.md` | event stream and sink outputs |
| Governança de documentação interna | `development/HUB_CONTROLE.md`, `governance/adr/INDEX.md` | update log + ADR consistency |

---

## 7. Documentation Quality Principles

This documentation follows engineering-oriented principles:

- **Consistency:** terminology and naming aligned with source code.
- **Scannability:** tables, section numbering, and concise subsections.
- **Reproducibility:** runnable examples and explicit mode semantics.
- **Traceability:** each conceptual statement maps to concrete artifacts.
- **Non-ambiguity:** explicit distinction between current behavior and roadmap.

---

## 8. Versioning Note

This index is maintained for the `v2.0/v3.0` documentation track. Behavioral details marked as roadmap should not be interpreted as current runtime guarantees unless explicitly implemented in code.
