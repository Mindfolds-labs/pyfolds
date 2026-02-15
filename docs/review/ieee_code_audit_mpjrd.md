# IEEE-Style Technical Audit — MPJRD Core (PyFolds)

## Abstract
This document presents a senior-level technical and mathematical review of the MPJRD implementation, with emphasis on (i) fidelity between theoretical equations and code, (ii) stability/domain/convergence conditions, and (iii) runtime flow consistency for learning modes (ONLINE/BATCH/SLEEP/INFERENCE).

## 1. Scope and Method
Analyzed modules:
- `src/pyfolds/core/{config.py, synapse.py, dendrite.py, neuron.py}`
- `src/pyfolds/telemetry/events.py`

Validation dimensions:
1. Mathematical correctness of update equations.
2. Domain constraints and boundedness.
3. Convergence/stability conditions under repeated updates.
4. Semantic coherence between state variables and implementation.

## 2. Theoretical-to-Implementation Mapping

### 2.1 Synaptic weight mapping
Implemented equation:
\[
W(N) = \frac{\log_2(1+N)}{w_{scale}},\quad N\in[n_{min}, n_{max}]
\]

Properties:
- Monotonic in `N` and naturally bounded for finite `n_max`.
- Diminishing increments (`ΔW`) with larger `N`, consistent with saturating structural plasticity.

### 2.2 Three-factor plasticity core
Implemented per-step (rate-based) increment:
\[
\Delta I \propto \eta\cdot R\cdot (pre\cdot post)\cdot gain(W)\cdot dt
\]
with clamping in `I ∈ [i_min, i_max]`.

Assessment:
- Sign and scale are coherent with reward-modulated Hebbian learning.
- Bounded `I` prevents explosive growth; promotion/demotion events map to `N` transitions.

### 2.3 Homeostasis and neuromodulation
- Threshold homeostasis uses rate tracking (`r_hat`, `theta`) in feedback form.
- Endogenous neuromodulation modes (`capacity`, `surprise`) are explicitly clamped to `[-1,1]`.

Assessment:
- Domain handling is robust.
- Stability depends mainly on `homeostasis_eta`, target rate mismatch, and reward variance.

## 3. Objective Findings (High Confidence)

### F1 — Telemetry module import-blocking syntax error (fixed)
`events.py` had signatures with a non-default argument after a default argument in lazy event constructors. This prevented package import and any runtime validation.

Action applied:
- Reordered function parameters so `payload_fn` appears before optional `neuron_id`.

### F2 — Loss of synapse locality in dendritic update (fixed)
In dendritic updates, each synapse received the whole `pre_rate` vector instead of its own scalar component. Because synapse update averages internally, this collapses local learning into a shared mean-like signal.

Mathematical impact:
- Violates local Hebbian assumption (per-synapse pre/post correlation).
- Tends to homogenize synaptic trajectories and reduce representational diversity.

Action applied:
- Enforced `pre_rate` shape `[n_synapses]`.
- Routed `pre_rate[i]` to synapse `i`.

### F3 — Interface inconsistency for short-term states `u` and `R` (fixed)
Dendrite cache/readout expected `syn.u` and `syn.R`, but core synapse class did not expose these attributes, creating a runtime inconsistency.

Action applied:
- Added `u` and `R` buffers in `MPJRDSynapse` initialized with config (`u0`, `R0`).
- Preserves state observability and prevents attribute errors.

## 4. Stability, Domain, and Convergence Notes

### 4.1 Sufficient practical conditions
For stable behavior under long runs:
- `0 < i_gamma < 1` (already coherent in default config).
- Moderate `i_eta·dt` to avoid threshold chattering.
- `homeostasis_eta <= 0.1` (project already warns about larger values).
- Keep reward/neuromodulation bounded (`[-1,1]`) as implemented.

### 4.2 Critical parameter sensitivities
- `beta_w`: positive values reinforce high-`W` synapses and can increase inequality.
- `ltd_threshold_saturated`: controls resilience of saturated synapses.
- `activity_threshold`: directly impacts active-mask density and effective learning rate.
- `consolidation_rate` and sleep duration: govern volatile→structural transfer speed.

## 5. Recommendations (Next Safe Iteration)
1. Add property-level tests for synapse-local updates (different `pre_rate[i]` must produce differentiated `I_i`).
2. Add import smoke test in CI to catch syntax regressions in telemetry modules.
3. Add analytical regression test for monotonicity and boundedness of `W(N)`.
4. Explicitly document STP ownership (core vs advanced module) to avoid duplicated semantics.

## 6. References (Canonical)
- Gerstner, W., Kistler, W. M., Naud, R., Paninski, L. *Neuronal Dynamics*. Cambridge University Press.
- Dayan, P., Abbott, L. F. *Theoretical Neuroscience*. MIT Press.
- Frémaux, N., Gerstner, W. (2016). Neuromodulated STDP and three-factor learning rules.
