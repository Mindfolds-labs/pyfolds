# Engineering Architecture Review

## Scope analyzed
- `src/pyfolds/core`
- `src/pyfolds/layers` (closest package to requested `models/`)
- `src/pyfolds/serialization`
- `src/pyfolds/telemetry`
- `src/pyfolds/utils`

## Findings

### 1) Tight coupling
1. `MPJRDNeuron` aggregates many responsibilities (forward compute, telemetry integration, runtime queue processing, homeostasis callbacks, gradient sanitization), creating a **god-object** hotspot and difficult unit boundaries.
2. `core/neuron.py` depends directly on concrete components (`MPJRDDendrite`, `HomeostasisController`, `Neuromodulator`, `StatisticsAccumulator`, telemetry functions), which limits replaceability and simulation backends.
3. `serialization/foldio.py` centralizes container format, integrity, provenance, cryptography hooks and compatibility checks in a single module; strategy-specific concerns are not isolated.

### 2) Unused or low-signal modules
- The repository contains overlapping telemetry abstractions (`buffer.py`, `ringbuffer.py`, `collector.py`) with partially duplicated behavior, suggesting candidate consolidation.
- Legacy/lateral package trees (e.g. `tf`, `mobile`, `advanced`) contain APIs with sparse references from core execution paths and should be continuously validated for usage drift.

### 3) Circular import risk
- Current imports avoid explicit immediate cycles in core paths, but `__init__.py` aggregator exports in `telemetry` and `serialization` increase **future cycle risk** because high-level modules re-export low-level implementation symbols.
- Recommendation: enforce import layering (`types -> buffer -> collector -> controller`, no reverse imports).

### 4) Naming and readability
- Inconsistent language mix (`Portuguese + English`) in docstrings/logs/class descriptions increases maintenance cost for external contributors.
- Some short names (`u`, `R`, `N`, `L`) are scientifically meaningful but should be systematically documented in public API docs.

### 5) Computational inefficiencies
- Neuron forward path previously iterated over dendrites in Python list comprehension, which reduced GPU kernel fusion opportunities.
- Synaptic updates in dendrites still iterate synapse-by-synapse (required for local plasticity semantics), but can be future-targeted via batched update primitives once equivalence is proven.

## Recommended roadmap
1. Isolate neuron compute kernel from orchestration (telemetry/homeostasis/runtime queue).
2. Keep a compatibility path and add feature-flagged vectorized execution (implemented in this refactor).
3. Move serialization choices to explicit strategy objects (implemented in this refactor).
4. Adopt strict typing incrementally by package, starting with newly added modules.
5. Add property tests to encode neuron invariants and reduce regression risk.
