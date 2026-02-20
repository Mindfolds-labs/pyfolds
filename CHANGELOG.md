# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Canonical root `CONTRIBUTING.md` and `CHANGELOG.md` to strengthen governance onboarding.
- `docs/development/DEVELOPMENT_HUB.md` compatibility shim.
- Release process content in `docs/development/release_process.md`.
- ADR traceability note in `src/pyfolds/serialization/foldio.py`.
- Optional dependency group for examples with `torchvision` in `pyproject.toml`.
- Public API deprecation policy for v1 aliases, including removal target in `2.0.0`, objective criteria (minimum major cycle, changelog notice, migration strategy), and contract test coverage.
- `ModelIntegrityMonitor` para sanity check periódico de hash SHA-256 de pesos/buffers em runtime.
- ADR-046 formalizando o monitoramento periódico de integridade de pesos como extensão de hardening da v2.0.2.

### Deprecated
- v1 public aliases remain supported during `1.x` with `DeprecationWarning` and are scheduled for removal in `2.0.0`: `MPJRDConfig` (use `NeuronConfig`), `MPJRDLayer` (use `AdaptiveNeuronLayer`), and `MPJRDNetwork` (use `SpikingNetwork`).

### Changed
- Canonical release version unified to `2.0.2` across package metadata, public modules and docs examples.
- `docs/ARCHITECTURE.md` diagram reference updated to `docs/architecture/blueprints/`.
- `docs/development/execution_queue.csv` ISSUE-005 marked as partial in progress.

### Fixed
- Fixed: refractory absolute respected; bAP applied; STDP LTD rule corrected; time counter updated; STDP uses pre-STP x.
- Fixed critical STDP batch scaling by normalizing deltas with `mean(dim=0)` instead of `sum(dim=0)`.
- Fixed inhibition/layer contract by exposing membrane potentials in layer output (`u_values`) and consuming them safely in `InhibitionLayer`.
- Fixed STP buffer device migration to keep `register_buffer` registry consistent after device changes.
- Fixed refractory threshold consistency by prioritizing `theta_eff` in refractory masking.
- Fixed health monitoring fallback to `spike_rate=0.0` (instead of `1.0`) when only `r_hat` is absent.


## [2.0.2] - 2026-02-20

### Added
- Added ADR-045 documenting hardening decisions for checkpoint serialization, shape validation and lazy log flush.

### Changed
- `VersionedCheckpoint` now supports safe weight serialization via `.safetensors` with JSON sidecar metadata.
- `CircularBufferFileHandler` now uses interval-based lazy flush with immediate flush on `ERROR+`.
- Release version updated to `2.0.2` across package metadata, module surfaces and docs examples.

### Fixed
- Added explicit tensor shape validation before `load_state_dict` to prevent silent incompatibility loads.

## [2.0.1] - 2026-02-20

### Changed
- MNIST training scripts now default to file-only logging (`train.log`) with optional console output.
- Added explicit CLI logging parameters (`--log-level`, `--log-file`) and `--batch-size` alias support.
- Improved failure handling with guaranteed `summary.json` on errors and automatic ADR/issue evidence generation.
- Updated PowerShell and bash runner guidance for resume flows and crash bundle collection.

## [2.0.0] - 2026-02-16

### Added
- Public v2.0 baseline for PyFolds framework and governance documentation.
