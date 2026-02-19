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

### Deprecated
- v1 public aliases remain supported during `1.x` with `DeprecationWarning` and are scheduled for removal in `2.0.0`: `MPJRDConfig` (use `NeuronConfig`), `MPJRDLayer` (use `AdaptiveNeuronLayer`), and `MPJRDNetwork` (use `SpikingNetwork`).

### Changed
- Canonical release version unified to `2.0.0` across package metadata, public modules and docs examples.
- `docs/ARCHITECTURE.md` diagram reference updated to `docs/architecture/blueprints/`.
- `docs/development/execution_queue.csv` ISSUE-005 marked as partial in progress.

### Fixed
- Fixed: refractory absolute respected; bAP applied; STDP LTD rule corrected; time counter updated; STDP uses pre-STP x.

## [2.0.0] - 2026-02-16

### Added
- Public v2.0 baseline for PyFolds framework and governance documentation.
