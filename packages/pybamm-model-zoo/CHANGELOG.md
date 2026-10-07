# Changelog

The model zoo has its own changelog so that zoo pull requests never touch
PyBaMM's.

## [Unreleased]

### Added

- The model zoo: per-model manifests, a registry, a ten-check contract suite, a
  template and generator, generated docs pages and status badges, and the
  `linearised_spm` reference entry ([#5727](https://github.com/pybamm-team/PyBaMM/pull/5727))
- `multilayer_3d_thermal`: a pouch cell stack resolved through its thickness into
  zones that are each PyBaMM's own SPM, SPMe, or DFN under the stack's options,
  each with its own 3D temperature field, connected in parallel or series
  ([#5815](https://github.com/pybamm-team/PyBaMM/pull/5815))
