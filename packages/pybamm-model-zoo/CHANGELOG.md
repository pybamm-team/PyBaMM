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
- Published to PyPI as `pybamm-model-zoo`, requiring `pybamm>=26.10`, with
  `pybamm_model_zoo.__version__`. `multilayer_3d_thermal` gets PyBaMM's
  finite-element meshing from a new `zoo-multilayer-3d-thermal` extra, which
  `zoo-all` includes ([#5881](https://github.com/pybamm-team/PyBaMM/pull/5881))

### Changed

- Every model shares the zoo's `pybamm>=` floor, and a model's dependencies are
  only its `zoo-<slug>` extra, read from the installed metadata: manifests no
  longer declare `pybamm_requires` or `[model.dependencies]`. A model whose extra
  is missing raises `ModelUnavailableError`, naming the `pip install` that
  provides it, from `zoo.load()` and from `pybamm_model_zoo.require()` at the
  start of the model's `__init__`, before any work
  ([#5881](https://github.com/pybamm-team/PyBaMM/pull/5881))
- In-tree models must be licensed BSD-3-Clause, enforced by a new unwaivable
  `license` contract check; the generator's `--license` option is removed ([#5862](https://github.com/pybamm-team/PyBaMM/pull/5862))
