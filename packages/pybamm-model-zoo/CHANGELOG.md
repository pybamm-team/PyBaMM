# Changelog

The model zoo has its own changelog so that zoo pull requests never touch
PyBaMM's.

## [Unreleased]

### Added

- The model zoo: per-model manifests, a registry, a ten-check contract suite, a
  template and generator, generated docs pages and status badges, and the
  `linearised_spm` reference entry ([#5727](https://github.com/pybamm-team/PyBaMM/pull/5727))

### Changed

- In-tree models must be licensed BSD-3-Clause, enforced by the `packaging`
  contract check; the generator's `--license` option is removed (PR_LINK)
