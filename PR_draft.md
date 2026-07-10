# Description

Adds three-electrode EIS support to `pybamm.EISSimulation`.

When `three_electrodes=True`, `EISSimulation` now inserts a default reference electrode at the separator midpoint if one has not already been inserted. The EIS setup promotes the positive and negative 3E potentials to algebraic probe variables, returns named impedance components in the `EISSolution`, and keeps the default `solution.impedance` as the cell impedance.

`EISSolution.nyquist_plot()` now detects three-electrode impedance components and plots the cell, positive electrode, and negative electrode curves together.

Fixes # (issue)

## Type of change

Feature.

Changelog entry required under `# [Unreleased]` / `## Features`:

- Added three-electrode EIS support to `pybamm.EISSimulation`, including automatic default reference-electrode insertion, named positive/negative electrode impedance outputs, and component-aware Nyquist plotting. ([#XXXX](https://github.com/pybamm-team/PyBaMM/pull/XXXX))

# Important checks:

Please confirm the following before marking the PR as ready for review:
- [ ] No style issues: `nox -s pre-commit`
- [ ] All tests pass: `nox -s tests`
- [ ] The documentation builds: `nox -s doctests`
- [x] Code is commented for hard-to-understand areas
- [x] Tests added that prove fix is effective or that feature works

Focused checks run locally:
- `.venv/bin/ruff check packages/pybamm/tests/unit/test_eis_simulation.py`
- `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplcache .venv/bin/pytest packages/pybamm/tests/unit/test_eis_simulation.py::TestEISSimulationSolve::test_three_electrode_impedances_sum_to_cell_impedance`