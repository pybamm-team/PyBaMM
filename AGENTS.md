# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, Gemini CLI, Qwen, Copilot,
Cursor) when working with code in this repository. It is the canonical instructions file;
`CLAUDE.md`, `GEMINI.md`, `QWEN.md`, `.cursorrules`, and `.github/copilot-instructions.md` are
symlinks to it.

PyBaMM (Python Battery Mathematical Modelling) is a battery simulation package. It is a custom
computer algebra system for writing systems of (partial) differential equations, plus a library
of battery models, parameters, solvers, and post-processing tools.

## Environment and commands

The project is managed with `uv`. Run all Python and tooling through `uv run` (never bare
`python -m`).

```bash
uv sync --extra all --group dev        # create/refresh the dev environment
```

Testing uses `pytest`; `unit`/`integration`/`memory` markers are assigned automatically from a
test's path (see `conftest.py`), so select suites by marker. Default `addopts` run in parallel
(`-nauto`) and treat warnings as errors.

```bash
uv run --group dev pytest -m unit                              # full unit suite
uv run --group dev pytest -m integration                       # integration suite
uv run --group dev pytest tests/unit/test_solvers/test_solution.py   # one file
uv run --group dev pytest "tests/unit/test_plotting/test_quick_plot.py::TestQuickPlot::test_simple_ode_model"  # one test
nox -s unit | nox -s tests | nox -s integration | nox -s doctests   # sessions used in CI
```

Lint/format is Ruff via pre-commit; this is the source of truth for style.

```bash
uv run pre-commit run --all-files
```

## Architecture

PyBaMM solves models with the **Method of Lines**: a model is defined symbolically, discretised
in space, then handed to a time-stepping solver. The end-to-end pipeline (orchestrated by
`simulation.py`) is:

```
Model (symbolic) -> ParameterValues -> Geometry -> Mesh -> Discretisation -> Solver -> Solution
```

- **Expression tree** (`expression_tree/`) — the computer algebra core. Every equation is a tree
  of `Symbol` nodes (`Scalar`, `Variable`, `StateVector`, binary/unary operators, `Broadcast`,
  `Concatenation`, ...). `expression_tree/operations/` holds tree passes: conversion to
  evaluable backends (CasADi etc.), Jacobians, simplification, discretisation.
- **Models** (`models/`) — `base_model.py` defines a model as dicts: `rhs` (ODEs), `algebraic`,
  `boundary_conditions`, `initial_conditions`, `variables`, `events`. Full battery models
  (`full_battery_models/`, e.g. `lithium_ion.DFN`) are assembled by composing `submodels/`
  (particle, electrolyte, thermal, interface, ...); each submodel contributes equations for its
  physics. Presence of `rhs`/`algebraic` determines whether the system is an ODE, DAE, or
  algebraic system.
- **Parameters** (`parameters/`) — symbolic `Parameter`/`FunctionParameter` nodes; concrete
  values come from `ParameterValues`, which substitutes them into the tree. Parameter sets and
  named models are also discoverable via entry points (`dispatch/`, see `pyproject.toml`).
- **Geometry / Meshes / Spatial methods** — `discretisations/discretisation.py` walks the tree
  and replaces spatial operators with matrices and `Variable`s with `StateVector`s, using the
  `spatial_methods/` for each domain (finite volume is the default; spectral volume and
  scikit-fem elements also exist).
- **Solvers** (`solvers/`) — wrap third-party integrators (`casadi_solver`, `idaklu_solver`,
  `scipy_solver`, `jax_*`). They consume the discretised model and return a `Solution`;
  `processed_variable*.py` turns raw solver output into the named, interpolatable variables users
  access via `solution["variable name"].data`.
- **Experiment / Simulation** (`experiment/`, `simulation.py`) — `Experiment` parses
  English-like step instructions; `Simulation` runs the pipeline, drives multi-step experiments,
  and computes summary variables.

## Writing model and expression-tree code

Equations are symbolic trees, not numbers. These idioms are the ones generated code most often
gets wrong:

- **Symbols are not Python values.** Never use `if`, `and`, `or`, `max`, `min`, or `==`-for-control
  on a `Symbol`. Use `pybamm.maximum`/`pybamm.minimum`, and `pybamm.sigmoid`/`pybamm.smooth_min`
  for differentiable switches.
- **State variables need domains.** `pybamm.Variable(name, domain=...,
  auxiliary_domains={"secondary": "current collector"})`; omitting them breaks discretisation.
  Distinguish `Variable` (unknown state) from `Parameter`/`FunctionParameter` (substituted by
  `ParameterValues`) and `Scalar` (literal).
- **Spatial operators act on the primary domain**: `pybamm.grad`, `pybamm.div`, `pybamm.surf`,
  `pybamm.boundary_value(sym, "left"|"right")`, `pybamm.x_average`. Lift lower-dimensional symbols
  with `PrimaryBroadcast`/`SecondaryBroadcast`/`FullBroadcast`.
- **Variable dict keys are the public API.** Use `"Capitalized description [unit]"` (e.g.
  `"Terminal voltage [V]"`); a typo silently breaks `solution[...]`. Units are documentation only —
  there is no unit checking, so keep the algebra dimensionally consistent yourself.
- **Submodels** subclass `BaseSubModel` and override only what they need:
  `get_fundamental_variables` → `get_coupled_variables(variables)` →
  `set_rhs`/`set_algebraic`/`set_boundary_conditions`/`set_initial_conditions`/`set_events`. The
  setters mutate dicts, e.g. `self.rhs = {var: expr}`; boundary conditions use
  `{var: {"left": (expr, "Neumann"), "right": (expr, "Dirichlet")}}`.
- **Reach parameters** via `self.param.X` (global), `self.domain_param.X` (per electrode, keyed by
  the first word, e.g. `"negative"`), and `self.phase_param.X` (per phase).
- **Register a citation** for new physics: `pybamm.citations.register("Author2021")`,
  conventionally in `__init__`.

## Code style

- Standard scientific-Python style (PEP 8, NumPy/SciPy idioms). Ruff lint+format is authoritative;
  do not hand-fight it or manually wrap lines to dodge warnings.
- **Inline comments must be concise — never more than two lines.** Comment the non-obvious *why*,
  not the *what*.
- **Docstrings are concise and follow the NumPy convention** (`Parameters`/`Returns`/`Raises`
  sections). Document only the object itself; do not describe callers, related code, or
  surrounding behaviour.
- Naming is descriptive: prefer full words over abbreviations (`mean`, not `mu`); avoid
  abbreviating class/argument names.
- Imports are absolute (relative imports are linted out), ruff-sorted (don't reorder by hand), and
  use Python 3.10+ syntax (`X | Y`, `dict[...]`). No bare `assert` in `src/`; validate and raise.
- Type-hint public functions; start new modules with `from __future__ import annotations`, type
  arrays as `npt.NDArray[np.float64]`, and reuse the aliases in `type_definitions.py`.
- **Raise PyBaMM's own exceptions for framework errors** — `OptionError`, `ModelError`,
  `DomainError`, `SolverError`, `GeometryError`, `DiscretisationError`, `ShapeError`
  (`expression_tree/exceptions.py`) — not bare `ValueError`/`RuntimeError`. Log through
  `pybamm.logger`, never `print`.
- **Import optional dependencies inside the function that uses them, never at module level** — via
  `pybamm.import_optional_dependency(...)` or guarded by `pybamm.has_jax()`. A top-level
  `import matplotlib` breaks `import pybamm` for minimal installs.
- Public, user-facing objects are re-exported through `src/pybamm/__init__.py` (users write
  `pybamm.X`) and get a `docs/source/api/*.rst` entry.
- Every feature or fix adds a `CHANGELOG.md` bullet under `# [Unreleased]` (Keep a Changelog
  format), ending with the PR link, e.g. `([#1234](https://github.com/pybamm-team/PyBaMM/pull/1234))`.

### Tests

- Write idiomatic, well-tested code: every feature or fix ships with tests, matching the structure
  of the nearest existing test. Tests under `tests/unit/` mirror the `src/pybamm/` layout.
- Tests are class-based (`class TestX:` with `test_*` methods). Mirroring the path assigns the
  `unit`/`integration`/`memory` marker automatically — never add `@pytest.mark.unit` by hand.
- Compare numbers with `np.testing.assert_allclose(a, b, rtol=, atol=)` (floats),
  `np.testing.assert_array_equal` (exact), or `pytest.approx` (scalars) — never `==` on floats.
  Assert on errors with `pytest.raises(pybamm.SomeError, match=r"...")`.
- Skip optional-backend tests with `@pytest.mark.skipif(not pybamm.has_jax(), reason=...)`.
- The suite is strict: warnings are errors, `xfail_strict` is on, and it runs in parallel
  (`-nauto`). Suppress expected warnings explicitly, and drop the marker from an `xfail` that
  starts passing.
- Reuse builders in `tests/shared.py` (`get_discretisation_for_testing`, `assert_domain_equal`, …);
  property-based tests use Hypothesis strategies in `tests/strategies/`.
