# Deprecate ScipySolver, CasadiSolver, CasadiAlgebraicSolver — steer all users to IDAKLUSolver

Date: 2026-06-17
Status: Design — awaiting approval

## Goal

Make `IDAKLUSolver` the unambiguous solver for PyBaMM users. Concretely:

1. Guarantee `IDAKLUSolver` works with every time-dependent (ODE/DAE) model PyBaMM ships,
   including the one legacy model (`lead_acid.LOQS`) that is currently forced onto
   `CasadiSolver` by a vestigial jacobian flag.
2. Emit a `DeprecationWarning` when a user constructs `ScipySolver`, `CasadiSolver`, or
   `CasadiAlgebraicSolver`, pointing them to `IDAKLUSolver`.
3. Remove the deprecated solvers from all user-facing docs (prose, tutorials, examples,
   recommendations) so the documentation only steers users to `IDAKLUSolver`.

This is a **deprecation, not a removal** — the three classes keep working this release. The
warning text uses the vague "will be removed in a future release" wording (no version
commitment).

## Scope

**In scope (deprecated):** `ScipySolver`, `CasadiSolver`, `CasadiAlgebraicSolver`.

**Out of scope (untouched):** `JaxSolver`, `AlgebraicSolver`, `NonlinearSolver`,
`CompositeSolver`, `DummySolver`. `NonlinearSolver` remains the correct default for
pure-algebraic models (IDAKLU is a time-integrator and cannot solve a pure-algebraic system),
and `CompositeSolver`/`AlgebraicSolver` remain the eSOH default. The claim "IDAKLU works with
all models" precisely means: **IDAKLU is the default for all ODE/DAE models; pure-algebraic
models legitimately stay on `NonlinearSolver`.**

## Background (verified)

- **IDAKLU is already the global default.** `models/base_model.py:709` returns
  `IDAKLUSolver()` for ODE/DAE models and `NonlinearSolver()` for pure-algebraic ones (changed
  in v25.4.0, commit `d88beaf27a`).
- **The one legacy holdout is `lead_acid.LOQS`.** `loqs.py:42` sets `use_jacobian = False`
  (when `dimensionality == 0`); `loqs.py:71` overrides `default_solver` to return
  `CasadiSolver()`. IDAKLU hard-requires a jacobian (`idaklu_solver.py:408`:
  `raise pybamm.SolverError("KLU requires the Jacobian")`), so LOQS cannot use it today.
  - Archaeology + numerics confirm `use_jacobian = False` is a **2019 performance default**
    (symbolic jacobian construction was costly then), *not* a singularity workaround. With the
    jacobian enabled, IDAKLU solves LOQS cleanly; the iteration matrix `J − cj·M` is full rank
    and well-conditioned (cond ≈ 1.05); results match `CasadiSolver` to ~2e-6 V over a 1-hour
    discharge. The `default_solver` override exists *only* because of the disabled jacobian.
- **Deprecation convention:** plain `warnings.warn(msg, DeprecationWarning, stacklevel=2)`.
  No custom helper exists; mirror the existing pattern (e.g. `symbol.py:1112`,
  `base_simulation.py:152`).
- **Only two internal construction sites** of the deprecated classes exist in `src/`:
  1. `loqs.py:72` — `CasadiSolver()` default (eliminated by the LOQS fix below).
  2. `base_solver.py:190` — `CasadiAlgebraicSolver(self.root_tol)`, reached **only** when a
     user explicitly passes `root_method="casadi"`. No internal default supplies `"casadi"`.
  `ScipySolver` is never constructed internally. The remaining `src/` hits are `isinstance`
  checks, docstrings, and error strings — not constructions.
- **Test/example blast radius:** dedicated `test_casadi_solver.py` (~34 sites) and
  `test_scipy_solver.py` (~19 sites); ~127 instantiations total across ~17 files; parametrized
  solver lists in `test_serialisation.py` / `test_round_trip.py`; 4 example scripts. The suite
  runs `-W error`, so any unhandled `DeprecationWarning` fails the suite.

## Approaches considered

**Deprecation mechanism** — chosen: a `warnings.warn(..., DeprecationWarning, stacklevel=2)`
on the first line of each class's `__init__`, matching the repo convention. (Rejected: a new
`pybamm.deprecate` helper — YAGNI, one doesn't exist and three call sites don't justify it.)

**API reference pages** (`docs/source/api/solvers/casadi_solver.rst`, `scipy_solver.rst`) —
the classes still exist and remain callable, so their autodoc pages should not 404. Two options:

- **(A — recommended)** Keep the API stub pages, add a Sphinx `.. deprecated::` note to each
  class docstring (autodoc renders it as a banner), and remove the solvers from all *prose,
  tutorials, comparisons, and recommendations*. This satisfies "users are always steered to
  IDAKLU" without breaking links or hiding API docs from people still on the old solvers.
- **(B)** Delete the `.rst` pages and their `index.rst` toctree entries entirely. Maximally
  "removes references" but leaves a public, callable class with no API page (and a toctree that
  no longer documents an exported symbol).

Recommending **A**. `CasadiAlgebraicSolver` is documented under the shared
`algebraic_solvers.rst` page — same treatment (add a `.. deprecated::` note, don't delete).

## Design — phased

### Phase 0 — Make IDAKLU work with every ODE/DAE model (the prerequisite)

1. In `loqs.py`: delete the `if self.options["dimensionality"] == 0: self.use_jacobian = False`
   block (lines 41-42) and delete the `default_solver` override (lines 70-72) so LOQS inherits
   the global `IDAKLUSolver()` default.
2. Update lead-acid tests that assumed the `CasadiSolver` default (the ~5 integration tests
   that build a LOQS `Simulation` without a solver and implicitly got Casadi). They should now
   pass with IDAKLU; adjust any solver-specific assertions/tolerances.
3. **Coverage sweep test:** add a parametrized test that instantiates each shipped ODE/DAE
   model (lithium_ion `SPM`/`SPMe`/`DFN`/`MPM`/`MSMR`, the `Basic*` models, `lead_acid.Full`/
   `BasicFull`/`LOQS`, `equivalent_circuit.Thevenin`, `sodium_ion`, `lithium_metal`) and solves
   a short sim with `IDAKLUSolver`, asserting it runs. This is the concrete, testable form of
   "IDAKLU works with all current models." Sweep LOQS across a few operating modes
   (current/voltage/power, surface-form algebraic/differential) since those change the
   algebraic-equation count.

### Phase 1 — Add deprecation warnings

1. Add `warnings.warn("pybamm.X is deprecated and will be removed in a future release. Use "
   "pybamm.IDAKLUSolver instead.", DeprecationWarning, stacklevel=2)` as the first statement of
   `ScipySolver.__init__`, `CasadiSolver.__init__`, and `CasadiAlgebraicSolver.__init__`.
2. Suppress the one internal construction at `base_solver.py:190` (the `root_method == "casadi"`
   branch) with `with warnings.catch_warnings(): warnings.simplefilter("ignore",
   DeprecationWarning)` — the user passed the string `"casadi"`, not the class, so PyBaMM should
   not emit the warning on their behalf.
3. Add three focused `pytest.warns(DeprecationWarning, match=...)` tests, one per class.

### Phase 2 — Migrate tests and example scripts to IDAKLU

1. **General test sites** (tests that just need *a* working solver, not solver-specific
   behaviour — e.g. `test_solution.py`, `test_summary_variables.py`,
   `test_simulation_with_experiment.py`): switch `CasadiSolver()`/`ScipySolver()` →
   `IDAKLUSolver()`. Adjust numeric tolerances/assertions where the solver change shifts values.
2. **Dedicated solver suites** (`test_casadi_solver.py`, `test_scipy_solver.py`, and a
   `CasadiAlgebraicSolver` test if present): keep them — these still exercise the deprecated
   classes until removal — but add a module-level
   `pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")` so `-W error` doesn't
   fail on the now-expected warning.
3. **Parametrized serialisation/round-trip lists** (`test_serialisation.py`,
   `test_round_trip.py`): serialisation must keep working for the deprecated classes, so keep
   them in the lists but apply the same `filterwarnings` suppression rather than removing them.
4. **Example scripts** (`custom_model.py`, `compare_dae_solver.py`, `heat_equation.py`,
   `create_model.py`): switch to `IDAKLUSolver()`. (`compare_dae_solver.py` is a Casadi-vs-other
   comparison — keep one deprecated reference there only if the script's purpose is the
   comparison; otherwise retarget to IDAKLU.)

### Phase 3 — Remove the deprecated solvers from the docs

1. `docs/source/user_guide/fundamentals/public_api.rst:45` — rewrite the "two main solvers"
   prose to recommend `IDAKLUSolver` only.
2. `docs/source/examples/notebooks/performance/03-pybamm-solvers.ipynb` — rewrite the solver
   list/comparison prose around `IDAKLUSolver`; drop the stale "will soon be replaced as the
   default" line (already done); remove `CasadiSolver` from the comparison loop or relabel it
   explicitly as a deprecated baseline.
3. ODE tutorials that teach `ScipySolver` (`creating_models/1-an-ode-model.ipynb`,
   `2-a-pde-model`, `3-negative-particle-problem`, `4-comparing-full-and-reduced-order-models`,
   `6-a-simple-SEI-model`, `7-creating-a-submodel`, `solvers/ode-solver.ipynb`,
   `models/unsteady-heat-equation.ipynb`, `parameterization/*`): switch code cells to
   `IDAKLUSolver()` and update surrounding prose. Confirm IDAKLU solves these bare custom
   ODE/PDE `BaseModel`s (it forces casadi-format conversion and needs the jacobian, which these
   models have by default).
4. `change-input-current.ipynb` — replace the "recommend CasadiSolver in fast mode for drive
   cycles" comment with IDAKLU guidance.
5. `models/lead-acid.ipynb` — switch the explicit `CasadiSolver()` cells to `IDAKLUSolver()`.
6. Stale **output cells** mentioning the old solvers ("Default solver for SPM model:
   CasadiSolver", "Solving using CasadiSolver solver...", the "default solver changed" warnings):
   regenerate by re-executing the edited notebooks (or clear outputs) so they don't reintroduce
   the names.
7. API pages: apply the recommended **Approach A** — add `.. deprecated::` notes to the three
   class docstrings; keep `casadi_solver.rst`, `scipy_solver.rst`, and the
   `CasadiAlgebraicSolver` entry in `algebraic_solvers.rst`.
8. `CHANGELOG.md` — one bullet under `# [Unreleased]` noting the deprecations + the LOQS/IDAKLU
   fix, ending with the PR link.

### Phase 4 — Verify

- `nox -s unit` and `nox -s integration` (full suites, `-W error`).
- `nox -s doctests`.
- Notebook execution (the notebook CI / nbmake path) on every edited notebook.
- Docs build (Sphinx) clean, including the `-W` intersphinx step (tolerate the known transient
  scipy.org flake per existing project convention).

## Risks

- **Test migration tolerances (highest-churn risk).** Swapping ~100 general test sites from
  Casadi/Scipy to IDAKLU can shift numbers enough to trip tight `assert_allclose` tolerances.
  Mitigation: migrate and run file-by-file; loosen tolerances only where physically justified.
- **LOQS operating-mode coverage.** The empirical check covered constant current; voltage/power
  and surface-form variants change the algebraic structure. Mitigation: the Phase 0 sweep
  exercises these explicitly.
- **Notebook output regeneration** is easy to forget and would silently re-leak the deprecated
  names. Mitigation: Phase 4 re-executes every edited notebook.
- **`compare_dae_solver.py`** may exist precisely to compare Casadi vs others; retargeting it
  blindly could defeat its purpose. Mitigation: read it before editing.

## Out of scope

- Removing the three classes (a later release).
- Touching `JaxSolver` or the algebraic/nonlinear/composite solvers.
- Changing `IDAKLUSolver` behaviour or defaults.
- Unrelated refactoring of the solver layer.
