# Electrode SOH Model Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `ElectrodeSOHSolver` wrapper with `ElectrodeSOHModel` and `CompositeElectrodeSOHModel` — `pybamm.BaseModel` subclasses that carry their own equations, bounds, initial conditions and feasibility logic, solved by the ordinary solver stack.

**Architecture:** The electrode SOH problem is already a small algebraic system, and `_ElectrodeSOH` is already a `BaseModel`. Everything else — a 1,127-line wrapper of caches, retry ladders, bracket inputs and a bespoke `Brent` rootfinder node — exists because that model was not given what `BaseModel` already supports: `bounds` on its variables and a sensible `initial_conditions`. This plan puts the logic in the model, deletes the wrapper, and keeps `ElectrodeSOHSolver` only as a deprecated alias.

**Tech Stack:** PyBaMM expression tree, `pybamm.BaseModel`, `pybamm.NonlinearSolver`, `pybamm.Simulation`, pytest.

---

## Context an engineer needs before starting

**Run everything through `pbwt`**, a wrapper on `PATH` that runs the current worktree's `pybamm` source against the main repo's `.venv`. Never `uv run` — it re-syncs and silently reinstalls `pybammsolvers` from `main`, which reverts any C++ build.

```bash
pbwt pytest -q -p no:randomly -m unit packages/pybamm/tests
```

**Test markers** (`unit`/`integration`) are assigned automatically from a test's path. Never add `@pytest.mark.unit` by hand. Warnings are errors and the suite runs in parallel.

**Baseline on this branch, measured 2026-08-11.** Anything worse than this is a regression:

| check | expected |
|---|---|
| `pbwt pytest -m unit packages/pybamm/tests` | 11 failed, 4225 passed |
| `test_esoh_convergence.py` | 14 passed |
| `-m integration -k "composite or esoh"` | 50 passed |

The 11 unit failures are pre-existing (MSMR paths, `get_initial_ocps`, `test_solve_with_initial_soc`). Do not try to fix them here; only make sure the count does not grow.

**Known separate bug, do not fix in this plan.** `Chayambuka2022` raises at every entry point on this branch (`get_initial_stoichiometries` 0/7 vs 7/7 on `main`) because `_ocp_domain` now reports its positive OCP table starting at `y = 0.21`, and `_check_esoh_feasible` then finds `V_lower_bound = 2.3085 V > V_min = 2.0 V`. Task 9 pins this as an xfail so the refactor is not blamed for it.

**External callers that must keep working** — these are the whole compatibility surface:

| file:line | uses |
|---|---|
| `simulation/base_simulation.py:229` | `ElectrodeSOHSolver(pv, direction=, param=, options=)` |
| `simulation/base_simulation.py:598` | same |
| `simulation/base_simulation.py:247` | `ElectrodeSOHComposite(options, direction, initialization_method=)` |
| `solvers/summary_variable.py:107` | `esoh_solver._get_electrode_soh_sims_full(direction)` — **private, but called** |
| `solvers/summary_variable.py:190` | `esoh_solver.solve(inputs=all_inputs)` |
| `lithium_ion/initial_state.py:96,155,209` | `get_initial_ocps`, `get_initial_stoichiometries_composite`, `get_initial_stoichiometries` |

---

## File Structure

- **Create** `packages/pybamm/src/pybamm/models/full_battery_models/lithium_ion/electrode_soh_model.py` — `ElectrodeSOHModel`: equations, bounds, initial conditions, limits, feasibility, and the `solve` / `get_*` API. One responsibility: *be* the electrode SOH problem.
- **Create** `packages/pybamm/tests/unit/test_models/test_full_battery_models/test_lithium_ion/test_electrode_soh_model.py` — unit tests for the new model.
- **Create** `packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_no_silent_failures.py` — the 2,310-solve sweep as a real test.
- **Modify** `.../lithium_ion/electrode_soh.py` — strip the Brent apparatus, then reduce `ElectrodeSOHSolver` to a deprecated alias.
- **Modify** `.../lithium_ion/electrode_soh_composite.py` — `CompositeElectrodeSOHModel`.
- **Modify** `.../lithium_ion/__init__.py`, `simulation/base_simulation.py`, `solvers/summary_variable.py`, `CHANGELOG.md`, `docs/source/api/models/lithium_ion/electrode_soh.rst`.

---

## Task 1: Prove the Brent apparatus is unnecessary

This is the gate. If the sweep is not clean without `Brent`, stop and report before deleting anything.

**Files:**
- Create: `packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_no_silent_failures.py`

- [ ] **Step 1: Write the sweep as a failing-if-wrong test**

```python
#
# The electrode SOH solve must never report success with an answer that does not
# satisfy its own equations. This is the guard on the whole subsystem.
#
import numpy as np
import pytest

import pybamm

PARAMETER_SETS = [
    "Ai2020", "Chayambuka2022", "Chen2020", "Ecker2015", "Marquis2019",
    "Mohtat2020", "NCA_Kim2011", "OKane2022", "ORegan2022", "Prada2013",
    "Ramadass2004",
]
CAPACITY_STATES = [(1.0, 1.0), (0.8, 1.0), (1.0, 0.8), (1.2, 0.9), (0.9, 1.2),
                   (0.7, 1.3), (1.3, 0.7)]
INVENTORIES = 30
RESIDUAL_TOLERANCE = 1e-6


def _residuals(solver, solution, inputs):
    """The five equations the answer must satisfy, in volts and A.h."""
    parameter_values, param = solver.parameter_values, solver.param
    V_max = float(parameter_values.evaluate(param.ocp_soc_100))
    V_min = float(parameter_values.evaluate(param.ocp_soc_0))
    x_100, x_0 = float(solution["x_100"]), float(solution["x_0"])
    y_100, y_0 = float(solution["y_100"]), float(solution["y_0"])
    return {
        "V_max": float(solution["Up(y_100) - Un(x_100)"]) - V_max,
        "V_min": float(solution["Up(y_0) - Un(x_0)"]) - V_min,
        "Q_Li": x_100 * inputs["Q_n"] + y_100 * inputs["Q_p"] - inputs["Q_Li"],
        "Q_n": inputs["Q_n"] * (x_100 - x_0) - float(solution["Q"]),
        "Q_p": inputs["Q_p"] * (y_0 - y_100) - float(solution["Q"]),
    }


class TestElectrodeSOHNoSilentFailures:
    @pytest.mark.parametrize("parameter_set", PARAMETER_SETS)
    def test_a_returned_answer_always_satisfies_the_equations(self, parameter_set):
        parameter_values = pybamm.ParameterValues(parameter_set)
        solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values)
        param = solver.param
        Q_n = float(parameter_values.evaluate(param.n.Q_init))
        Q_p = float(parameter_values.evaluate(param.p.Q_init))
        x0_min, x100_max, y100_min, y0_max = solver.lims_ocp

        defects = []
        for scale_n, scale_p in CAPACITY_STATES:
            inputs = {"Q_n": Q_n * scale_n, "Q_p": Q_p * scale_p}
            low = inputs["Q_n"] * x0_min + inputs["Q_p"] * y100_min
            high = inputs["Q_n"] * x100_max + inputs["Q_p"] * y0_max
            for Q_Li in np.linspace(low, high, INVENTORIES + 2)[1:-1]:
                request = {**inputs, "Q_Li": float(Q_Li)}
                try:
                    solution = solver.solve(dict(request))
                except (pybamm.SolverError, ValueError):
                    continue  # refusing is always allowed; answering wrongly is not
                residuals = _residuals(solver, solution, request)
                worst = max(abs(value) for value in residuals.values())
                stoichiometries = [
                    float(solution[name])
                    for name in ("x_0", "x_100", "y_0", "y_100")
                ]
                if not all(-1e-9 <= v <= 1 + 1e-9 for v in stoichiometries):
                    defects.append((request["Q_Li"], "outside [0, 1]", stoichiometries))
                elif not np.isfinite(worst) or worst > RESIDUAL_TOLERANCE:
                    defects.append((request["Q_Li"], "residual", worst))

        assert not defects, (
            f"{len(defects)} of {len(CAPACITY_STATES) * INVENTORIES} answers for "
            f"{parameter_set} do not satisfy the equations: {defects[:3]}"
        )
```

- [ ] **Step 2: Run it and record the baseline**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_no_silent_failures.py
```

Expected: `11 passed`. Write the number down. On `main` this same sweep produces 320 defects.

- [ ] **Step 3: Commit**

```bash
git add packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_no_silent_failures.py
git commit -m "test: guard the electrode SOH solve against silently wrong answers"
```

- [ ] **Step 4: Turn the Brent path off without deleting it**

In `packages/pybamm/src/pybamm/models/full_battery_models/lithium_ion/electrode_soh.py`, in `_ElectrodeSOH.__init__`, replace the body of `solve_for_limit` so it returns a plain algebraic variable instead of a `Brent` node. Find:

```python
            return pybamm.Brent(
                residual,
                unknown,
                bracket,
                max_expansions=max_expansions,
                name=name,
            )
```

Replace the whole `solve_for_limit` helper and its call sites with the equation form `main` uses. That is, delete `solve_for_limit`, delete the `bracket = (...)` block above it, and restore:

```python
        # Define variables for 100% state of charge
        if "x_100" in solve_for:
            x_100 = pybamm.Variable("x_100", bounds=(0, 1))
            if known_value == "cyclable lithium capacity":
                y_100 = (Q_Li - x_100 * Q_n) / Q_p
            elif known_value == "cell capacity":
                y_100 = pybamm.Variable("y_100", bounds=(0, 1))
                Q_Li = y_100 * Q_p + x_100 * Q_n
        else:
            x_100 = pybamm.InputParameter("x_100")
            y_100 = pybamm.InputParameter("y_100")
```

and, at the point where the 100% equation is set:

```python
        if "x_100" in solve_for:
            self.algebraic[x_100] = Up_100 - Un_100 - V_max
            self.initial_conditions[x_100] = pybamm.Scalar(0.9)
```

Use `git show main:packages/pybamm/src/pybamm/models/full_battery_models/lithium_ion/electrode_soh.py` as the reference for the exact `x_0` equations too — copy them verbatim, they are correct.

- [ ] **Step 5: Make `solve` stop short-circuiting**

In `ElectrodeSOHSolver.solve`, delete the `_evaluate` branch so every request goes through the real solve:

```python
    def solve(self, inputs, direction=None):
        if self.known_value == "cyclable lithium capacity":
            x_min, x_max, _, _ = self._get_lims(inputs)
            inputs = {**inputs, "x_min": x_min, "x_max": x_max}
        ics = self._set_up_solve(inputs, direction)
        ...
```

becomes, with the bracket inputs no longer needed by any equation:

```python
    def solve(self, inputs, direction=None):
        ics = self._set_up_solve(inputs, direction)
        ...
```

- [ ] **Step 6: Re-run the gate**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_no_silent_failures.py
```

Expected: `11 passed`, the same as Step 2.

**If any parameter set now fails, STOP.** Report which sets and how many defects, and do not continue to Task 2 — that is the evidence that the bracket was load-bearing after all.

- [ ] **Step 7: Run the rest**

```bash
pbwt pytest -q -p no:randomly -m unit packages/pybamm/tests
pbwt pytest -q -p no:randomly packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_convergence.py
```

Expected: no more than 11 unit failures; 14 passed.

- [ ] **Step 8: Commit**

```bash
git add packages/pybamm/src/pybamm/models/full_battery_models/lithium_ion/electrode_soh.py
git commit -m "refactor: solve the electrode SOH limits as algebraic equations again"
```

---

## Task 2: Give `NonlinearSolver` the bounds the model declares

`pybamm.Variable(..., bounds=(0, 1))` is currently ignored — the word `bounds` does not appear in `packages/pybamm/src/pybamm/solvers/nonlinear_solver.py`. The bracket was doing this job, so the model needs it back before the bracket goes.

**Files:**
- Modify: `packages/pybamm/src/pybamm/solvers/nonlinear_solver.py`
- Test: `packages/pybamm/tests/unit/test_solvers/test_nonlinear_solver.py`

- [ ] **Step 1: Write the failing test**

Append to `packages/pybamm/tests/unit/test_solvers/test_nonlinear_solver.py`:

```python
class TestNonlinearSolverBounds:
    def test_the_iterate_stays_inside_the_declared_bounds(self):
        # x^2 = 4 has roots at -2 and +2; the bounds pick one
        model = pybamm.BaseModel()
        x = pybamm.Variable("x", bounds=(0, 10))
        model.algebraic = {x: x**2 - 4}
        model.initial_conditions = {x: pybamm.Scalar(9)}
        model.variables = {"x": x}
        solution = pybamm.NonlinearSolver().solve(model, [0])
        np.testing.assert_allclose(solution["x"].data[0], 2.0, rtol=1e-8)

    def test_a_guess_outside_the_bounds_is_pulled_in(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x", bounds=(0, 1))
        model.algebraic = {x: x - 0.25}
        model.initial_conditions = {x: pybamm.Scalar(5)}
        model.variables = {"x": x}
        solution = pybamm.NonlinearSolver().solve(model, [0])
        np.testing.assert_allclose(solution["x"].data[0], 0.25, rtol=1e-8)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/unit/test_solvers/test_nonlinear_solver.py -k Bounds
```

Expected: FAIL — either the wrong root (`-2.0`) or a solver error.

- [ ] **Step 3: Read how bounds already reach a solver**

Discretisation sets `model.bounds` to a `(lower, upper)` pair of arrays over the state
vector (`discretisations/discretisation.py:214`), and `AlgebraicSolver` already consumes
it. Read that first and copy the pattern — do not invent a new channel:

```bash
grep -n "bounds" packages/pybamm/src/pybamm/solvers/algebraic_solver.py
```

- [ ] **Step 4: Clamp the iterate in the Newton loop**

In `nonlinear_solver.py`, inside the iteration, after the step is taken and before the residual is re-evaluated, project onto the box:

```python
        lower, upper = model.bounds
        y = np.clip(y, lower, upper)
```

Clip the initial guess the same way before the first residual evaluation. Projection is the whole change — no line-search rework.

- [ ] **Step 5: Run the test to verify it passes**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/unit/test_solvers/test_nonlinear_solver.py
```

Expected: PASS, and the pre-existing `test_matches_casadi_algebraic_solver` failure unchanged.

- [ ] **Step 6: Commit**

```bash
git add packages/pybamm/src/pybamm/solvers/nonlinear_solver.py packages/pybamm/tests/unit/test_solvers/test_nonlinear_solver.py
git commit -m "feat: honour variable bounds in NonlinearSolver"
```

- [ ] **Step 7: Route the model through `NonlinearSolver` alone**

`get_esoh_default_solver` currently returns a four-deep retry ladder:

```python
    return pybamm.CompositeSolver(
        [
            pybamm.NonlinearSolver(atol=tol, rtol=0, step_tol=0, max_backtracks=100),
            pybamm.AlgebraicSolver(tol=tol),
            pybamm.AlgebraicSolver(method="lsq", tol=tol),
            pybamm.AlgebraicSolver(method="minimize", tol=tol),
        ]
    )
```

With bounds honoured, the fallbacks should be unnecessary. Replace it with:

```python
def get_esoh_default_solver(tol: float = 1e-6) -> pybamm.NonlinearSolver:
    return pybamm.NonlinearSolver(atol=tol, rtol=0, step_tol=0, max_backtracks=100)
```

- [ ] **Step 8: Re-run the gate and decide**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_no_silent_failures.py
pbwt pytest -q -p no:randomly -m unit packages/pybamm/tests
```

Expected: `11 passed`, and no more than 11 unit failures.

**If either regresses, revert Step 7 only** and keep the `CompositeSolver`, recording in
the commit message which parameter sets needed a fallback. The rest of the plan does not
depend on this step.

- [ ] **Step 9: Commit**

```bash
git add packages/pybamm/src/pybamm/models/full_battery_models/lithium_ion/electrode_soh.py
git commit -m "refactor: solve the electrode SOH model with NonlinearSolver alone"
```

---

## Task 3: Delete the Brent apparatus from the electrode SOH path

Only after Task 1's gate passed.

**Files:**
- Modify: `.../lithium_ion/electrode_soh.py`
- Modify: `.../lithium_ion/electrode_soh_composite.py`
- Modify: `packages/pybamm/tests/unit/test_models/test_full_battery_models/test_lithium_ion/test_electrode_soh.py`

- [ ] **Step 1: Remove the members that only existed for Brent**

From `ElectrodeSOHSolver`, delete `_evaluate`, `__get_evaluator`, `_both_ocps_provably_decreasing`, the `self.max_expansions = ...` line in `__init__`, and the two `_get_evaluator` lines in `__getstate__`/`__setstate__`. From `_ElectrodeSOH.__init__`, delete the `max_expansions=0` parameter. From `__get_electrode_soh_sims_full` and `__get_electrode_soh_sims_split`, delete every `max_expansions=self.max_expansions,` argument.

- [ ] **Step 2: Remove the composite gate**

In `electrode_soh_composite.py`, delete `_secondary_stoichiometry_solver` and the block in `_solve_secondary_stoichiometry` that calls it:

```python
    solve = _secondary_stoichiometry_solver(
        parameter_values, param, electrode, lith_sec, T
    )
    if solve is not None:
        U_target = parameter_values.process_symbol(U_prim).evaluate(
            inputs={"z_1": primary_stoich}
        )
        return float(solve(U_target, primary_stoich))
```

Keep `_primary_solver` and its cache — that is a plain performance win, unrelated to Brent.

- [ ] **Step 3: Delete the tests that only covered the deleted path**

Remove `TestCompositeSecondaryStoichiometry` and `TestCompositeLinearInterpolants::test_it_agrees_with_the_analytic_set` from `test_electrode_soh.py`. Keep `TestCompositeLinearInterpolants::test_every_phase_becomes_provable_at_the_reference_temperature` — it tests `U_is_strictly_decreasing`, which survives.

- [ ] **Step 4: Run everything**

```bash
pbwt pytest -q -p no:randomly -m unit packages/pybamm/tests
pbwt pytest -q -p no:randomly -m integration packages/pybamm/tests -k "composite or esoh"
```

Expected: no more than 11 unit failures; 50 integration passed.

- [ ] **Step 5: Commit**

```bash
git add -A packages/pybamm
git commit -m "refactor: drop the bracketed rootfind from the electrode SOH path"
```

---

## Task 4: `ElectrodeSOHModel`

The model gains the logic. `ElectrodeSOHSolver` keeps working by delegating.

**Files:**
- Create: `.../lithium_ion/electrode_soh_model.py`
- Test: `packages/pybamm/tests/unit/test_models/test_full_battery_models/test_lithium_ion/test_electrode_soh_model.py`

- [ ] **Step 1: Write the failing test**

```python
#
# Tests for ElectrodeSOHModel
#
import numpy as np
import pytest

import pybamm


class TestElectrodeSOHModel:
    def test_it_is_a_model_and_solves_itself(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        model = pybamm.lithium_ion.ElectrodeSOHModel(parameter_values)
        assert isinstance(model, pybamm.BaseModel)

        Q_n = float(parameter_values.evaluate(model.param.n.Q_init))
        Q_p = float(parameter_values.evaluate(model.param.p.Q_init))
        Q_Li = float(parameter_values.evaluate(model.param.Q_Li_particles_init))
        solution = model.solve({"Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li})

        for name in ("x_0", "x_100", "y_0", "y_100"):
            assert 0 <= float(solution[name]) <= 1

    def test_the_answer_satisfies_the_voltage_limits(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        model = pybamm.lithium_ion.ElectrodeSOHModel(parameter_values)
        Q_n = float(parameter_values.evaluate(model.param.n.Q_init))
        Q_p = float(parameter_values.evaluate(model.param.p.Q_init))
        Q_Li = float(parameter_values.evaluate(model.param.Q_Li_particles_init))
        solution = model.solve({"Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li})

        V_max = float(parameter_values.evaluate(model.param.ocp_soc_100))
        V_min = float(parameter_values.evaluate(model.param.ocp_soc_0))
        np.testing.assert_allclose(
            float(solution["Up(y_100) - Un(x_100)"]), V_max, atol=1e-6
        )
        np.testing.assert_allclose(
            float(solution["Up(y_0) - Un(x_0)"]), V_min, atol=1e-6
        )

    def test_an_infeasible_request_raises(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        model = pybamm.lithium_ion.ElectrodeSOHModel(parameter_values)
        Q_n = float(parameter_values.evaluate(model.param.n.Q_init))
        Q_p = float(parameter_values.evaluate(model.param.p.Q_init))
        with pytest.raises(ValueError, match="outside the range of possible values"):
            model.solve({"Q_n": Q_n, "Q_p": Q_p, "Q_Li": 100 * (Q_n + Q_p)})

    def test_the_old_solver_name_still_works_and_warns(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        with pytest.warns(DeprecationWarning, match="ElectrodeSOHModel"):
            solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values)
        assert isinstance(solver, pybamm.lithium_ion.ElectrodeSOHModel)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/unit/test_models/test_full_battery_models/test_lithium_ion/test_electrode_soh_model.py
```

Expected: FAIL with `AttributeError: module 'pybamm.lithium_ion' has no attribute 'ElectrodeSOHModel'`.

- [ ] **Step 3: Create the model by moving, not rewriting**

Create `electrode_soh_model.py` containing `class ElectrodeSOHModel(_BaseElectrodeSOH)`. Move these members off `ElectrodeSOHSolver` unchanged — cut and paste, do not retype:

`__init__`, `__getstate__`, `__setstate__`, `_ocp_domain`, `_get_lims_ocp`, `__get_electrode_soh_sims_full`, `__get_electrode_soh_sims_split`, `solve`, `_set_up_solve`, `_solve_full`, `_solve_split`, `_get_lims`, `_check_esoh_feasible`, `get_initial_stoichiometries`, `get_min_max_stoichiometries`, `get_initial_ocps`, `get_min_max_ocps`, `__get_energy_ocv_function`, `theoretical_energy_integral`.

`__init__` must call `super().__init__()` first so the `BaseModel` half is initialised, then keep its existing body:

```python
class ElectrodeSOHModel(_BaseElectrodeSOH):
    def __init__(
        self,
        parameter_values,
        direction=None,
        param=None,
        known_value="cyclable lithium capacity",
        options=None,
    ):
        super().__init__()
        self.parameter_values = parameter_values
        self.param = param or pybamm.LithiumIonParameters(options)
        ...
```

- [ ] **Step 4: Make `ElectrodeSOHSolver` a deprecated alias**

In `electrode_soh.py`, replace the class with:

```python
def ElectrodeSOHSolver(*args, **kwargs):
    """Deprecated. Use :class:`pybamm.lithium_ion.ElectrodeSOHModel`."""
    warnings.warn(
        "ElectrodeSOHSolver is deprecated, use ElectrodeSOHModel instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return ElectrodeSOHModel(*args, **kwargs)
```

- [ ] **Step 5: Export it**

In `.../lithium_ion/__init__.py`, add `ElectrodeSOHModel` to the `from .electrode_soh_model import (...)` block and keep `ElectrodeSOHSolver` exported from `.electrode_soh`.

- [ ] **Step 6: Run the test to verify it passes**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/unit/test_models/test_full_battery_models/test_lithium_ion/test_electrode_soh_model.py
```

Expected: PASS, 4 tests.

- [ ] **Step 7: Commit**

```bash
git add -A packages/pybamm
git commit -m "feat: add ElectrodeSOHModel, deprecating ElectrodeSOHSolver"
```

---

## Task 5: Move the internal callers onto the model

**Files:**
- Modify: `packages/pybamm/src/pybamm/simulation/base_simulation.py:229,598`
- Modify: `packages/pybamm/src/pybamm/solvers/summary_variable.py:107,190`
- Modify: `packages/pybamm/src/pybamm/models/full_battery_models/lithium_ion/electrode_soh.py`

- [ ] **Step 1: Replace the three construction sites**

At `base_simulation.py:229` and `:598`, change `pybamm.lithium_ion.ElectrodeSOHSolver(` to `pybamm.lithium_ion.ElectrodeSOHModel(`. Arguments are unchanged.

- [ ] **Step 2: Give `summary_variable.py` a public entry point**

`summary_variable.py:107` reaches into the private `_get_electrode_soh_sims_full`. Add a public method to `ElectrodeSOHModel` and use it:

```python
    def simulation(self, direction=None):
        """The built simulation for this model, at a given equilibrium direction."""
        return self._get_electrode_soh_sims_full(direction)
```

Then at `summary_variable.py:107`:

```python
            esoh_model = self.esoh_solver.simulation(direction)
```

- [ ] **Step 3: Point the module-level functions at the model**

In `electrode_soh.py`, the four module functions `get_initial_stoichiometries`, `get_min_max_stoichiometries`, `get_initial_ocps`, `get_min_max_ocps` each construct an `ElectrodeSOHSolver`. Change each to construct `ElectrodeSOHModel` directly so calling them does not emit a deprecation warning.

- [ ] **Step 4: Run everything**

```bash
pbwt pytest -q -p no:randomly -m unit packages/pybamm/tests
```

Expected: no more than 11 failures, and no `DeprecationWarning` errors (warnings are errors in this suite — if one appears, an internal caller was missed).

- [ ] **Step 5: Commit**

```bash
git add -A packages/pybamm
git commit -m "refactor: move internal callers to ElectrodeSOHModel"
```

---

## Task 6: `CompositeElectrodeSOHModel`

**Files:**
- Modify: `.../lithium_ion/electrode_soh_composite.py`
- Modify: `.../lithium_ion/__init__.py`
- Test: `packages/pybamm/tests/unit/test_models/test_full_battery_models/test_lithium_ion/test_electrode_soh_model.py`

- [ ] **Step 1: Write the failing test**

Append to `test_electrode_soh_model.py`:

```python
class TestCompositeElectrodeSOHModel:
    OPTIONS = {"particle phases": ("2", "1")}

    def test_it_solves_and_the_phases_share_a_potential(self):
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        model = pybamm.lithium_ion.CompositeElectrodeSOHModel(
            parameter_values, options=self.OPTIONS
        )
        assert isinstance(model, pybamm.BaseModel)
        got = model.solve(0.5)
        for name in ("x_100_1", "x_0_1", "x_100_2", "x_0_2"):
            assert 0 <= got[name] <= 1

    def test_it_matches_the_function_it_replaces(self):
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        model = pybamm.lithium_ion.CompositeElectrodeSOHModel(
            parameter_values, options=self.OPTIONS
        )
        for soc in (0.2, 0.5, 0.8):
            reference = pybamm.lithium_ion.get_initial_stoichiometries_composite(
                soc, parameter_values, options=self.OPTIONS
            )
            got = model.solve(soc)
            for name in reference:
                np.testing.assert_allclose(got[name], reference[name], atol=1e-9)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/unit/test_models/test_full_battery_models/test_lithium_ion/test_electrode_soh_model.py -k Composite
```

Expected: FAIL with `AttributeError: ... has no attribute 'CompositeElectrodeSOHModel'`.

- [ ] **Step 3: Make the class hold the logic**

Rename `ElectrodeSOHComposite` to `CompositeElectrodeSOHModel`, give it `parameter_values` in `__init__`, and move these module functions onto it as methods, unchanged in body: `_get_primary_only_options`, `_get_stoich_variables`, `_get_initial_conditions`, `_get_direction`, `_get_prefix`, `_get_electrode_capacity_equation`, `_get_cyclable_lithium_equation`, `_solve_secondary_stoichiometry`, `_primary_solver`.

Turn the two `staticmethod`s `solve_split` and `solve_full` into instance methods, and add:

```python
    def solve(self, initial_value, direction=None, tol=1e-6, inputs=None):
        """Stoichiometries at `initial_value`, splitting the solve when it can."""
        try:
            return self.solve_split(initial_value, direction=direction, tol=tol,
                                    inputs=inputs)
        except (pybamm.SolverError, ValueError):
            return self.solve_full(initial_value, direction=direction, tol=tol,
                                   inputs=inputs)
```

Keep `ElectrodeSOHComposite` as a deprecated alias, in the same shape as Task 4 Step 4, and keep `get_initial_stoichiometries_composite` as a thin wrapper that constructs the model and calls `solve`.

- [ ] **Step 4: Run the test to verify it passes**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/unit/test_models/test_full_battery_models/test_lithium_ion/test_electrode_soh_model.py
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Update `base_simulation.py:247`**

```python
            model = pybamm.lithium_ion.CompositeElectrodeSOHModel(
                pv,
                options=options,
                direction=direction,
                initialization_method=initialization_method,
            )
```

- [ ] **Step 6: Run everything and commit**

```bash
pbwt pytest -q -p no:randomly -m unit packages/pybamm/tests
pbwt pytest -q -p no:randomly -m integration packages/pybamm/tests -k "composite or esoh"
git add -A packages/pybamm
git commit -m "feat: add CompositeElectrodeSOHModel, deprecating ElectrodeSOHComposite"
```

---

## Task 7: Delete `pybamm.Brent` if nothing else uses it

**Files:**
- Delete: `packages/pybamm/src/pybamm/expression_tree/brent.py`
- Delete: `packages/pybamm/tests/unit/test_expression_tree/test_brent.py`
- Delete: `packages/pybammsolvers/src/pybammsolvers/idaklu_source/brent.{hpp,cpp}`, `brent_impl.hpp`
- Delete: `packages/pybammsolvers/tests/test_brent_rootfinder.py`
- Modify: `packages/pybamm/src/pybamm/__init__.py`, `packages/pybamm/tests/strategies/symbols.py`, `packages/pybammsolvers/CMakeLists.txt`

- [ ] **Step 1: Check for remaining users**

```bash
grep -rn "Brent\|brent" packages/pybamm/src packages/pybammsolvers/src --include=*.py --include=*.cpp --include=*.hpp --include=*.txt | grep -v idaklu_source/brent
```

If `U_inverse` in `parameters/lithium_ion_parameters.py` is the only hit, decide with the user whether `U_inverse` stays. It is independent of electrode SOH and has its own tests.

- [ ] **Step 2: If it stays, stop here.** `Brent` is still needed and this task is done. Record that in the commit message for Task 6 and move to Task 8.

- [ ] **Step 3: If it goes, delete the files above**, remove `Brent`/`BrentUnknown` from `packages/pybamm/src/pybamm/__init__.py`, remove `_brent_branch` and the `pybamm.BrentUnknown` entry from `packages/pybamm/tests/strategies/symbols.py`, and remove the `brent.cpp` entry and the `brent_impl_source.hpp` generation rule from `packages/pybammsolvers/CMakeLists.txt`.

- [ ] **Step 4: Rebuild and run**

```bash
cd /Users/marcberliner/Documents/GitHub/PyBaMM
./.venv/bin/pip install --no-deps --force-reinstall --no-build-isolation \
  ./.claude/worktrees/electrode-soh-solver-params-c07ed2/packages/pybammsolvers
pbwt pytest -q -p no:randomly -m unit packages/pybamm/tests
```

Expected: no more than 11 failures.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove the Brent rootfinder plugin"
```

---

## Task 8: Documentation and changelog

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/source/api/models/lithium_ion/electrode_soh.rst`

- [ ] **Step 1: Add the changelog entries**

Under `# [Unreleased]` → `## Features`:

```markdown
- Added `ElectrodeSOHModel` and `CompositeElectrodeSOHModel`, which carry the electrode SOH equations, bounds and feasibility checks themselves. `ElectrodeSOHSolver` and `ElectrodeSOHComposite` are deprecated aliases.
- `NonlinearSolver` now honours the bounds declared on a model's variables.
```

- [ ] **Step 2: Add the API entries**

In `docs/source/api/models/lithium_ion/electrode_soh.rst`, add an `autoclass` block for each new class, matching the style of the existing `ElectrodeSOHSolver` block.

- [ ] **Step 3: Build the docs**

```bash
cd docs && pbwt sphinx-build -b html source _build/html 2>&1 | tail -5
```

Expected: no warnings about the new classes. Transient `scipy.org` intersphinx timeouts are known and tolerated.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md docs
git commit -m "docs: document the electrode SOH models"
```

---

## Task 9: Pin the Chayambuka2022 regression separately

This is not caused by the refactor, but it must not be silently carried.

**Files:**
- Modify: `packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_no_silent_failures.py`

- [ ] **Step 1: Add the test that documents it**

```python
    @pytest.mark.xfail(
        reason="_ocp_domain reports Chayambuka2022's positive OCP starting at "
        "y = 0.21, so _check_esoh_feasible finds a 2.3085 V floor against a "
        "2.0 V target and refuses every request. Works on main. See #TODO-issue.",
        strict=True,
    )
    def test_chayambuka2022_can_be_initialised(self):
        parameter_values = pybamm.ParameterValues("Chayambuka2022")
        for soc in (0.0, 0.5, 1.0):
            x, y = pybamm.lithium_ion.get_initial_stoichiometries(soc, parameter_values)
            assert 0 <= float(x) <= 1
            assert 0 <= float(y) <= 1
```

- [ ] **Step 2: Run it**

```bash
pbwt pytest -q -p no:randomly packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_no_silent_failures.py -k chayambuka
```

Expected: `1 xfailed`. `xfail_strict` is on, so if it starts passing the suite fails and the marker must be dropped.

- [ ] **Step 3: Open the issue and replace `#TODO-issue`** with its number before committing.

- [ ] **Step 4: Commit**

```bash
git add packages/pybamm/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_esoh_no_silent_failures.py
git commit -m "test: pin the Chayambuka2022 initialisation regression"
```

---

## Done when

- [ ] `pbwt pytest -m unit packages/pybamm/tests` — no more than 11 failures
- [ ] `test_esoh_no_silent_failures.py` — 11 passed, 1 xfailed
- [ ] `test_esoh_convergence.py` — 14 passed
- [ ] `-m integration -k "composite or esoh"` — 50 passed
- [ ] `grep -rn "ElectrodeSOHSolver" packages/pybamm/src` returns only the deprecated alias and its docstring references
- [ ] `uv run pre-commit run --all-files` clean on the changed files
