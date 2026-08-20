"""Parity: native (Rust) observation vs CasADi, for both rust-core solvers."""

import numpy as np
import pytest

import pybamm
from pybamm.solvers.observation import NativeObservation


def _solve(solver_cls, native, model_cls=pybamm.lithium_ion.SPM, **opts):
    model = model_cls()
    solver = solver_cls(**opts)
    if native:
        model.convert_to_format = "rust"
        # Force the per-solver capability flag on; the instance attribute survives
        # the shallow solver.copy() that Simulation performs.
        solver._observes_via_compiled_model = True
    sim = pybamm.Simulation(model, solver=solver)
    return sim.solve([0, 3600])


def _native_idaklu():
    """An IDAKLUSolver with native observation forced on (foundation tests only).

    idaklu's `_observes_via_compiled_model` stays False until prereqs A + B; the
    instance attribute set here survives the shallow solver.copy() Simulation does.
    Pair with `model.convert_to_format = "rust"`.
    """
    solver = pybamm.IDAKLUSolver()
    solver._observes_via_compiled_model = True
    return solver


SOLVERS = [
    ("idaklu", pybamm.IDAKLUSolver, {}),
    ("diffsol", pybamm.DiffsolSolver, {}),
]
NATIVE_IDS = [s[0] for s in SOLVERS]

MODELS = [
    ("spm", pybamm.lithium_ion.SPM),
    ("spme", pybamm.lithium_ion.SPMe),
]
MODEL_IDS = [m[0] for m in MODELS]

# 0D scalars plus 1D particle (r-axis) and electrolyte (x-axis) fields, so the
# unroll/transpose contract is exercised on two distinct 1D axes.
DIRECT_VARS = [
    "Terminal voltage [V]",  # 0D
    "Current [A]",  # 0D
    "Negative particle concentration [mol.m-3]",  # 1D (r-axis)
    "Electrolyte concentration [mol.m-3]",  # 1D (x-axis)
]


@pytest.mark.parametrize("solver_name,solver_cls,native_opts", SOLVERS, ids=NATIVE_IDS)
@pytest.mark.parametrize("model_name,model_cls", MODELS, ids=MODEL_IDS)
@pytest.mark.parametrize("var", DIRECT_VARS)
def test_direct_variable_parity(
    solver_name, solver_cls, native_opts, model_name, model_cls, var
):
    native_sol = _solve(solver_cls, native=True, model_cls=model_cls, **native_opts)
    casadi_sol = _solve(solver_cls, native=False, model_cls=model_cls)
    np.testing.assert_allclose(
        native_sol[var].entries, casadi_sol[var].entries, rtol=1e-9, atol=1e-11
    )


class TestZeroCopyStateTrajectory:
    """Pin the F-contiguity precondition for zero-copy `eval_trajectory`.

    `all_ys[i]` is exactly what the native path hands to
    `CompiledFunction.eval_trajectory`. It MUST be F-contiguous `(n_states,
    n_times)`; otherwise every observe gather-copies the whole state matrix
    (silent perf regression, no error).
    """

    @pytest.mark.parametrize(
        "solver_name,solver_cls,native_opts", SOLVERS, ids=NATIVE_IDS
    )
    def test_full_state_trajectory_is_f_contiguous(
        self, solver_name, solver_cls, native_opts
    ):
        sol = _solve(solver_cls, native=True, **native_opts)
        assert isinstance(sol.observation, NativeObservation)  # native path active
        for ys in sol.all_ys:
            arr = np.asarray(ys)
            assert arr.ndim == 2
            # F-contiguous ⇒ columns_slice borrows (zero-copy).
            assert arr.flags["F_CONTIGUOUS"], (
                f"{solver_name}: all_ys segment is not F-contiguous "
                f"(shape={arr.shape}, strides={arr.strides}) — eval_trajectory "
                "would gather-copy the whole state matrix on every observe."
            )

    def test_composed_trajectory_segments_are_f_contiguous(self):
        # Stepping accumulates per-segment arrays into one Solution, and each segment
        # is consumed by its own eval_trajectory, so each must stay F-contiguous.
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "rust"
        sim = pybamm.Simulation(model, solver=_native_idaklu())
        for _ in range(3):
            sim.step(360)
        for ys in sim.solution.all_ys:
            arr = np.asarray(ys)
            assert arr.flags["F_CONTIGUOUS"], (
                f"composed all_ys segment is not F-contiguous "
                f"(shape={arr.shape}, strides={arr.strides})"
            )


@pytest.mark.parametrize("solver_name,solver_cls,native_opts", SOLVERS, ids=NATIVE_IDS)
def test_time_integral_parity(solver_name, solver_cls, native_opts):
    native_sol = _solve(solver_cls, native=True, **native_opts)
    casadi_sol = _solve(solver_cls, native=False)
    np.testing.assert_allclose(
        native_sol["Discharge capacity [A.h]"].entries,
        casadi_sol["Discharge capacity [A.h]"].entries,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize("solver_name,solver_cls,native_opts", SOLVERS, ids=NATIVE_IDS)
def test_explicit_time_integral_parity(solver_name, solver_cls, native_opts):
    # A genuine ExplicitTimeIntegral output, unlike "Discharge capacity [A.h]", so
    # this drives the native backend's time-integral branch, not the direct one.
    def _build(native):
        model = pybamm.lithium_ion.SPM()
        model.variables["Integrated current [A.s]"] = pybamm.ExplicitTimeIntegral(
            model.variables["Current [A]"], pybamm.Scalar(0.0)
        )
        solver = solver_cls(**native_opts)
        if native:
            model.convert_to_format = "rust"
            solver._observes_via_compiled_model = True
        return pybamm.Simulation(model, solver=solver).solve([0, 3600])

    native_sol = _build(native=True)
    casadi_sol = _build(native=False)
    assert isinstance(native_sol.observation, NativeObservation)  # native path active
    native_entries = native_sol["Integrated current [A.s]"].entries
    # time-integral output is time-independent → a single scalar
    assert native_entries.shape == (1,), (
        f"Expected scalar shape (1,) for time-integral output, got {native_entries.shape}"
    )
    np.testing.assert_allclose(
        native_entries,
        casadi_sol["Integrated current [A.s]"].entries,
        rtol=1e-8,
        atol=1e-10,
    )


def test_stepping_addition_parity():
    # __add__: Simulation.step accumulates sub-solutions into one combined
    # Solution; observe on it. `sim.solution` is the running combined result.
    model = pybamm.lithium_ion.SPM()
    model.convert_to_format = "rust"
    sim = pybamm.Simulation(model, solver=_native_idaklu())
    for _ in range(5):
        sim.step(360)
    native_v = sim.solution["Terminal voltage [V]"].entries

    cmodel = pybamm.lithium_ion.SPM()
    csim = pybamm.Simulation(cmodel, solver=pybamm.IDAKLUSolver())
    for _ in range(5):
        csim.step(360)
    casadi_v = csim.solution["Terminal voltage [V]"].entries
    np.testing.assert_allclose(native_v, casadi_v, rtol=1e-8, atol=1e-10)


def test_first_last_state_parity():
    model = pybamm.lithium_ion.SPM()
    model.convert_to_format = "rust"
    sim = pybamm.Simulation(model, solver=_native_idaklu())
    sol = sim.solve([0, 3600])
    # Observation must still work through first/last_state, which carry the context.
    assert sol.first_state["Terminal voltage [V]"].entries.size >= 1
    assert sol.last_state["Terminal voltage [V]"].entries.size >= 1


def test_experiment_parity():
    experiment = pybamm.Experiment(
        ["Discharge at 1C until 3.0 V", "Charge at 1C until 4.0 V"]
    )
    model = pybamm.lithium_ion.SPM()
    model.convert_to_format = "rust"
    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        solver=_native_idaklu(),
    )
    sol = sim.solve()
    # Verify the experiment result genuinely used the native path.
    assert isinstance(sol.observation, NativeObservation), (
        "Experiment solution lost its native backend — native path not active"
    )
    native_v = sol["Terminal voltage [V]"].entries

    cmodel = pybamm.lithium_ion.SPM()
    csim = pybamm.Simulation(
        cmodel, experiment=experiment, solver=pybamm.IDAKLUSolver()
    )
    csol = csim.solve()
    np.testing.assert_allclose(
        native_v, csol["Terminal voltage [V]"].entries, rtol=1e-7, atol=1e-9
    )


def test_diffsol_native_by_default():
    # After the flip, a plain DiffsolSolver observes natively without any opt-in.
    t_eval = np.linspace(0, 3600, 51)

    model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(model, solver=pybamm.DiffsolSolver())
    sol = sim.solve(t_eval=t_eval)
    assert isinstance(sol.observation, NativeObservation)  # native path active

    # Native diffsol vs CasADi-backed IDAKLU at their default tolerances
    # (1e-6 and 1e-4), so rtol=1e-4 is the right cross-solver bound.
    cmodel = pybamm.lithium_ion.SPM()
    csim = pybamm.Simulation(cmodel, solver=pybamm.IDAKLUSolver())
    csol = csim.solve(t_eval=t_eval, t_interp=t_eval)
    np.testing.assert_allclose(
        sol["Terminal voltage [V]"].entries,
        csol["Terminal voltage [V]"].entries,
        rtol=1e-4,
        atol=1e-6,
    )


def test_diffsol_does_not_build_casadi_observation(monkeypatch):
    import pybamm.solvers.solution as solution_mod

    model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(model, solver=pybamm.DiffsolSolver())
    sol = sim.solve([0, 600])

    called = {"casadi": False}
    orig = solution_mod.Solution._convert_to_casadi

    def spy(self, *a, **k):
        called["casadi"] = True
        return orig(self, *a, **k)

    monkeypatch.setattr(solution_mod.Solution, "_convert_to_casadi", spy)
    _ = sol["Terminal voltage [V]"]
    assert called["casadi"] is False


def test_multistep_time_integral_parity():
    # Phase-2 x Phase-3: a time integral observed over a MULTI-SEGMENT solution
    # must integrate ONCE over the full trajectory (shape (1,)), not per segment.
    def _build(native):
        model = pybamm.lithium_ion.SPM()
        model.variables["Integrated current [A.s]"] = pybamm.ExplicitTimeIntegral(
            model.variables["Current [A]"], pybamm.Scalar(0.0)
        )
        solver = pybamm.IDAKLUSolver()
        if native:
            model.convert_to_format = "rust"
            solver._observes_via_compiled_model = True
        sim = pybamm.Simulation(model, solver=solver)
        for _ in range(3):
            sim.step(360)
        return sim.solution

    native_sol = _build(native=True)
    casadi_sol = _build(native=False)
    assert isinstance(native_sol.observation, NativeObservation)
    nv = native_sol["Integrated current [A.s]"].entries
    cv = casadi_sol["Integrated current [A.s]"].entries
    # Single integral over the whole 3-step trajectory, NOT one value per step.
    assert nv.shape == (1,), f"expected single scalar, got shape {nv.shape}"
    np.testing.assert_allclose(nv, cv, rtol=1e-8, atol=1e-10)


def test_discrete_time_sum_parity():
    # Both paths sum the integrand over solution.t, so t_interp == discrete times
    # makes the sum run over exactly those times.
    discrete_times = np.linspace(0.0, 3600.0, 11)
    data = pybamm.DiscreteTimeData(discrete_times, np.zeros(11), "td")

    def _build(native):
        model = pybamm.lithium_ion.SPM()
        model.variables["DT sum"] = pybamm.DiscreteTimeSum(
            model.variables["Voltage [V]"] - data
        )
        solver = pybamm.IDAKLUSolver()
        if native:
            model.convert_to_format = "rust"
            solver._observes_via_compiled_model = True
        return pybamm.Simulation(model, solver=solver).solve(
            t_eval=[0, 3600], t_interp=discrete_times
        )

    native_sol = _build(native=True)
    casadi_sol = _build(native=False)
    assert isinstance(native_sol.observation, NativeObservation)
    nv = native_sol["DT sum"].entries
    cv = casadi_sol["DT sum"].entries
    assert nv.shape == (1,), f"expected a single scalar, got shape {nv.shape}"
    np.testing.assert_allclose(nv, cv, rtol=1e-8, atol=1e-10)


def test_native_observed_variable_sensitivities_require_calculate_sensitivities():
    # Without calculate_sensitivities, a native solve with an input parameter must
    # raise the same informative error as CasADi, not return None or {}.
    def _build(calculate_sensitivities):
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "rust"
        param = model.default_parameter_values
        param["Current function [A]"] = pybamm.InputParameter("I")
        sim = pybamm.Simulation(model, parameter_values=param, solver=_native_idaklu())
        return sim.solve(
            [0, 3600],
            inputs={"I": 0.5},
            calculate_sensitivities=calculate_sensitivities,
        )

    sol = _build(calculate_sensitivities=False)
    assert isinstance(sol.observation, NativeObservation)
    v = sol["Terminal voltage [V]"]
    assert len(sol.all_inputs[0]) > 0  # inputs present, so the no-input shortcut
    with pytest.raises(ValueError, match=r"calculate_sensitivities"):
        _ = v.sensitivities

    # Requesting sensitivities: the native path returns real values, not {}.
    sens_sol = _build(calculate_sensitivities=True)
    assert sens_sol["Terminal voltage [V]"].sensitivities != {}


@pytest.mark.parametrize("input_val", [1, np.int64(1), np.float32(1.0)])
def test_native_observation_nonfloat_input_parity(input_val):
    # non-float inputs (int64/float32) must match CasADi, not crash the input pack.
    def build(native):
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "rust"
        param = model.default_parameter_values
        param["Current function [A]"] = pybamm.InputParameter("I")
        solver = pybamm.IDAKLUSolver()
        solver._observes_via_compiled_model = native
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
        return sim.solve([0, 3600], inputs={"I": input_val})

    native_sol = build(native=True)
    casadi_sol = build(native=False)
    assert isinstance(native_sol.observation, NativeObservation)
    np.testing.assert_allclose(
        native_sol["Terminal voltage [V]"].entries,
        casadi_sol["Terminal voltage [V]"].entries,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize("solver_cls", [pybamm.IDAKLUSolver, pybamm.DiffsolSolver])
def test_dense_trajectory_parity(solver_cls):
    # parity must hold over a dense t_interp (large n_t), not only adaptive steps.
    t_interp = np.linspace(0, 3600, 500)

    def build(native):
        model = pybamm.lithium_ion.SPMe()
        model.convert_to_format = "rust"
        solver = solver_cls()
        solver._observes_via_compiled_model = native
        return pybamm.Simulation(model, solver=solver).solve(
            t_eval=[0, 3600], t_interp=t_interp
        )

    native_sol = build(native=True)
    casadi_sol = build(native=False)
    assert isinstance(native_sol.observation, NativeObservation)
    assert native_sol["Terminal voltage [V]"].entries.shape[-1] == len(t_interp)
    for var in ("Terminal voltage [V]", "Electrolyte concentration [mol.m-3]"):
        np.testing.assert_allclose(
            native_sol[var].entries, casadi_sol[var].entries, rtol=1e-8, atol=1e-10
        )


def test_2d_variable_native_parity():
    # A 2D field (resolved across both r and x in DFN) drives initialise_2D and
    # locks the native column-major -> .T -> reshape contract against CasADi.
    var = "Negative particle concentration [mol.m-3]"
    native_sol = _solve(
        pybamm.IDAKLUSolver, native=True, model_cls=pybamm.lithium_ion.DFN
    )
    casadi_sol = _solve(
        pybamm.IDAKLUSolver, native=False, model_cls=pybamm.lithium_ion.DFN
    )
    native_entries = native_sol[var].entries
    casadi_entries = casadi_sol[var].entries
    # (r, x, t): genuinely 2D in space, so this is the 2D unroll path.
    assert native_entries.ndim == 3, (
        f"expected a 2D field (r, x, t), got ndim={native_entries.ndim}"
    )
    assert native_entries.shape == casadi_entries.shape
    # This test guards the 2D layout contract. Native and CasADi use different eval
    # cores, whose tiny RHS differences DFN amplifies to ~1e-6, over the trajectory.
    np.testing.assert_allclose(native_entries, casadi_entries, rtol=1e-4, atol=1e-3)


def test_3d_variable_native_parity():
    # 3D field (MPM, r x R x x): drives initialise_3D / unroll_3D. Order-pinning
    # test for the unroll_3D fix; a scrambled field is non-uniform at t=0.
    var = "Negative particle concentration distribution [mol.m-3]"
    native_sol = _solve(
        pybamm.IDAKLUSolver, native=True, model_cls=pybamm.lithium_ion.MPM
    )
    casadi_sol = _solve(
        pybamm.IDAKLUSolver, native=False, model_cls=pybamm.lithium_ion.MPM
    )
    native_entries = native_sol[var].entries
    casadi_entries = casadi_sol[var].entries
    # (r, R, x, t): genuinely 3D in space, so this is the 3D unroll path.
    assert native_entries.ndim == 4, (
        f"expected a 3D field (r, R, x, t), got ndim={native_entries.ndim}"
    )
    assert native_entries.shape == casadi_entries.shape
    # Layout contract, not numerical equality (cross eval-core drift); a wrong
    # 3D reshape/axis-swap would mismatch by orders of magnitude.
    np.testing.assert_allclose(native_entries, casadi_entries, rtol=1e-4, atol=1e-3)
