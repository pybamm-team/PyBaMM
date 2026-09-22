"""Parity: native (Rust) observation vs CasADi observation of a rust model."""

import numpy as np
import pytest

import pybamm
from pybamm.solvers.observation import _native_caches


def _solve(solver_cls, native, model_cls=pybamm.lithium_ion.SPM, **opts):
    model = model_cls()
    model.convert_to_format = "rust"
    solver = solver_cls(**opts)
    # The instance attribute survives the shallow solver.copy() Simulation performs.
    solver._observes_via_compiled_model = native
    sim = pybamm.Simulation(model, solver=solver)
    return sim.solve([0, 3600])


def _native_idaklu(native=True):
    """An IDAKLUSolver observing natively (``native=True``) or through CasADi."""
    solver = pybamm.IDAKLUSolver()
    solver._observes_via_compiled_model = native
    return solver


SOLVERS = [
    ("idaklu", pybamm.IDAKLUSolver, {}),
]
NATIVE_IDS = [s[0] for s in SOLVERS]

MODELS = [
    ("spm", pybamm.lithium_ion.SPM),
    ("spme", pybamm.lithium_ion.SPMe),
]
MODEL_IDS = [m[0] for m in MODELS]

# 0D scalars plus 1D fields on two distinct axes (r and x).
DIRECT_VARS = [
    "Terminal voltage [V]",
    "Current [A]",
    "Negative particle concentration [mol.m-3]",
    "Electrolyte concentration [mol.m-3]",
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
    """Non-F-contiguous `all_ys` would make every native observe copy the states."""

    @pytest.mark.parametrize(
        "solver_name,solver_cls,native_opts", SOLVERS, ids=NATIVE_IDS
    )
    def test_full_state_trajectory_is_f_contiguous(
        self, solver_name, solver_cls, native_opts
    ):
        sol = _solve(solver_cls, native=True, **native_opts)
        assert _native_caches(sol) is not None
        for ys in sol.all_ys:
            arr = np.asarray(ys)
            assert arr.ndim == 2
            assert arr.flags["F_CONTIGUOUS"], (
                f"{solver_name}: all_ys segment is not F-contiguous "
                f"(shape={arr.shape}, strides={arr.strides}) — eval_trajectory "
                "would gather-copy the whole state matrix on every observe."
            )

    def test_composed_trajectory_segments_are_f_contiguous(self):
        # Each stepped segment is consumed by its own eval_trajectory.
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
    # Unlike "Discharge capacity [A.h]", this drives the native time-integral read.
    def _build(native):
        model = pybamm.lithium_ion.SPM()
        model.variables["Integrated current [A.s]"] = pybamm.ExplicitTimeIntegral(
            model.variables["Current [A]"], pybamm.Scalar(0.0)
        )
        solver = solver_cls(**native_opts)
        model.convert_to_format = "rust"
        solver._observes_via_compiled_model = native
        return pybamm.Simulation(model, solver=solver).solve([0, 3600])

    native_sol = _build(native=True)
    casadi_sol = _build(native=False)
    assert _native_caches(native_sol) is not None
    native_entries = native_sol["Integrated current [A.s]"].entries
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
    model = pybamm.lithium_ion.SPM()
    model.convert_to_format = "rust"
    sim = pybamm.Simulation(model, solver=_native_idaklu())
    for _ in range(5):
        sim.step(360)
    native_v = sim.solution["Terminal voltage [V]"].entries

    cmodel = pybamm.lithium_ion.SPM()
    cmodel.convert_to_format = "rust"
    csim = pybamm.Simulation(cmodel, solver=_native_idaklu(native=False))
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
    assert _native_caches(sol) is not None, "Experiment solution does not read natively"
    native_v = sol["Terminal voltage [V]"].entries

    cmodel = pybamm.lithium_ion.SPM()
    cmodel.convert_to_format = "rust"
    csim = pybamm.Simulation(
        cmodel, experiment=experiment, solver=_native_idaklu(native=False)
    )
    csol = csim.solve()
    np.testing.assert_allclose(
        native_v, csol["Terminal voltage [V]"].entries, rtol=1e-7, atol=1e-9
    )


def test_multistep_time_integral_parity():
    # A time integral over a multi-segment solution integrates once, not per segment.
    def _build(native):
        model = pybamm.lithium_ion.SPM()
        model.variables["Integrated current [A.s]"] = pybamm.ExplicitTimeIntegral(
            model.variables["Current [A]"], pybamm.Scalar(0.0)
        )
        solver = pybamm.IDAKLUSolver()
        model.convert_to_format = "rust"
        solver._observes_via_compiled_model = native
        sim = pybamm.Simulation(model, solver=solver)
        for _ in range(3):
            sim.step(360)
        return sim.solution

    native_sol = _build(native=True)
    casadi_sol = _build(native=False)
    assert _native_caches(native_sol) is not None
    nv = native_sol["Integrated current [A.s]"].entries
    cv = casadi_sol["Integrated current [A.s]"].entries
    assert nv.shape == (1,), f"expected single scalar, got shape {nv.shape}"
    np.testing.assert_allclose(nv, cv, rtol=1e-8, atol=1e-10)


def test_discrete_time_sum_parity():
    # Both paths sum over solution.t, so pin it to the discrete times.
    discrete_times = np.linspace(0.0, 3600.0, 11)
    data = pybamm.DiscreteTimeData(discrete_times, np.zeros(11), "td")

    def _build(native):
        model = pybamm.lithium_ion.SPM()
        model.variables["DT sum"] = pybamm.DiscreteTimeSum(
            model.variables["Voltage [V]"] - data
        )
        solver = pybamm.IDAKLUSolver()
        model.convert_to_format = "rust"
        solver._observes_via_compiled_model = native
        return pybamm.Simulation(model, solver=solver).solve(
            t_eval=[0, 3600], t_interp=discrete_times
        )

    native_sol = _build(native=True)
    casadi_sol = _build(native=False)
    assert _native_caches(native_sol) is not None
    nv = native_sol["DT sum"].entries
    cv = casadi_sol["DT sum"].entries
    assert nv.shape == (1,), f"expected a single scalar, got shape {nv.shape}"
    np.testing.assert_allclose(nv, cv, rtol=1e-8, atol=1e-10)


def test_native_observed_variable_sensitivities_require_calculate_sensitivities():
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
    assert _native_caches(sol) is not None
    v = sol["Terminal voltage [V]"]
    assert len(sol.all_inputs[0]) > 0
    with pytest.raises(ValueError, match=r"calculate_sensitivities"):
        _ = v.sensitivities

    sens_sol = _build(calculate_sensitivities=True)
    assert sens_sol["Terminal voltage [V]"].sensitivities != {}


@pytest.mark.parametrize("input_val", [1, np.int64(1), np.float32(1.0)])
def test_native_observation_nonfloat_input_parity(input_val):
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
    assert _native_caches(native_sol) is not None
    np.testing.assert_allclose(
        native_sol["Terminal voltage [V]"].entries,
        casadi_sol["Terminal voltage [V]"].entries,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize("solver_cls", [pybamm.IDAKLUSolver])
def test_dense_trajectory_parity(solver_cls):
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
    assert _native_caches(native_sol) is not None
    assert native_sol["Terminal voltage [V]"].entries.shape[-1] == len(t_interp)
    for var in ("Terminal voltage [V]", "Electrolyte concentration [mol.m-3]"):
        np.testing.assert_allclose(
            native_sol[var].entries, casadi_sol[var].entries, rtol=1e-8, atol=1e-10
        )


def test_2d_variable_native_parity():
    # Locks the native column-major -> .T -> reshape contract for a 2D (r, x) field.
    var = "Negative particle concentration [mol.m-3]"
    native_sol = _solve(
        pybamm.IDAKLUSolver, native=True, model_cls=pybamm.lithium_ion.DFN
    )
    casadi_sol = _solve(
        pybamm.IDAKLUSolver, native=False, model_cls=pybamm.lithium_ion.DFN
    )
    native_entries = native_sol[var].entries
    casadi_entries = casadi_sol[var].entries
    assert native_entries.ndim == 3, (
        f"expected a 2D field (r, x, t), got ndim={native_entries.ndim}"
    )
    assert native_entries.shape == casadi_entries.shape
    # This test guards the 2D layout contract, not numerical equality.
    np.testing.assert_allclose(native_entries, casadi_entries, rtol=1e-4, atol=1e-3)


def test_3d_variable_native_parity():
    # Pins the unroll_3D order for a 3D (r, R, x) field.
    var = "Negative particle concentration distribution [mol.m-3]"
    native_sol = _solve(
        pybamm.IDAKLUSolver, native=True, model_cls=pybamm.lithium_ion.MPM
    )
    casadi_sol = _solve(
        pybamm.IDAKLUSolver, native=False, model_cls=pybamm.lithium_ion.MPM
    )
    native_entries = native_sol[var].entries
    casadi_entries = casadi_sol[var].entries
    assert native_entries.ndim == 4, (
        f"expected a 3D field (r, R, x, t), got ndim={native_entries.ndim}"
    )
    assert native_entries.shape == casadi_entries.shape
    # Layout contract: a wrong 3D reshape would mismatch by orders of magnitude.
    np.testing.assert_allclose(native_entries, casadi_entries, rtol=1e-4, atol=1e-3)
