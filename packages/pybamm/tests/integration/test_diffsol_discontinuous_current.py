"""diffsol under a current profile with many discontinuities."""

from __future__ import annotations

import numpy as np
import pytest

import pybamm

PULSE_AMPLITUDE_A = 5.0
PULSE_ON_S = 60.0
PULSE_REST_S = 120.0
PULSE_RAMP_S = 2.0
DURATION_S = 1800.0
SOLVER_TOL = 1e-6


def _pulse_train_breakpoints() -> tuple[np.ndarray, np.ndarray]:
    """Corners of a ramped pulse/rest train; finite ramps keep it Lipschitz."""
    times, values = [0.0], [0.0]
    start = 0.0
    period = PULSE_ON_S + PULSE_REST_S
    while start < DURATION_S:
        times.extend(
            [
                start + PULSE_RAMP_S,
                start + PULSE_ON_S,
                start + PULSE_ON_S + PULSE_RAMP_S,
                start + period,
            ]
        )
        values.extend([PULSE_AMPLITUDE_A, PULSE_AMPLITUDE_A, 0.0, 0.0])
        start += period
    times_arr = np.asarray(times, dtype=np.float64)
    keep = times_arr <= DURATION_S
    return times_arr[keep], np.asarray(values, dtype=np.float64)[keep]


def _pulse_train_parameter_values(inputs: dict[str, str] | None = None):
    parameter_values = pybamm.ParameterValues("Chen2020")
    times, values = _pulse_train_breakpoints()
    parameter_values["Current function [A]"] = pybamm.Interpolant(
        times, values, pybamm.t, interpolator="linear"
    )
    for name, input_name in (inputs or {}).items():
        parameter_values[name] = pybamm.InputParameter(input_name)
    return parameter_values


INFERENCE_INPUTS = {
    "Negative particle diffusivity [m2.s-1]": "D_n",
    "Positive particle diffusivity [m2.s-1]": "D_p",
    "Negative electrode active material volume fraction": "eps_n",
    "Positive electrode active material volume fraction": "eps_p",
}
INPUT_VALUES = {"D_n": 3.3e-14, "D_p": 4.0e-15, "eps_n": 0.75, "eps_p": 0.665}


def _solve(
    solver,
    calculate_sensitivities=False,
    inputs=None,
    options=None,
    t_eval=None,
    model_factory=pybamm.lithium_ion.DFN,
):
    model = model_factory()
    model.convert_to_format = "casadi" if solver == "casadi_idaklu" else "rust"
    parameter_values = _pulse_train_parameter_values(
        INFERENCE_INPUTS if inputs else None
    )
    if solver == "diffsol":
        instance = pybamm.DiffsolSolver(
            rtol=SOLVER_TOL, atol=SOLVER_TOL, options=options
        )
    else:
        instance = pybamm.IDAKLUSolver(rtol=SOLVER_TOL, atol=SOLVER_TOL)
    breakpoints, _ = _pulse_train_breakpoints()
    simulation = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=instance
    )
    return simulation.solve(
        breakpoints if t_eval is None else t_eval,
        t_interp=np.linspace(0.0, DURATION_S, 100),
        initial_soc=0.5,
        inputs=inputs,
        calculate_sensitivities=calculate_sensitivities,
    )


class TestDiffsolPulseTrain:
    def test_the_workload_still_costs_nonlinear_solver_failures(self):
        # The failure-budget test below can only bite on a workload that spends some.
        solution = _solve("diffsol")
        assert solution.solver_statistics.number_of_nonlinear_solver_fails > 5

    def test_a_long_pulse_train_completes(self):
        solution = _solve("diffsol")
        assert solution.t[-1] == pytest.approx(DURATION_S)
        assert np.all(np.isfinite(np.asarray(solution["Voltage [V]"](solution.t))))

    def test_the_failure_budget_option_reaches_diffsol(self):
        with pytest.raises(pybamm.SolverError, match=r"nonlinear solver failures"):
            _solve("diffsol", options={"max_nonlinear_solver_failures": 5})

    def test_values_match_idaklu(self):
        native = _solve("diffsol")
        reference = _solve("casadi_idaklu")
        t = np.linspace(0.0, DURATION_S, 100)
        np.testing.assert_allclose(
            np.asarray(native["Voltage [V]"](t)),
            np.asarray(reference["Voltage [V]"](t)),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_t_eval_breakpoints_are_stop_times_not_just_output_times(self):
        # Equal cost either way would mean t_eval never reached the step control.
        given = _solve("diffsol")
        hidden = _solve("diffsol", t_eval=np.array([0.0, DURATION_S]))
        assert (
            given.solver_statistics.number_of_nonlinear_solver_fails
            < hidden.solver_statistics.number_of_nonlinear_solver_fails
        )
        assert (
            given.solver_statistics.number_of_error_test_failures
            < hidden.solver_statistics.number_of_error_test_failures
        )

    def test_gradients_hold_error_control_across_the_corners(self):
        # Corners must be met with consistent sensitivity derivatives, not just states.
        # Whether one solve this tight relaxes is down to rounding, so most must hold.
        relaxed = []
        for k in range(6):
            inputs = {**INPUT_VALUES, "eps_n": INPUT_VALUES["eps_n"] * (1 + k * 1e-9)}
            solution = _solve(
                "diffsol", calculate_sensitivities=sorted(inputs), inputs=inputs
            )
            relaxed.append(solution.solver_statistics.sens_error_control_relaxed)
        assert sum(relaxed) <= 2, relaxed

    def test_a_pure_ode_model_meets_the_corners_too(self):
        # With no algebraic state diffsol's `set_consistent` returns early.
        native = _solve("diffsol", model_factory=pybamm.lithium_ion.SPM)
        reference = _solve("casadi_idaklu", model_factory=pybamm.lithium_ion.SPM)
        t = np.linspace(0.0, DURATION_S, 100)
        np.testing.assert_allclose(
            np.asarray(native["Voltage [V]"](t)),
            np.asarray(reference["Voltage [V]"](t)),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_a_long_pulse_train_completes_with_sensitivities(self):
        inputs = INPUT_VALUES
        names = sorted(inputs)
        solution = _solve("diffsol", calculate_sensitivities=names, inputs=inputs)

        assert solution.t[-1] == pytest.approx(DURATION_S)
        for name in names:
            gradient = np.asarray(solution["Voltage [V]"].sensitivities[name]).ravel()
            assert np.all(np.isfinite(gradient))
            assert np.abs(gradient).max() > 0.0


STEP_TIME_S = 1800.0


def _solve_stepped_current(solver, t_eval):
    """An SPM whose current steps down at ``STEP_TIME_S``."""
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values["Current function [A]"] = lambda t: 5 * (t < STEP_TIME_S) + 1
    simulation = pybamm.Simulation(
        pybamm.lithium_ion.SPM(), parameter_values=parameter_values, solver=solver
    )
    return simulation.solve(t_eval)


class TestDiffsolSteppedCurrent:
    """A Heaviside step adds a stop time to ``t_eval``, which leaves t0 coarse."""

    @pytest.mark.parametrize(
        "t_eval",
        [[0, 2 * STEP_TIME_S], np.array([0, STEP_TIME_S, 2 * STEP_TIME_S])],
        ids=["bare span", "three points"],
    )
    @pytest.mark.parametrize(
        ("hermite", "read_atol"),
        [(True, 3e-4), (False, 3e-3)],
        ids=["hermite", "linear"],
    )
    def test_voltage_matches_idaklu(self, t_eval, hermite, read_atol):
        native = _solve_stepped_current(
            pybamm.DiffsolSolver(
                rtol=SOLVER_TOL, atol=SOLVER_TOL, hermite_interpolation=hermite
            ),
            t_eval,
        )
        reference = _solve_stepped_current(
            pybamm.IDAKLUSolver(rtol=SOLVER_TOL, atol=SOLVER_TOL),
            [0, 2 * STEP_TIME_S],
        )
        np.testing.assert_allclose(
            native["Voltage [V]"](native.t),
            reference["Voltage [V]"](native.t),
            rtol=0,
            atol=1e-5,
        )
        # Off the 100-point grid a read is only as good as its interpolant.
        t = np.linspace(0.0, 2 * STEP_TIME_S, 50)
        np.testing.assert_allclose(
            native["Voltage [V]"](t),
            reference["Voltage [V]"](t),
            rtol=0,
            atol=read_atol,
        )
