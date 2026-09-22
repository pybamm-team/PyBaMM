"""``DiffsolSolver`` native initial conditions against PyBaMM's root solve."""

import re

import numpy as np
import pytest

import pybamm

UNSTRUCTURED_VAR_PTS = {
    "r_n": 5,
    "r_p": 5,
    "x_n": 10,
    "x_s": 10,
    "x_p": 10,
    "y": 5,
    "z": 5,
}


def _chen2020(c_rate):
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values["Current function [A]"] = (
        c_rate * parameter_values["Nominal cell capacity [A.h]"]
    )
    return parameter_values


def _solve(calc_ic, parameter_values=None, model=None, var_pts=None, **solve_kwargs):
    simulation = pybamm.Simulation(
        model or pybamm.lithium_ion.DFN(),
        parameter_values=parameter_values,
        solver=pybamm.DiffsolSolver(calc_ic=calc_ic),
        var_pts=var_pts,
    )
    return simulation.solve(**solve_kwargs)


class TestDiffsolInitialConditions:
    @pytest.mark.parametrize(
        "parameter_values",
        [None, _chen2020(1.0)],
        ids=["Marquis2019", "Chen2020"],
    )
    def test_native_initialisation_matches_the_root_solve(self, parameter_values):
        t_eval = np.linspace(0, 3000, 51)
        native = _solve(True, parameter_values, t_eval=[0, 3000], t_interp=t_eval)
        python = _solve(False, parameter_values, t_eval=[0, 3000], t_interp=t_eval)
        np.testing.assert_allclose(
            native["Voltage [V]"](t_eval), python["Voltage [V]"](t_eval), atol=1e-5
        )

    def test_off_grid_reads_in_the_first_interval_match_the_root_solve(self):
        # A first-step secant at t0 once read 14 mV off at t = 17.8 s on this grid.
        grid = np.linspace(0, 600, 11)
        t_check = np.array([2.0, 10.0, 17.8, 30.0, 50.0])
        native = _solve(True, t_eval=[0, 600], t_interp=grid)
        python = _solve(False, t_eval=[0, 600], t_interp=grid)
        np.testing.assert_allclose(
            native["Voltage [V]"](t_check), python["Voltage [V]"](t_check), atol=1e-5
        )

    def test_charging_from_empty_starts_at_the_consistent_state(self):
        # The guessed potentials sit exactly on the 2.5 V cut-off; the consistent ones don't.
        native = _solve(True, _chen2020(-1.0), t_eval=[0, 3000], initial_soc=0)
        python = _solve(False, _chen2020(-1.0), t_eval=[0, 3000], initial_soc=0)
        assert native.termination == "event: Maximum voltage [V]"
        assert native.termination == python.termination
        np.testing.assert_allclose(native.t[-1], python.t[-1], atol=1.0)

    @pytest.mark.parametrize("calc_ic", [True, False])
    def test_discharging_from_empty_is_infeasible_at_initial_conditions(self, calc_ic):
        with pytest.raises(
            pybamm.SolverError,
            match=re.escape(
                "Events ['Minimum voltage [V]'] are non-positive at initial conditions"
            ),
        ):
            _solve(calc_ic, _chen2020(1.0), t_eval=[0, 600], initial_soc=0)

    @pytest.mark.parametrize("calc_ic", [True, False])
    def test_an_experiment_skips_a_step_infeasible_at_its_start(self, calc_ic):
        # From a full cell the charge step's 4.1 V cut-off is already exceeded.
        experiment = pybamm.Experiment(
            [("Charge at 1C until 4.1 V", "Rest for 1 minute")]
        )
        simulation = pybamm.Simulation(
            pybamm.lithium_ion.DFN(),
            parameter_values=pybamm.ParameterValues("Chen2020"),
            experiment=experiment,
            solver=pybamm.DiffsolSolver(calc_ic=calc_ic),
        )
        solution = simulation.solve(initial_soc=1)
        charge, rest = solution.cycles[0].steps
        assert isinstance(charge, pybamm.EmptySolution)
        assert charge.termination == "Event exceeded in initial conditions"
        assert rest.termination == "final time"

    def test_an_outputs_only_first_state_is_the_consistent_state(self):
        # The local ECM resistance reads the potentials, so it is ~0 at the guess.
        name = "Local ECM resistance [Ohm]"
        outputs = pybamm.Simulation(
            pybamm.lithium_ion.DFN(),
            solver=pybamm.DiffsolSolver(output_variables=["Voltage [V]"]),
        ).solve([0, 60])
        full = _solve(True, t_eval=[0, 60])
        np.testing.assert_allclose(
            outputs.first_state[name](0), full.first_state[name](0), rtol=1e-5
        )

    def test_a_3d_model_initialises_natively(self):
        model_options = {"dimensionality": 2}
        var_pts = {
            **pybamm.lithium_ion.BasicDFNUnstructured(model_options).default_var_pts,
            **UNSTRUCTURED_VAR_PTS,
        }
        grid = np.linspace(0, 600, 11)
        native, python = (
            _solve(
                calc_ic,
                model=pybamm.lithium_ion.BasicDFNUnstructured(model_options),
                var_pts=var_pts,
                t_eval=[0, 600],
                t_interp=grid,
            )
            for calc_ic in (True, False)
        )
        assert native.termination == "final time"
        np.testing.assert_allclose(
            native["Voltage [V]"](grid), python["Voltage [V]"](grid), atol=1e-5
        )
