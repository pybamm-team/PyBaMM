"""
Regression tests for historical electrode state of health (eSOH) bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
import pytest

import pybamm


class TestESOHHalfCellFixes:
    """Guards for eSOH half-cell model bug fixes."""

    def test_initial_soc_with_drive_cycle_half_cell(self):
        """
        Guards against: PR #3467 - fix esoh bug

        The bug caused `initial_soc` to error when using a drive cycle for
        half-cell models.
        """
        model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")

        t_data = np.linspace(0, 3600, 100)
        I_data = np.sin(t_data / 600) * 0.5 + 0.5

        param.update(
            {"Current function [A]": pybamm.Interpolant(t_data, I_data, pybamm.t)}
        )

        sim = pybamm.Simulation(model, parameter_values=param)

        sol = sim.solve([0, 600], initial_soc=0.8)

        assert len(sol.t) > 0
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.0)
        assert np.all(V < 5.0)


class TestInitialConditionsFromFixes:
    """Guards for initial_conditions_from scale evaluation fixes."""

    def test_inputs_propagate_to_ic_scale_evaluation(self):
        """
        Guards against: PR #5285 - Bugfix: inputs for `initial_conditions_from`
        scale evaluation

        The bug was that inputs were not properly propagated for scale
        evaluation when calling set_initial_conditions_from().
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
        param.update(
            {
                "Maximum concentration in negative electrode [mol.m-3]": "[input]",
            }
        )

        sim = pybamm.Simulation(model, parameter_values=param)

        sol1 = sim.solve(
            [0, 300],
            inputs={"Maximum concentration in negative electrode [mol.m-3]": c_n_max},
            initial_soc=0.8,
        )

        model2 = pybamm.lithium_ion.SPM()
        sim2 = pybamm.Simulation(model2, parameter_values=param)
        sim2.build()

        sim2.built_model.set_initial_conditions_from(
            sol1,
            inputs={"Maximum concentration in negative electrode [mol.m-3]": c_n_max},
        )

        sol2 = sim2.solve(
            [0, 300],
            inputs={"Maximum concentration in negative electrode [mol.m-3]": c_n_max},
        )

        assert len(sol2.t) > 0

        c_n = sol2["X-averaged negative particle concentration [mol.m-3]"].data
        assert np.all(c_n > 0)
        assert np.all(c_n < c_n_max)


class TestStartingSolutionFixes:
    """Guards for starting_solution bug fixes."""

    def test_last_state_as_starting_solution(self):
        """
        Guards against: PR #2822 / 6f81b35bb - Fix use of last-state as
        starting-solution in Simulation.solve()

        The bug was that using solution.last_state or a previous solution as
        starting_solution didn't work correctly.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        exp1 = pybamm.Experiment(["Discharge at C/2 for 20 minutes"])
        sim1 = pybamm.Simulation(model, parameter_values=param, experiment=exp1)
        sol1 = sim1.solve(calc_esoh=False)

        assert sol1.last_state is not None
        assert sol1.last_state.t[-1] == pytest.approx(1200, rel=1e-3)

        exp2 = pybamm.Experiment(["Discharge at C/2 for 10 minutes"])
        sim2 = pybamm.Simulation(model, parameter_values=param, experiment=exp2)
        sol2 = sim2.solve(calc_esoh=False, starting_solution=sol1)

        assert sol2["Time [s]"].entries[-1] == pytest.approx(1800, rel=1e-3)

        Q_end_sol1 = sol1["Discharge capacity [A.h]"].data[-1]
        Q_end_sol2 = sol2["Discharge capacity [A.h]"].data[-1]
        assert Q_end_sol2 > Q_end_sol1

    def test_starting_solution_with_experiment(self):
        """
        Verify starting_solution works with experiments (second scenario).
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        exp1 = pybamm.Experiment(["Discharge at 1C for 30 minutes"])
        sim1 = pybamm.Simulation(model, parameter_values=param, experiment=exp1)
        sol1 = sim1.solve()

        exp2 = pybamm.Experiment(
            ["Rest for 10 minutes", "Discharge at 0.5C for 30 minutes"]
        )
        sim2 = pybamm.Simulation(model, parameter_values=param, experiment=exp2)
        sol2 = sim2.solve(starting_solution=sol1)

        assert len(sol2.t) > 0
