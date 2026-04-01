#
# Tests for the Summary Variables class
#

import numpy as np
import pytest

import pybamm


class TestSummaryVariables:
    @staticmethod
    def create_sum_vars():
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c
        model.summary_variables = ["2c"]

        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        sum_vars = pybamm.SummaryVariables(solution)

        return sum_vars, solution

    def test_init(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c
        model.summary_variables = ["2c"]

        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        sum_vars = pybamm.SummaryVariables(solution)

        # no variables should have been calculated until called
        assert sum_vars._variables == {}

        assert sum_vars.first_state == solution.first_state
        assert sum_vars.last_state == solution.last_state
        assert sum_vars.cycles is None

    def test_init_with_cycle_summary_variables(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c
        model.summary_variables = ["2c"]

        sol1 = pybamm.ScipySolver().solve(model, np.linspace(0, 1))
        sol2 = pybamm.ScipySolver().solve(model, np.linspace(1, 2))
        sol3 = pybamm.ScipySolver().solve(model, np.linspace(2, 3))

        cycle_sol = sol1 + sol2 + sol3

        all_sum_vars = [
            pybamm.SummaryVariables(sol1),
            pybamm.SummaryVariables(sol2),
            pybamm.SummaryVariables(sol3),
        ]

        cycle_sum_vars = pybamm.SummaryVariables(
            cycle_sol,
            cycle_summary_variables=all_sum_vars,
        )

        assert cycle_sum_vars.first_state is None
        assert cycle_sum_vars.last_state is None
        assert cycle_sum_vars._variables == {}
        assert cycle_sum_vars.cycles == all_sum_vars
        np.testing.assert_array_equal(cycle_sum_vars.cycle_number, np.array([1, 2, 3]))

    def test_get_variable(self):
        sum_vars, solution = self.create_sum_vars()

        summary_c = sum_vars["2c"]

        assert summary_c == solution["2c"].data[-1]
        assert list(sum_vars._variables.keys()) == ["2c", "Change in 2c"]

    def test_get_esoh_variable(self):
        model = pybamm.lithium_ion.SPM()
        rtol = 1e-4
        atol = 1e-6
        sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver(rtol=rtol, atol=atol))
        sol = sim.solve([0, 1])
        esoh_solver = sim.get_esoh_solver(True, None)
        sum_vars_esoh = pybamm.SummaryVariables(sol, esoh_solver=esoh_solver)

        assert np.isclose(
            sum_vars_esoh["x_100"], 0.9493, rtol=rtol * 10, atol=atol * 10
        )

        # all esoh vars should be calculated at the same time to reduce solver calls
        assert "Practical NPR" in sum_vars_esoh._variables

    def test_get_variable_error_not_summary_variable(self):
        sum_vars, _ = self.create_sum_vars()

        with pytest.raises(KeyError, match=r"Variable 'c' is not a summary variable"):
            sum_vars["c"]

    def test_summary_vars_all_variables(self):
        # no esoh
        sum_vars, _ = self.create_sum_vars()

        assert sum_vars.all_variables == ["2c", "Change in 2c"]

        # with esoh
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve(np.linspace(0, 1))
        esoh_solver = sim.get_esoh_solver(True, None)
        sum_vars_esoh = pybamm.SummaryVariables(sol, esoh_solver=esoh_solver)

        assert sum_vars_esoh.calc_esoh is True
        assert "Total lithium lost [mol]" in sum_vars_esoh.all_variables
        assert "x_100" in sum_vars_esoh.all_variables

        assert "x_100" in sum_vars_esoh._esoh_variables

    def test_get_with_cycle_summary_variables(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c
        model.summary_variables = ["2c"]

        sol1 = pybamm.ScipySolver().solve(model, np.linspace(0, 1))
        sol2 = pybamm.ScipySolver().solve(model, np.linspace(1, 2))
        sol3 = pybamm.ScipySolver().solve(model, np.linspace(2, 3))

        cycle_sol = sol1 + sol2 + sol3

        all_sum_vars = [
            pybamm.SummaryVariables(sol1),
            pybamm.SummaryVariables(sol2),
            pybamm.SummaryVariables(sol3),
        ]

        cycle_sum_vars = pybamm.SummaryVariables(
            cycle_sol,
            cycle_summary_variables=all_sum_vars,
        )

        np.testing.assert_array_equal(cycle_sum_vars.cycle_number, np.array([1, 2, 3]))
        np.testing.assert_allclose(
            cycle_sum_vars["2c"], np.array([0.735758, 0.735758, 0.735758])
        )

    def test_get_esoh_cycle_summary_vars(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C for 1 sec",
                    "Charge at 1C for 1 sec",
                ),
            ]
            * 10,
        )
        rtol = 1e-4
        atol = 1e-6
        solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
        sol = sim.solve()

        assert len(sol.summary_variables.cycles) == 10
        assert sol.summary_variables["Cycle number"][0] == 1
        assert sol.summary_variables["Cycle number"][9] == 10
        assert len(sol.summary_variables["x_100"]) == 10
        assert sol.summary_variables["x_100"][0] == pytest.approx(
            0.9493, rel=rtol * 10, abs=atol * 10
        )

    def test_summary_vars_get_all_variables(self):
        # no esoh
        sum_vars, _ = self.create_sum_vars()

        summary_vars = sum_vars.get_summary_variables()

        assert isinstance(summary_vars, dict)
        assert list(summary_vars.keys()) == ["2c", "Change in 2c", "Cycle number"]
        np.isclose(summary_vars["2c"], 0.735758)
