#
# Tests for the Processed Variable class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import numpy as np
import unittest


class TestProcessedVariable(unittest.TestCase):
    def test_simple_processed_variable(self):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = t * y
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        processed_var = pybamm.ProcessedVariable(var, t_sol, y_sol)
        np.testing.assert_array_equal(processed_var.entries, t_sol * y_sol)

    def test_processed_var_space(self):
        t = pybamm.t
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        eqn = t * var + x

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries
        var_sol = disc.process_symbol(var)
        eqn_sol = disc.process_symbol(eqn)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * np.linspace(0, 5)

        processed_var = pybamm.ProcessedVariable(var_sol, t_sol, y_sol, mesh=disc.mesh)
        np.testing.assert_array_equal(processed_var.entries, y_sol)
        processed_eqn = pybamm.ProcessedVariable(eqn_sol, t_sol, y_sol, mesh=disc.mesh)
        np.testing.assert_array_equal(
            processed_eqn.entries, t_sol * y_sol + x_sol[:, np.newaxis]
        )

    def test_processed_var_interpolation(self):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var1 = y
        t_sol = np.linspace(0, 1, 1000)
        y_sol = np.array([np.linspace(0, 5, 1000)])
        processed_var1 = pybamm.ProcessedVariable(var1, t_sol, y_sol)
        np.testing.assert_array_equal(processed_var1.evaluate(t_sol), y_sol)
        np.testing.assert_array_equal(processed_var1.evaluate(0.5), 2.5)
        np.testing.assert_array_equal(processed_var1.evaluate(0.7), 3.5)

        var2 = t * y
        processed_var2 = pybamm.ProcessedVariable(var2, t_sol, y_sol)
        np.testing.assert_array_equal(processed_var2.evaluate(t_sol), t_sol * y_sol)
        np.testing.assert_array_almost_equal(processed_var2.evaluate(0.5), 0.5 * 2.5)

        # var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        # x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        # eqn = t * var + x
        #
        # disc = tests.get_discretisation_for_testing()
        # disc.set_variable_slices([var])
        # x_sol = disc.process_symbol(x).entries
        # var_sol = disc.process_symbol(var)
        # eqn_sol = disc.process_symbol(eqn)
        # t_sol = np.linspace(0, 1)
        # y_sol = np.ones_like(x_sol)[:, np.newaxis] * np.linspace(0, 5)
        #
        # processed_var = pybamm.ProcessedVariable(var_sol, t_sol, y_sol, mesh=disc.mesh)
        # np.testing.assert_array_equal(processed_var.entries, y_sol)
        # processed_eqn = pybamm.ProcessedVariable(eqn_sol, t_sol, y_sol, mesh=disc.mesh)
        # np.testing.assert_array_equal(
        #     processed_eqn.entries, t_sol * y_sol + x_sol[:, np.newaxis]
        # )

    def test_processed_variable_ode_pde_solution(self):
        # without space
        model = pybamm.BaseModel()
        c = pybamm.Variable("conc")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables = {"c": c}
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y
        processed_var = pybamm.ProcessedVariable(model.variables["c"], t_sol, y_sol)
        np.testing.assert_array_almost_equal(processed_var.entries[0], np.exp(-t_sol))

        # with space
        # set up and solve model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        c = pybamm.Variable("conc", domain=whole_cell)
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables = {"c": c}
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
        # set up testing
        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y
        x = pybamm.SpatialVariable("x", domain=whole_cell)
        x_sol = modeltest.disc.process_symbol(x).entries
        processed_var = pybamm.ProcessedVariable(
            model.variables["c"], t_sol, y_sol, mesh=modeltest.disc.mesh
        )
        # test
        np.testing.assert_array_almost_equal(
            processed_var.entries, np.ones_like(x_sol)[:, np.newaxis] * np.exp(-t_sol)
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
