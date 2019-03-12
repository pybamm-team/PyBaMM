#
# Tests for the Scikits Solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
from tests import StandardModelTest

import unittest
import numpy as np


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestScikitsSolver(unittest.TestCase):
    def test_model_solver_dae_autograd(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1", domain=[])
        var2 = pybamm.Variable("var2", domain=[])
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.initial_conditions_ydot = {var1: 0.1, var2: 0.2}
        disc = StandardModelTest(model).disc
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solver.solve(model, t_eval)
        np.testing.assert_array_equal(solver.t, t_eval)
        np.testing.assert_allclose(solver.y[0], np.exp(0.1 * solver.t))
        np.testing.assert_allclose(solver.y[-1], 2 * np.exp(0.1 * solver.t))

        # Check Jacobian and mass matrix
        mass = np.array([[1.0, 0.0], [0.0, 0.0]])
        jac = np.array([[0.1, 0.0], [2.0, -1.0]])
        y0 = model.concatenated_initial_conditions
        ydot0 = model.concatenated_initial_conditions_ydot
        automass, autojac = solver.jacobian(0.0, y0, ydot0)
        np.testing.assert_allclose(automass, mass)
        np.testing.assert_allclose(autojac, jac)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
