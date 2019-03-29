#
# Tests for the Parameter class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import numpy as np
import unittest


class TestVariable(unittest.TestCase):
    def test_variable_init(self):
        a = pybamm.Variable("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])
        a = pybamm.Variable("a", domain=["test"])
        self.assertEqual(a.domain[0], "test")
        self.assertRaises(TypeError, pybamm.Variable("a", domain="test"))

    def test_variable_id(self):
        a1 = pybamm.Variable("a", domain=["negative electrode"])
        a2 = pybamm.Variable("a", domain=["negative electrode"])
        self.assertEqual(a1.id, a2.id)
        a3 = pybamm.Variable("b", domain=["negative electrode"])
        a4 = pybamm.Variable("a", domain=["positive electrode"])
        self.assertNotEqual(a1.id, a3.id)
        self.assertNotEqual(a1.id, a4.id)


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

        # with space

    def test_processed_var_interpolation(self):
        pass

    def test_processed_variable_ode_pde_solution(self):
        # without space
        model = pybamm.BaseModel()
        c = pybamm.Variable("conc")
        model.rhs = {c: 1}
        model.initial_conditions = {c: 1}
        model.variables = {"c": c}
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
