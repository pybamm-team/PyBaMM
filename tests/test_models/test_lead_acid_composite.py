#
# Tests for the lead-acid composite model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest
import numpy as np


@unittest.skip("")
class TestLeadAcidComposite(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.Composite()

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_solution(self):
        model = pybamm.lead_acid.Composite()

        # process parameter values, discretise and solve
        model.default_parameter_values.process_model(model)
        disc = model.default_discretisation
        disc.process_model(model)
        t_eval = disc.mesh["time"]
        solver = model.default_solver
        solver.solve(model, t_eval)
        T, Y = solver.t, solver.y

        # check output
        # make sure concentration and voltage are monotonically decreasing
        # for a discharge
        np.testing.assert_array_less(
            model.variables["c"].evaluate(T, Y)[:, 1:],
            model.variables["c"].evaluate(T, Y)[:, :-1],
        )
        np.testing.assert_array_less(
            model.variables["V"].evaluate(T, Y)[1:],
            model.variables["V"].evaluate(T, Y)[:-1],
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
