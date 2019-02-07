#
# Tests for the lead-acid models
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidLOQS(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.LeadAcidLOQS()

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    @unittest.skip("")
    def test_solution(self):
        model = pybamm.LeadAcidLOQS()

        # process parameter values, discretise and solve
        model.default_parameter_values.process_model(model)
        disc = model.default_discretisation
        disc.process_model(model)
        t_eval = disc.mesh["time"]
        solver = model.default_solver
        solver.solve(model, t_eval)
        T, Y = solver.t, solver.y

        # check output
        import ipdb

        ipdb.set_trace()
        np.testing.assert_array_almost_equal(
            model.variables["c"].evaluate(T, Y)[0], -T[np.newaxis, :]
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
