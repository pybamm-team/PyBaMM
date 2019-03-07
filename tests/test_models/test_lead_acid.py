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
        model = pybamm.lead_acid.LOQS()
        modeltest = tests.StandardModelTest(model)

        modeltest.test_all()

    def test_solution(self):
        model = pybamm.lead_acid.LOQS()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 2))
        T, Y = modeltest.solver.t, modeltest.solver.y

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
        # Make sure the concentration is always positive (cut-off event working)
        np.testing.assert_array_less(0, model.variables["c"].evaluate(T, Y))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
