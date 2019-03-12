#
# Tests for the lead-acid composite model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidComposite(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.Composite()

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_solution(self):
        model = pybamm.lead_acid.Composite()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
        T, Y = modeltest.solver.t, modeltest.solver.y

        # check output
        for idx in range(len(T) - 1):
            # Check concentration decreases
            np.testing.assert_array_less(
                model.variables["c"].evaluate(T[idx + 1], Y[:, idx + 1]),
                model.variables["c"].evaluate(T[idx], Y[:, idx]),
            )
            # Check cut-off
            np.testing.assert_array_less(
                0, model.variables["c"].evaluate(T[idx + 1], Y[:, idx + 1])
            )
            try:
                self.assertLess(
                    model.variables["V"].evaluate(T[idx + 1], Y[:, idx + 1]),
                    model.variables["V"].evaluate(T[idx], Y[:, idx]),
                )
            except AssertionError:
                import ipdb

                ipdb.set_trace()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
