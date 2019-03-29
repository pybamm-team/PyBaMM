#
# Tests for the lead-acid Newman-Tiedemann model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidNewmanTiedemann(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        # Make grid very coarse for quick test (note that r domain doesn't matter)
        var = pybamm.standard_spatial_vars
        self.default_var_pts = {
            var.x_n: 3,
            var.x_s: 3,
            var.x_p: 3,
            var.r_n: 1,
            var.r_p: 1,
        }
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 0.1, 5))

    def test_solution(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        # Make grid very coarse for quick test (note that r domain doesn't matter)
        var = pybamm.standard_spatial_vars
        self.default_var_pts = {
            var.x_n: 3,
            var.x_s: 3,
            var.x_p: 3,
            var.r_n: 1,
            var.r_p: 1,
        }
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 0.1, 5))
        T, Y = modeltest.solver.t, modeltest.solver.y

        # check output
        for idx in range(len(T) - 1):
            # Check concentration decreases
            np.testing.assert_array_less(
                model.variables["Electrolyte concentration"].evaluate(
                    T[idx + 1], Y[:, idx + 1]
                ),
                model.variables["Electrolyte concentration"].evaluate(
                    T[idx], Y[:, idx]
                ),
            )
            # Check cut-off
            np.testing.assert_array_less(
                0,
                model.variables["Electrolyte concentration"].evaluate(
                    T[idx + 1], Y[:, idx + 1]
                ),
            )
            self.assertLess(
                model.variables["Voltage"].evaluate(T[idx + 1], Y[:, idx + 1]),
                model.variables["Voltage"].evaluate(T[idx], Y[:, idx]),
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
