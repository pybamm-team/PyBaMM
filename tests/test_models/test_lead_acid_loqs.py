#
# Tests for the lead-acid LOQS model
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
        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y

        # Post-process variables
        conc = pybamm.ProcessedVariable(
            model.variables["Electrolyte concentration"],
            t_sol,
            y_sol,
            mesh=modeltest.disc.mesh,
        )
        voltage = pybamm.ProcessedVariable(model.variables["Voltage"], t_sol, y_sol)

        # check output
        # concentration and voltage should be monotonically decreasing for a discharge
        np.testing.assert_array_less(conc.entries[:, 1:], conc.entries[:, :-1])
        np.testing.assert_array_less(voltage.entries[1:], voltage.entries[:-1])
        # Make sure the concentration is always positive (cut-off event working)
        np.testing.assert_array_less(0, conc.entries)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
