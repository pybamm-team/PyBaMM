#
# Tests for the Processed Variable class
#
import pybamm
import tests

import numpy as np
import unittest


class TestProcessedCasadiVariable(unittest.TestCase):
    def test_processed_variable_0D(self):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = t * y
        var.mesh = None
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        processed_var = pybamm.ProcessedVariable(var, pybamm.Solution(t_sol, y_sol))
        np.testing.assert_array_equal(processed_var.entries, t_sol * y_sol[0])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
