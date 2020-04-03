#
# Tests for the Casadi Solution class
#
import pybamm
import unittest
import numpy as np


class TestCasadiSolution(unittest.TestCase):
    def test_init(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        sol = pybamm.CasadiSolution(t, y)
        np.testing.assert_array_equal(sol.t, t)
        np.testing.assert_array_equal(sol.y, y)
        self.assertEqual(sol.t_event, None)
        self.assertEqual(sol.y_event, None)
        self.assertEqual(sol.termination, "final time")
        self.assertEqual(sol.inputs, {})
        self.assertEqual(sol.model, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
