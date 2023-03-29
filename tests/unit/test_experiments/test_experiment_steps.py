#
# Test the base experiment class
#
import pybamm
import numpy as np
import unittest
import pandas as pd
import os


class TestExperimentSteps(unittest.TestCase):
    def test_cc(self):
        expcc = pybamm.CC(1)
        self.assertEqual(expcc.c_rate, 1)
        self.assertEqual(expcc.temperature, None)
        self.assertEqual(expcc.duration, None)
        self.assertEqual(expcc.upper_cutoff, None)

        expccall = pybamm.CC(1, 298, 3600, 4.2)
        self.assertEqual(expccall.c_rate, 1)
        self.assertEqual(expccall.temperature, 298)
        self.assertEqual(expccall.duration, 3600)
        self.assertEqual(expccall.upper_cutoff, 4.2)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
