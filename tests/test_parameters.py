from pybamm.mesh import *
from pybamm.parameters import *

import numpy as np
import unittest

class TestParameters(unittest.TestCase):

    def test_parameters_defaults(self):
        # basic tests on how the parameters interact
        param = Parameters()
        self.assertAlmostEqual(param.ln + param.ls + param.lp, 1, places=10)

    def test_parameters_options(self):
        param = Parameters(
            optional_parameters={'Ln': 1/3, 'Ls': 1/3, 'Lp': 1/3})
        self.assertAlmostEqual(param.ln + param.ls + param.lp, 1, places=10)
        param = Parameters(
            optional_parameters='input/optional_parameters_test.csv')
        self.assertAlmostEqual(param.ln + param.ls + param.lp, 1, places=10)

if __name__ == "__main__":
    unittest.main()
