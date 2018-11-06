from pybat_lead_acid.mesh import *
from pybat_lead_acid.parameters import *

import numpy as np
import unittest

class TestParameters(unittest.TestCase):

    def test_parameters(self):
        # basic tests on how the parameters interact
        param = Parameters()
        self.assertAlmostEqual(param.ln + param.ls + param.lp, 1, places=10)

if __name__ == "__main__":
    unittest.main()
