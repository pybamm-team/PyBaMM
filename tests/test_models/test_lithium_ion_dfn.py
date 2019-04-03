#
# Tests for the lithium-ion DFN model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests
import numpy as np
import unittest


class TestDFN(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lithium_ion.DFN()
        var = pybamm.standard_spatial_vars
        self.default_var_pts = {
            var.x_n: 3,
            var.x_s: 3,
            var.x_p: 3,
            var.r_n: 1,
            var.r_p: 1,
        }

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
