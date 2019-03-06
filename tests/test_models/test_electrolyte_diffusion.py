#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest


class TestStefanMaxwellDiffusion(unittest.TestCase):
    def test_make_tree(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        G = pybamm.Scalar(1)
        c_e = pybamm.Variable("c_e", domain=whole_cell)
        pybamm.electrolyte_diffusion.StefanMaxwell(c_e, G)

    def test_basic_processing(self):
        G = pybamm.Scalar(0.001)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c_e = pybamm.Variable("c_e", domain=whole_cell)
        model = pybamm.electrolyte_diffusion.StefanMaxwell(c_e, G)

        modeltest = tests.StandardModelTest(model)
        # Either
        # 1. run the tests individually (can pass options to individual tests)
        # modeltest.test_processing_parameters()
        # modeltest.test_processing_disc()
        # modeltest.test_solving()

        # Or
        # 2. run all the tests in one go
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
