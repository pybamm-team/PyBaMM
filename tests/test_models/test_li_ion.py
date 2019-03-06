#
# Tests for the li-ion models
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest
import numpy as np


class TestLiIonSPM(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.li_ion.SPM()
        modeltest = tests.StandardModelTest(model)

        modeltest.test_all()

    def test_surface_concentrartion(self):
        model = pybamm.li_ion.SPM()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
        T, Y = modeltest.solver.t, modeltest.solver.y

        # check surface concentration decreases in negative particle and
        # increases in positive particle for discharge
        np.testing.assert_array_less(
            model.variables["cn_surf"].evaluate(T, Y)[:, 1:],
            model.variables["cn_surf"].evaluate(T, Y)[:, :-1],
        )
        np.testing.assert_array_less(
            model.variables["cp_surf"].evaluate(T, Y)[:, :-1],
            model.variables["cp_surf"].evaluate(T, Y)[:, 1:],
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
