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

    @unittest.skip("bug in SPM")
    def test_surface_concentration(self):
        model = pybamm.li_ion.SPM()
        params = model.default_parameter_values
        params.process_model(model)
        disc = model.default_discretisation
        disc.process_model(model)
        t_eval = np.linspace(0, 1, 100)
        solver = model.default_solver
        solver.solve(model, t_eval)
        T, Y = solver.t, solver.y

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
