#
# Tests for the lithium-ion DFN model
#
import pybamm
import tests

import numpy as np
import unittest


pybamm.settings.debug_mode = True


@unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
class TestDFN(unittest.TestCase):
    def test_basic_processing(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.DFN(options)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all()

    def test_optimisations(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.DFN(options)
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        simp_and_python = optimtest.evaluate_model(simplify=True, to_python=True)
        np.testing.assert_array_almost_equal(original, simplified)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known)

        np.testing.assert_array_almost_equal(original, simp_and_python)

    def test_full_thermal(self):
        options = {"thermal": "full"}
        model = pybamm.lithium_ion.DFN(options)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all()

    def test_lumped_thermal(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.DFN(options)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
