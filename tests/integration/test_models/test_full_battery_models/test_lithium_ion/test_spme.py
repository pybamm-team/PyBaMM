#
# Tests for the lithium-ion SPMe model
#
import pybamm
import tests

import numpy as np
import unittest


class TestSPMe(unittest.TestCase):
    def test_basic_processing(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPMe(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    @unittest.skipIf(pybamm.have_scikit_fem(), "scikits.odes not installed")
    def test_basic_processing_2plus1D(self):
        options = {"bc_options": {"dimensionality": 2}}
        model = pybamm.lithium_ion.SPMe(options)
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 5,
            var.x_s: 5,
            var.x_p: 5,
            var.r_n: 10,
            var.r_p: 10,
            var.y: 8,
            var.z: 8,
        }
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

    def test_optimisations(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPMe(options)
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

    def test_thermal(self):
        pybamm.settings.debug_mode = True
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

        options = {"thermal": "full"}
        model = pybamm.lithium_ion.SPMe(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
