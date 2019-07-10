#
# Tests for the lithium-ion SPMe model
#
import pybamm
import tests

import numpy as np
import unittest

pybamm.set_logging_level("DEBUG")
pybamm.settings.debug_mode = True


class TestSPMe(unittest.TestCase):
    # def test_basic_processing(self):
    #     options = {"thermal": None}
    #     model = pybamm.lithium_ion.SPMe(options)
    #     modeltest = tests.StandardModelTest(model)
    #     modeltest.test_all()

    # def test_optimisations(self):
    #     options = {"thermal": None}
    #     model = pybamm.lithium_ion.SPMe(options)
    #     optimtest = tests.OptimisationsTest(model)

    #     original = optimtest.evaluate_model()
    #     simplified = optimtest.evaluate_model(simplify=True)
    #     using_known_evals = optimtest.evaluate_model(use_known_evals=True)
    #     simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
    #     simp_and_python = optimtest.evaluate_model(simplify=True, to_python=True)
    #     np.testing.assert_array_almost_equal(original, simplified)
    #     np.testing.assert_array_almost_equal(original, using_known_evals)
    #     np.testing.assert_array_almost_equal(original, simp_and_known)
    #     np.testing.assert_array_almost_equal(original, simp_and_python)

    def test_thermal(self):
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
