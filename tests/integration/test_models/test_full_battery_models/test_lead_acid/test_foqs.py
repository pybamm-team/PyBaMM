#
# Tests for the lead-acid FOQS model
#
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidFOQS(unittest.TestCase):
    def test_basic_processing(self):
        options = {"thermal": None, "convection": False}
        model = pybamm.lead_acid.FOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_basic_processing_with_convection(self):
        options = {"thermal": None, "convection": True}
        model = pybamm.lead_acid.FOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        options = {"thermal": None, "convection": False}
        model = pybamm.lead_acid.FOQS(options)
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


class TestLeadAcidFOQSSurfaceForm(unittest.TestCase):
    def test_basic_processing_differential(self):
        options = {"surface form": "differential", "thermal": None, "convection": False}
        model = pybamm.lead_acid.FOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_basic_processing_algebraic(self):
        options = {"surface form": "algebraic", "thermal": None, "convection": False}
        model = pybamm.lead_acid.FOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        options = {"surface form": "differential", "thermal": None, "convection": False}
        model = pybamm.lead_acid.FOQS(options)
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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
