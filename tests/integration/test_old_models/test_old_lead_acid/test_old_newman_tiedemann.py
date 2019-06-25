#
# Tests for the lead-acid Newman-Tiedemann model
#
import pybamm
import tests

import unittest
import numpy as np


class TestOldLeadAcidNewmanTiedemann(unittest.TestCase):
    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_basic_processing(self):
        options = {"thermal": None, "Voltage": "On"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 0.6))

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_basic_processing_with_convection(self):
        options = {"thermal": None, "Voltage": "On", "convection": True}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10}
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(t_eval=np.linspace(0, 0.6))

    def test_optimisations(self):
        options = {"thermal": None, "Voltage": "On"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
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


class TestOldLeadAcidNewmanTiedemannCapacitance(unittest.TestCase):
    def test_basic_processing_differential(self):
        options = {"capacitance": "differential"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_basic_processing_algebraic(self):
        options = {"capacitance": "algebraic"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        options = {"capacitance": "differential"}
        model = pybamm.lead_acid.NewmanTiedemann(options)
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        np.testing.assert_array_almost_equal(original, simplified, decimal=5)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known, decimal=5)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
