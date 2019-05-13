#
# Tests for the lead-acid Newman-Tiedemann capacitance model
#
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
import tests

import unittest
import numpy as np


class TestLeadAcidNewmanTiedemannCapacitance(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.NewmanTiedemannCapacitance()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(skip_output_tests=True)

    @unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
    def test_basic_processing_no_capacitance(self):
        model = pybamm.lead_acid.NewmanTiedemannCapacitance(use_capacitance=False)
        modeltest = tests.StandardModelTest(model)

        modeltest.test_all(skip_output_tests=True)

    def test_optimisations(self):
        model = pybamm.lead_acid.NewmanTiedemannCapacitance()
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
