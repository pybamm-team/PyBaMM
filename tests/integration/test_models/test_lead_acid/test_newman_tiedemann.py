#
# Tests for the lead-acid Newman-Tiedemann model
#
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
import tests

import unittest
import numpy as np


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestLeadAcidNewmanTiedemann(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 0.6))

    def test_optimisations(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        np.testing.assert_array_almost_equal(original, simplified)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
