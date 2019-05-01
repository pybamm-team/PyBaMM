#
# Tests for the lead-acid Newman-Tiedemann capacitance model
#
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidNewmanTiedemannCapacitance(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.NewmanTiedemannCapacitance()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()  # t_eval=np.linspace(0, 0.1))

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

    @unittest.skip("not yet working")
    def test_solution(self):
        model = pybamm.lead_acid.NewmanTiedemannCapacitance()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 0.1))
        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y

        # Post-process variables
        conc = pybamm.ProcessedVariable(
            model.variables["Electrolyte concentration"],
            t_sol,
            y_sol,
            mesh=modeltest.disc.mesh,
        )
        voltage = pybamm.ProcessedVariable(
            model.variables["Terminal voltage"], t_sol, y_sol
        )

        # check output
        # concentration and voltage should be monotonically decreasing for a discharge
        np.testing.assert_array_less(conc.entries[:, 1:], conc.entries[:, :-1])
        np.testing.assert_array_less(voltage.entries[1:], voltage.entries[:-1])
        # Make sure the concentration is always positive (cut-off event working)
        np.testing.assert_array_less(0, conc.entries)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
