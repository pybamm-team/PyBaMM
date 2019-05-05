#
# Tests for the lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
import tests

import unittest
import numpy as np


class TestLeadAcidLOQSCapacitance(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.LOQSCapacitance()
        modeltest = tests.StandardModelTest(model)

        modeltest.test_all()

    @unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
    def test_basic_processing_no_capacitance(self):
        model = pybamm.lead_acid.LOQSCapacitance(use_capacitance=False)
        modeltest = tests.StandardModelTest(model)

        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.lead_acid.LOQSCapacitance()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        np.testing.assert_array_almost_equal(original, simplified, decimal=5)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known, decimal=5)

    def test_solution(self):
        model = pybamm.lead_acid.LOQSCapacitance()
        modeltest = tests.StandardModelTest(model)

        modeltest.test_all()
        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y

        # Post-process variables
        conc = pybamm.ProcessedVariable(
            model.variables["Electrolyte concentration"],
            t_sol,
            y_sol,
            mesh=modeltest.disc.mesh,
        )
        voltage = pybamm.ProcessedVariable(
            model.variables["Terminal voltage [V]"], t_sol, y_sol
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
