#
# Tests for the lead-acid Newman-Tiedemann model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
import tests

import unittest
import numpy as np


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestLeadAcidNewmanTiedemann(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        # Make grid very coarse for quick test (note that r domain doesn't matter)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 3, var.x_s: 3, var.x_p: 3, var.r_n: 1, var.r_p: 1}
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(t_eval=np.linspace(0, 0.1, 5))

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

    def test_solution(self):
        model = pybamm.lead_acid.NewmanTiedemann()
        # Make grid very coarse for quick test (note that r domain doesn't matter)
        var = pybamm.standard_spatial_vars
        model.default_var_pts = {
            var.x_n: 3,
            var.x_s: 3,
            var.x_p: 3,
            var.r_n: 1,
            var.r_p: 1,
        }
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 0.1, 5))
        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y

        # Post-process variables
        processed_variables = pybamm.post_process_variables(
            model.variables, t_sol, y_sol, mesh=modeltest.disc.mesh
        )
        conc = processed_variables["Electrolyte concentration"]
        voltage = processed_variables["Terminal voltage"]

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
