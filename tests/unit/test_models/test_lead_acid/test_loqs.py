#
# Tests for the lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidLOQS(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.LOQS()
        modeltest = tests.StandardModelTest(model)

        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.lead_acid.LOQS()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        np.testing.assert_array_almost_equal(original, simplified)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known)

    def test_solution(self):
        model = pybamm.lead_acid.LOQS()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 2))
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

    def test_charge(self):
        model = pybamm.lead_acid.LOQS()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Typical current density": -1})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y
        # check surface concentration increases in negative particle and
        # decreases in positive particle for charge
        processed_variables = pybamm.post_process_variables(
            model.variables, t_sol, y_sol, mesh=modeltest.disc.mesh
        )
        conc = processed_variables["Electrolyte concentration"]
        voltage = processed_variables["Terminal voltage"]
        # neg surf concentration should be monotonically increasing for a charge
        np.testing.assert_array_less(c_e.entries[:, :-1], c_e.entries[:, 1:])
        np.testing.assert_array_less(voltage.entries[:-1], voltage.entries[1:])

    def test_zero_current(self):
        model = pybamm.lead_acid.LOQS()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Typical current density": 0})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y
        # check surface concentration increases in negative particle and
        # decreases in positive particle for charge
        processed_variables = pybamm.post_process_variables(
            model.variables, t_sol, y_sol, mesh=modeltest.disc.mesh
        )
        conc = processed_variables["Electrolyte concentration"]
        voltage = processed_variables["Terminal voltage"]
        # variables should remain unchanged
        np.testing.assert_almost_equal(c_e.entries - c_e.entries[0], 0)
        np.testing.assert_almost_equal(voltage.entries - voltage.entries[0], 0)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
