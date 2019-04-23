#
# Tests for the lithium-ion SPM model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests
import numpy as np
import unittest


class TestSPM(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lithium_ion.SPM()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.lithium_ion.SPM()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        np.testing.assert_array_almost_equal(original, simplified)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known)

    def test_surface_concentration(self):
        model = pybamm.lithium_ion.SPM()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y

        # check surface concentration decreases in negative particle and
        # increases in positive particle for discharge
        c_s_n_surf = pybamm.ProcessedVariable(
            model.variables["Negative particle surface concentration"],
            t_sol,
            y_sol,
            mesh=modeltest.disc.mesh,
        )
        c_s_p_surf = pybamm.ProcessedVariable(
            model.variables["Positive particle surface concentration"],
            t_sol,
            y_sol,
            mesh=modeltest.disc.mesh,
        )
        # neg surf concentration should be monotonically decreasing for a discharge
        np.testing.assert_array_less(
            c_s_n_surf.entries[:, 1:], c_s_n_surf.entries[:, :-1]
        )
        # pos surf concentration should be monotonically increasing for a discharge
        np.testing.assert_array_less(
            c_s_p_surf.entries[:, :-1], c_s_p_surf.entries[:, 1:]
        )

        # test that surface concentrations are all positive
        np.testing.assert_array_less(0, c_s_n_surf.entries)
        np.testing.assert_array_less(0, c_s_p_surf.entries)

    def test_charge(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Typical current density": -1})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y
        # check surface concentration increases in negative particle and
        # decreases in positive particle for charge
        c_s_n_surf = pybamm.ProcessedVariable(
            model.variables["Negative particle surface concentration"],
            t_sol,
            y_sol,
            mesh=modeltest.disc.mesh,
        )
        c_s_p_surf = pybamm.ProcessedVariable(
            model.variables["Positive particle surface concentration"],
            t_sol,
            y_sol,
            mesh=modeltest.disc.mesh,
        )
        voltage = pybamm.ProcessedVariable(
            model.variables["Terminal voltage"], t_sol, y_sol
        )
        # neg surf concentration should be monotonically increasing for a charge
        np.testing.assert_array_less(
            c_s_n_surf.entries[:, :-1], c_s_n_surf.entries[:, 1:]
        )
        # pos surf concentration should be monotonically decreasing for a charge
        np.testing.assert_array_less(
            c_s_p_surf.entries[:, 1:], c_s_p_surf.entries[:, :-1]
        )

        # test that surface concentrations are all positive
        np.testing.assert_array_less(c_s_n_surf.entries, 1)
        np.testing.assert_array_less(c_s_p_surf.entries, 1)

        np.testing.assert_array_less(voltage.entries[:-1], voltage.entries[1:])

    def test_zero_current(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Typical current density": 0})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

        t_sol, y_sol = modeltest.solver.t, modeltest.solver.y
        # check surface concentration increases in negative particle and
        # decreases in positive particle for charge
        c_s_n_surf = pybamm.ProcessedVariable(
            model.variables["Negative particle surface concentration"],
            t_sol,
            y_sol,
            mesh=modeltest.disc.mesh,
        )
        c_s_p_surf = pybamm.ProcessedVariable(
            model.variables["Positive particle surface concentration"],
            t_sol,
            y_sol,
            mesh=modeltest.disc.mesh,
        )
        voltage = pybamm.ProcessedVariable(
            model.variables["Terminal voltage"], t_sol, y_sol
        )
        # variables should remain unchanged
        np.testing.assert_almost_equal(c_s_n_surf.entries - c_s_n_surf.entries[:, 0], 0)
        np.testing.assert_almost_equal(c_s_p_surf.entries - c_s_p_surf.entries[:, 0], 0)
        np.testing.assert_almost_equal(voltage.entries - voltage.entries[0], 0)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
