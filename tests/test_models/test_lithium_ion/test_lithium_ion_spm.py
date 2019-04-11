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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
