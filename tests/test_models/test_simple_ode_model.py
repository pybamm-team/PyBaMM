#
# Tests for the simple ODE model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest
import numpy as np


class TestSimpleODEModel(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.SimpleODEModel()

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.SimpleODEModel()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        np.testing.assert_array_almost_equal(original, simplified)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known)

    def test_solution(self):
        model = pybamm.SimpleODEModel()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
        T, Y = modeltest.solver.t, modeltest.solver.y
        mesh = modeltest.disc.mesh
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        combined_submesh = mesh.combine_submeshes(*whole_cell)

        # check output
        np.testing.assert_array_almost_equal(
            model.variables["a"].evaluate(T, Y), 2 * T[np.newaxis, :]
        )
        np.testing.assert_array_almost_equal(
            model.variables["b broadcasted"].evaluate(T, Y),
            np.ones((combined_submesh[0].npts, T.size)),
        )
        np.testing.assert_array_almost_equal(
            model.variables["c broadcasted"].evaluate(T, Y),
            np.ones(
                sum([mesh[d][0].npts for d in ["negative electrode", "separator"]])
            )[:, np.newaxis]
            * np.exp(-T),
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
