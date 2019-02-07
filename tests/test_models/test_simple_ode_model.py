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

    def test_output(self):
        model = pybamm.SimpleODEModel()

        # discretise and solve
        disc = model.default_discretisation
        disc.process_model(model)
        t_eval = disc.mesh["time"]
        solver = model.default_solver
        solver.solve(model, t_eval)
        T, Y = solver.t, solver.y

        # check output
        np.testing.assert_array_almost_equal(
            model.variables["a"].evaluate(T, Y), 2 * T[np.newaxis, :]
        )
        np.testing.assert_array_almost_equal(
            model.variables["b broadcasted"].evaluate(T, Y),
            np.ones((disc.mesh["whole cell"].npts, T.size)),
        )
        np.testing.assert_array_almost_equal(
            model.variables["c broadcasted"].evaluate(T, Y),
            np.ones(
                sum([disc.mesh[d].npts for d in ["negative electrode", "separator"]])
            )[:, np.newaxis]
            * np.exp(-T),
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
