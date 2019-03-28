#
# Tests for the Reaction diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import numpy as np
import unittest


class TestReactionDiffusionModel(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.ReactionDiffusionModel()

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_convergence(self):
        # Convergence of c at x=0.5
        model = pybamm.ReactionDiffusionModel()
        # Make ln and lp nicer for testing
        param = model.default_parameter_values
        param.update(
            {
                "Negative electrode width": 4e-4,
                "Separator width": 2e-4,
                "Positive electrode width": 4e-4,
            }
        )
        # Process model and geometry
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        # Set up solver
        t_eval = np.linspace(0, 1, 100)
        solver = model.default_solver

        # Function for convergence testing
        def get_l2_error(n):
            # Set up discretisation
            submesh_pts = {
                "negative electrode": {"x": n},
                "separator": {"x": n},
                "positive electrode": {"x": n},
                "negative particle": {"r": 1},
                "positive particle": {"r": 1},
            }
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, submesh_pts)
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            import ipdb

            ipdb.set_trace()

            # Discretise and solve
            disc.process_model(model)
            solver.solve(model, t_eval)
            disc.set_variable_slices([var])
            c_approx = model.variables["Electrolyte concentration"].evaluate(
                solver.t, solver.y
            )

            # Calculate errors
            return np.linalg.norm(c_approx - c_exact) / np.linalg.norm(c_exact)

        # Get errors
        ns = 100 * (2 ** np.arange(2, 7))
        errs = np.array([get_l2_error(int(n)) for n in ns])

        # Get rates: expect h**2 convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
