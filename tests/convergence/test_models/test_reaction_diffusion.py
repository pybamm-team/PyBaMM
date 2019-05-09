#
# Tests for the Reaction diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
import unittest


class TestReactionDiffusionModel(unittest.TestCase):
    def test_convergence(self):
        # Convergence of c at x=0.5
        model = pybamm.ReactionDiffusionModel()
        # # Make ln and lp nicer for testing
        param = model.default_parameter_values
        # Process model and geometry
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        # Set up solver
        t_eval = np.linspace(0, 1)
        solver = model.default_solver

        # Function for convergence testing
        def get_concs(n):
            # Set up discretisation
            var = pybamm.standard_spatial_vars
            submesh_pts = {var.x_n: n, var.x_s: n, var.x_p: n, var.r_n: 1, var.r_p: 1}
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, submesh_pts)
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)

            # Discretise and solve
            model_disc = disc.process_model(model, inplace=False)
            solver.solve(model_disc, t_eval)
            t, y = solver.t, solver.y
            return pybamm.ProcessedVariable(
                model_disc.variables["Electrolyte concentration"], t, y, mesh=disc.mesh
            )

        # Get concentrations
        ns = 2 ** np.arange(8)
        concs = [get_concs(int(n)) for n in ns]

        # Test the value at a range of times
        for t in np.linspace(0.01, 1, 5):
            # Get errors at the points from the coarsest solution
            x = concs[0].x_sol
            # Use inf norm (max abs error), comparing each solution to the finest sol
            errs = np.array(
                [
                    np.linalg.norm(concs[i](t, x) - concs[-1](t, x), np.inf)
                    for i in range(len(ns) - 1)
                ]
            )
            # Get rates: expect h**2 convergence
            rates = np.log2(errs[:-1] / errs[1:])
            np.testing.assert_array_less(1.99 * np.ones_like(rates), rates)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
