#
# Tests for the Casadi Solver Mode
#
from tests import TestCase
import pybamm
import unittest
import numpy as np


class TestCasadiMode(TestCase):
    def test_casadi_solver_mode(self):
        solvers = [
            pybamm.CasadiSolver(mode="fast", atol=1e-8, rtol=1e-8),
            pybamm.CasadiSolver(mode="safe", atol=1e-8, rtol=1e-8),
            pybamm.CasadiSolver(mode="fast with events", atol=1e-8, rtol=1e-8),
        ]
        solutions = []
        # for model in models:
        for solver in solvers:
            model = pybamm.lithium_ion.SPM()
            # create geometry
            geometry = model.default_geometry

            # load parameter values and process model and geometry
            param = model.default_parameter_values
            param.process_model(model)
            param.process_geometry(geometry)

            # set mesh
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )

            # discretise model
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)

            # solve model
            t_eval = np.linspace(0, 3600, 100)
            solutions.append(solver.solve(model, t_eval))

        output_vars = [
            "Terminal voltage [V]",
            "Current [A]",
            "Electrolyte concentration [mol.m-3]",
        ]
        for var in output_vars:
            np.testing.assert_allclose(
                solutions[0][var].data, solutions[1][var].data, rtol=1e-8, atol=1e-8
            )
            np.testing.assert_allclose(
                solutions[0][var].data, solutions[2][var].data, rtol=1e-8, atol=1e-8
            )
            np.testing.assert_allclose(
                solutions[1][var].data, solutions[2][var].data, rtol=1e-8, atol=1e-8
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
