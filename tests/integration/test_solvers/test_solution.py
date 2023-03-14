#
# Tests for the Solution class
#
import pybamm
import unittest
import numpy as np


class TestSolution(unittest.TestCase):
    def test_append(self):
        model = pybamm.lithium_ion.SPMe()
        # create geometry
        geometry = model.default_geometry

        # load parameter values and process model and geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)

        # set mesh
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # solve model
        t_eval = np.linspace(0, 3600, 100)
        solver = model.default_solver
        solution = solver.solve(model, t_eval)

        # step model
        old_t = 0
        step_solver = model.default_solver
        step_solution = None
        # dt should be dimensional
        for t in solution.t[1:]:
            dt = t - old_t
            step_solution = step_solver.step(step_solution, model, dt=dt, npts=10)
            if t == solution.t[1]:
                # Create voltage variable
                step_solution.update("Voltage [V]")
            old_t = t

        # Check both give the same answer
        np.testing.assert_array_almost_equal(
            solution["Voltage [V]"](solution.t[:-1]),
            step_solution["Voltage [V]"](solution.t[:-1]),
            decimal=4,
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
