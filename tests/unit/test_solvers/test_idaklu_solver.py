#
# Tests for the KLU Solver class
#
import pybamm
import numpy as np
import unittest


@unittest.skipIf(not pybamm.have_idaklu(), "idaklu solver is not installed")
class TestIDAKLUSolver(unittest.TestCase):
    def test_ida_roberts_klu(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 0.1 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 1}
        model.events = {"1": u - 0.2, "2": v}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()

        t_eval = np.linspace(0, 3, 100)
        solution = solver.solve(model, t_eval)

        # test that final time is time of event
        # y = 0.1 t + y0 so y=0.2 when t=2
        np.testing.assert_array_almost_equal(solution.t[-1], 2.0)

        # test that final value is the event value
        np.testing.assert_array_almost_equal(solution.y[0, -1], 0.2)

        # test that y[1] remains constant
        np.testing.assert_array_almost_equal(
            solution.y[1, :], np.ones(solution.t.shape)
        )

        # test that y[0] = to true solution
        true_solution = 0.1 * solution.t
        np.testing.assert_array_almost_equal(solution.y[0, :], true_solution)

    def test_set_atol(self):
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        solver = pybamm.IDAKLUSolver()

        variable_tols = {"Electrolyte concentration": 1e-3}
        solver.set_atol_by_variable(variable_tols, model)

    def test_model_step_events(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = {
            "var1 = 1.5": pybamm.min(var1 - 1.5),
            "var2 = 2.5": pybamm.min(var2 - 2.5),
        }
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        step_solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
        dt = 0.05
        time = 0
        end_time = 5
        step_solution = None
        while time < end_time:
            step_solution = step_solver.step(step_solution, model, dt=dt, npts=10)
            time += dt
        np.testing.assert_array_less(step_solution.y[0], 1.5)
        np.testing.assert_array_less(step_solution.y[-1], 2.5001)
        np.testing.assert_array_almost_equal(
            step_solution.y[0], np.exp(0.1 * step_solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            step_solution.y[-1], 2 * np.exp(0.1 * step_solution.t), decimal=5
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
