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
        for form in ["python", "casadi"]:
            model = pybamm.BaseModel()
            model.convert_to_format = form
            u = pybamm.Variable("u")
            v = pybamm.Variable("v")
            model.rhs = {u: 0.1 * v}
            model.algebraic = {v: 1 - v}
            model.initial_conditions = {u: 0, v: 1}
            model.events = [pybamm.Event("1", u - 0.2), pybamm.Event("2", v)]

            disc = pybamm.Discretisation()
            disc.process_model(model)

            solver = pybamm.IDAKLUSolver(root_method="lm")

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

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_ida_roberts_klu_sensitivities(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        for form in ["python", "casadi", "jax"]:
            print(form)
            model = pybamm.BaseModel()
            model.convert_to_format = form
            u = pybamm.Variable("u")
            v = pybamm.Variable("v")
            a = pybamm.InputParameter("a")
            model.rhs = {u: a * v}
            model.algebraic = {v: 1 - v}
            model.initial_conditions = {u: 0, v: 1}

            disc = pybamm.Discretisation()
            disc.process_model(model)

            solver = pybamm.IDAKLUSolver(root_method="lm")

            t_eval = np.linspace(0, 3, 100)
            a_value = 0.1

            # solve first without sensitivities
            sol = solver.solve(
                model, t_eval, inputs={"a": a_value},
            )

            # test that y[1] remains constant
            np.testing.assert_array_almost_equal(
                sol.y[1, :], np.ones(sol.t.shape)
            )

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[0, :], true_solution)

            # should be no sensitivities calculated
            with self.assertRaises(KeyError):
                print(sol.sensitivities["a"])

            # now solve with sensitivities (this should cause set_up to be run again)
            sol = solver.solve(
                model, t_eval, inputs={"a": a_value},
                calculate_sensitivities=True
            )

            # test that y[1] remains constant
            np.testing.assert_array_almost_equal(
                sol.y[1, :], np.ones(sol.t.shape)
            )

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[0, :], true_solution)

            # evaluate the sensitivities using idas
            dyda_ida = sol.sensitivities["a"]

            # evaluate the sensitivities using finite difference
            h = 1e-6
            sol_plus = solver.solve(model, t_eval, inputs={"a": a_value + 0.5 * h})
            sol_neg = solver.solve(model, t_eval, inputs={"a": a_value - 0.5 * h})
            dyda_fd = (sol_plus.y - sol_neg.y) / h
            dyda_fd = dyda_fd.transpose().reshape(-1, 1)

            np.testing.assert_array_almost_equal(
                dyda_ida, dyda_fd
            )

    def test_set_atol(self):
        model = pybamm.lithium_ion.DFN()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        solver = pybamm.IDAKLUSolver(root_method="lm")

        variable_tols = {"Porosity times concentration": 1e-3}
        solver.set_atol_by_variable(variable_tols, model)

        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        model.rhs = {u: -0.1 * u}
        model.initial_conditions = {u: 1}
        t_eval = np.linspace(0, 3, 100)

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # numpy array atol
        atol = np.zeros(1)
        solver = pybamm.IDAKLUSolver(root_method="lm", atol=atol)
        solver.solve(model, t_eval)

        # list atol
        atol = [1]
        solver = pybamm.IDAKLUSolver(root_method="lm", atol=atol)
        solver.solve(model, t_eval)

        # wrong size (should fail)
        atol = [1, 2]
        solver = pybamm.IDAKLUSolver(root_method="lm", atol=atol)
        with self.assertRaisesRegex(pybamm.SolverError, 'Absolute tolerances'):
            solver.solve(model, t_eval)

    def test_failures(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        model = pybamm.BaseModel()
        model.use_jacobian = False
        u = pybamm.Variable("u")
        model.rhs = {u: -0.1 * u}
        model.initial_conditions = {u: 1}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver(root_method="lm")

        t_eval = np.linspace(0, 3, 100)
        with self.assertRaisesRegex(pybamm.SolverError, "KLU requires the Jacobian"):
            solver.solve(model, t_eval)

        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        model.rhs = {u: -0.1 * u}
        model.initial_conditions = {u: 1}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver(root_method="lm")

        # will give solver error
        t_eval = np.linspace(0, -3, 100)
        with self.assertRaisesRegex(
            pybamm.SolverError, 't_eval must increase monotonically'
        ):
            solver.solve(model, t_eval)

    def test_dae_solver_algebraic_model(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var + 1}
        model.initial_conditions = {var: 0}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()
        t_eval = np.linspace(0, 1)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.y, -1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
