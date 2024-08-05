import pybamm
import unittest
from tests import get_mesh_for_testing

import sys
import numpy as np

if pybamm.have_jax():
    import jax


@unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
class TestJaxSolver(unittest.TestCase):
    def test_model_solver(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1.0}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        for method in ["RK45", "BDF"]:
            # Solve
            solver = pybamm.JaxSolver(method=method, rtol=1e-8, atol=1e-8)
            t_eval = np.linspace(0, 1, 80)
            solution = solver.solve(model, t_eval)
            np.testing.assert_array_equal(solution.t, t_eval)
            np.testing.assert_allclose(
                solution.y[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-6
            )

            # Test time
            self.assertEqual(
                solution.total_time, solution.solve_time + solution.set_up_time
            )
            self.assertEqual(solution.termination, "final time")

            second_solution = solver.solve(model, t_eval)

            np.testing.assert_array_equal(second_solution.y, solution.y)

    def test_semi_explicit_model(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        var2 = pybamm.Variable("var2", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.algebraic = {var2: var2 - 2.0 * var}
        # give inconsistent initial conditions, should calculate correct ones
        model.initial_conditions = {var: 1.0, var2: 1.0}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Solve
        solver = pybamm.JaxSolver(method="BDF", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 80)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        soln = np.exp(0.1 * solution.t)
        np.testing.assert_allclose(solution.y[0], soln, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(solution.y[-1], 2 * soln, rtol=1e-7, atol=1e-7)

        # Test time
        self.assertEqual(
            solution.total_time, solution.solve_time + solution.set_up_time
        )
        self.assertEqual(solution.termination, "final time")

        second_solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(second_solution.y, solution.y)

    def test_solver_sensitivities(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1.0}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        for method in ["RK45", "BDF"]:
            # Solve
            solver = pybamm.JaxSolver(method=method, rtol=1e-8, atol=1e-8)
            t_eval = np.linspace(0, 1, 80)

            h = 0.0001
            rate = 0.1

            # need to solve the model once to get it set up by the base solver
            solver.solve(model, t_eval, inputs={"rate": rate})
            solve = solver.get_solve(model, t_eval)

            # create a dummy "model" where we calculate the sum of the time series
            def solve_model(rate, solve=solve):
                return jax.numpy.sum(solve({"rate": rate}))

            # check answers with finite difference
            eval_plus = solve_model(rate + h)
            eval_neg = solve_model(rate - h)
            grad_num = (eval_plus - eval_neg) / (2 * h)

            grad_solve = jax.jit(jax.grad(solve_model))
            grad = grad_solve(rate)

            self.assertAlmostEqual(grad, grad_num, places=1)

    def test_solver_only_works_with_jax(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: -pybamm.sqrt(var)}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 3, 100)

        # solver needs a model converted to jax
        for convert_to_format in ["casadi", "python", "something_else"]:
            model.convert_to_format = convert_to_format

            solver = pybamm.JaxSolver()
            with self.assertRaisesRegex(RuntimeError, "must be converted to JAX"):
                solver.solve(model, t_eval)

    def test_solver_doesnt_support_events(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -0.1 * var}
        model.initial_conditions = {var: 1}
        # needs to work with multiple events (to avoid bug where only last event is
        # used)
        model.events = [
            pybamm.Event("var=0.5", pybamm.min(var - 0.5)),
            pybamm.Event("var=-0.5", pybamm.min(var + 0.5)),
        ]
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.JaxSolver()
        t_eval = np.linspace(0, 10, 100)
        with self.assertRaisesRegex(RuntimeError, "Terminate events not supported"):
            solver.solve(model, t_eval)

    def test_model_solver_with_inputs(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial
        # operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.JaxSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 80)
        solution = solver.solve(model, t_eval, inputs={"rate": 0.1})

        np.testing.assert_allclose(
            solution.y[0], np.exp(-0.1 * solution.t), rtol=1e-6, atol=1e-6
        )

        solution = solver.solve(model, t_eval, inputs={"rate": 0.2})

        np.testing.assert_allclose(
            solution.y[0], np.exp(-0.2 * solution.t), rtol=1e-6, atol=1e-6
        )

    def test_get_solve(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial
        # operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # test that another method string gives error
        with self.assertRaises(ValueError):
            solver = pybamm.JaxSolver(method="not_real")

        # Solve
        solver = pybamm.JaxSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 80)

        with self.assertRaisesRegex(RuntimeError, "Model is not set up for solving"):
            solver.get_solve(model, t_eval)

        solver.solve(model, t_eval, inputs={"rate": 0.1})
        solver = solver.get_solve(model, t_eval)
        y = solver({"rate": 0.1})

        np.testing.assert_allclose(y[0], np.exp(-0.1 * t_eval), rtol=1e-6, atol=1e-6)

        y = solver({"rate": 0.2})

        np.testing.assert_allclose(y[0], np.exp(-0.2 * t_eval), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
