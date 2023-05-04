import pybamm
import unittest
from tests import get_mesh_for_testing
from tests import TestCase
import sys
import time
import numpy as np

if pybamm.have_jax():
    import jax


@unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
class TestJaxBDFSolver(TestCase):
    def test_solver_(self):  # Trailing _ manipulates the random seed
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Solve
        t_eval = np.linspace(0.0, 1.0, 80)
        y0 = model.concatenated_initial_conditions.evaluate().reshape(-1)
        rhs = pybamm.EvaluatorJax(model.concatenated_rhs)

        def fun(y, t):
            return rhs(t=t, y=y).reshape(-1)

        t0 = time.perf_counter()
        y = pybamm.jax_bdf_integrate(fun, y0, t_eval, rtol=1e-8, atol=1e-8)
        t1 = time.perf_counter() - t0

        # test accuracy
        np.testing.assert_allclose(y[:, 0], np.exp(0.1 * t_eval), rtol=1e-6, atol=1e-6)

        t0 = time.perf_counter()
        y = pybamm.jax_bdf_integrate(fun, y0, t_eval, rtol=1e-8, atol=1e-8)
        t2 = time.perf_counter() - t0

        # second run should be much quicker
        self.assertLess(t2, t1)

        # test second run is accurate
        np.testing.assert_allclose(y[:, 0], np.exp(0.1 * t_eval), rtol=1e-6, atol=1e-6)

    def test_mass_matrix(self):
        # Solve
        t_eval = np.linspace(0.0, 1.0, 80)

        def fun(y, t):
            return jax.numpy.stack([0.1 * y[0], y[1] - 2.0 * y[0]])

        mass = jax.numpy.array([[2.0, 0.0], [0.0, 0.0]])

        # give some bad initial conditions, solver should calculate correct ones using
        # this as a guess
        y0 = jax.numpy.array([1.0, 1.5])

        t0 = time.perf_counter()
        y = pybamm.jax_bdf_integrate(fun, y0, t_eval, mass=mass, rtol=1e-8, atol=1e-8)
        t1 = time.perf_counter() - t0

        # test accuracy
        soln = np.exp(0.05 * t_eval)
        np.testing.assert_allclose(y[:, 0], soln, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(y[:, 1], 2.0 * soln, rtol=1e-7, atol=1e-7)

        t0 = time.perf_counter()
        y = pybamm.jax_bdf_integrate(fun, y0, t_eval, mass=mass, rtol=1e-8, atol=1e-8)
        t2 = time.perf_counter() - t0

        # second run should be much quicker
        self.assertLess(t2, t1)

        # test second run is accurate
        np.testing.assert_allclose(y[:, 0], np.exp(0.05 * t_eval), rtol=1e-7, atol=1e-7)

    def test_solver_sensitivities(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}

        # create discretisation
        mesh = get_mesh_for_testing(xpts=10)
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Solve
        t_eval = np.linspace(0, 10, 4)
        y0 = model.concatenated_initial_conditions.evaluate().reshape(-1)
        rhs = pybamm.EvaluatorJax(model.concatenated_rhs)

        def fun(y, t, inputs):
            return rhs(t=t, y=y, inputs=inputs).reshape(-1)

        h = 0.0001
        rate = 0.1

        # create a dummy "model" where we calculate the sum of the time series
        @jax.jit
        def solve_bdf(rate):
            return jax.numpy.sum(
                pybamm.jax_bdf_integrate(
                    fun, y0, t_eval, {"rate": rate}, rtol=1e-9, atol=1e-9
                )
            )

        # check answers with finite difference
        eval_plus = solve_bdf(rate + h)
        eval_neg = solve_bdf(rate - h)
        grad_num = (eval_plus - eval_neg) / (2 * h)

        grad_solve_bdf = jax.jit(jax.grad(solve_bdf))
        grad_bdf = grad_solve_bdf(rate)

        self.assertAlmostEqual(grad_bdf, grad_num, places=3)

    def test_mass_matrix_with_sensitivities(self):
        # Solve
        t_eval = np.linspace(0.0, 1.0, 80)

        def fun(y, t, inputs):
            return jax.numpy.stack([inputs["rate"] * y[0], y[1] - 2.0 * y[0]])

        mass = jax.numpy.array([[2.0, 0.0], [0.0, 0.0]])

        y0 = jax.numpy.array([1.0, 2.0])

        h = 0.0001
        rate = 0.1

        # create a dummy "model" where we calculate the sum of the time series
        @jax.jit
        def solve_bdf(rate):
            return jax.numpy.sum(
                pybamm.jax_bdf_integrate(
                    fun, y0, t_eval, {"rate": rate}, mass=mass, rtol=1e-9, atol=1e-9
                )
            )

        # check answers with finite difference
        eval_plus = solve_bdf(rate + h)
        eval_neg = solve_bdf(rate - h)
        grad_num = (eval_plus - eval_neg) / (2 * h)

        grad_solve_bdf = jax.jit(jax.grad(solve_bdf))
        grad_bdf = grad_solve_bdf(rate)

        self.assertAlmostEqual(grad_bdf, grad_num, places=3)

    def test_solver_with_inputs(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Solve
        t_eval = np.linspace(0, 10, 80)
        y0 = model.concatenated_initial_conditions.evaluate().reshape(-1)
        rhs = pybamm.EvaluatorJax(model.concatenated_rhs)

        def fun(y, t, inputs):
            return rhs(t=t, y=y, inputs=inputs).reshape(-1)

        y = pybamm.jax_bdf_integrate(
            fun, y0, t_eval, {"rate": 0.1}, rtol=1e-9, atol=1e-9
        )

        np.testing.assert_allclose(y[:, 0].reshape(-1), np.exp(-0.1 * t_eval))


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
