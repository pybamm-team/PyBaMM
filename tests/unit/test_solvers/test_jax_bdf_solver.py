import pybamm
import unittest
from tests import get_mesh_for_testing
import sys
import time
import numpy as np
from platform import system
import jax


@unittest.skipIf(system() == "Windows", "JAX not supported on windows")
class TestJaxBDFSolver(unittest.TestCase):
    def test_solver(self):
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
        t_eval = np.linspace(0, 1, 80)
        y0 = model.concatenated_initial_conditions.evaluate().reshape(-1)
        rhs = pybamm.EvaluatorJax(model.concatenated_rhs)

        def fun(y, t):
            return rhs.evaluate(t=t, y=y).reshape(-1)

        t0 = time.perf_counter()
        y = pybamm.jax_bdf_integrate(fun, y0, t_eval, rtol=1e-8, atol=1e-8)
        t1 = time.perf_counter() - t0

        # test accuracy
        np.testing.assert_allclose(y[0, :], np.exp(0.1 * t_eval),
                                   rtol=1e-7, atol=1e-7)

        t0 = time.perf_counter()
        y = pybamm.jax_bdf_integrate(fun, y0, t_eval, rtol=1e-8, atol=1e-8)
        t2 = time.perf_counter() - t0

        # second run should be much quicker
        self.assertLess(t2, t1)

        # test second run is accurate
        np.testing.assert_allclose(y[0, :], np.exp(0.1 * t_eval),
                                   rtol=1e-7, atol=1e-7)

    def test_solver_sensitivities(self):
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
            return rhs.evaluate(t=t, y=y, inputs=inputs).reshape(-1)

        grad_integrate = jax.jacfwd(pybamm.jax_bdf_integrate, argnums=3)

        grad = grad_integrate(fun, y0, t_eval, {"rate": 0.1}, rtol=1e-9, atol=1e-9)
        print(grad)

        np.testing.assert_allclose(y[0, :].reshape(-1), np.exp(-0.1 * t_eval))

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
            return rhs.evaluate(t=t, y=y, inputs=inputs).reshape(-1)

        y = pybamm.jax_bdf_integrate(fun, y0, t_eval, {
            "rate": 0.1}, rtol=1e-9, atol=1e-9)

        np.testing.assert_allclose(y[0, :].reshape(-1), np.exp(-0.1 * t_eval))


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
