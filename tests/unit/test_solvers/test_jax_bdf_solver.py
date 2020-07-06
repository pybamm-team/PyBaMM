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
        t_eval = np.linspace(0.0, 1.0, 80)
        y0 = model.concatenated_initial_conditions.evaluate().reshape(-1)
        rhs = pybamm.EvaluatorJax(model.concatenated_rhs)

        def fun(y, t):
            return rhs.evaluate(t=t, y=y).reshape(-1)

        t0 = time.perf_counter()
        y = pybamm.jax_bdf_integrate(fun, y0, t_eval, rtol=1e-8, atol=1e-8)
        t1 = time.perf_counter() - t0

        # test accuracy
        np.testing.assert_allclose(y[:, 0], np.exp(0.1 * t_eval),
                                   rtol=1e-7, atol=1e-7)

        t0 = time.perf_counter()
        y = pybamm.jax_bdf_integrate(fun, y0, t_eval, rtol=1e-8, atol=1e-8)
        t2 = time.perf_counter() - t0

        # second run should be much quicker
        self.assertLess(t2, t1)

        # test second run is accurate
        np.testing.assert_allclose(y[:, 0], np.exp(0.1 * t_eval),
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

        #model = pybamm.BaseModel()
        #var = pybamm.Variable("var")
        #model.rhs = {var: -pybamm.InputParameter("rate") * var}
        #model.initial_conditions = {var: 1}
        ## No need to set parameters; can use base discretisation (no spatial operators)

        ## create discretisation
        #disc = pybamm.Discretisation()
        #disc.process_model(model)

        #t_eval = np.linspace(0, 10, 4)


        # Solve
        t_eval = np.linspace(0, 10, 4)
        y0 = model.concatenated_initial_conditions.evaluate().reshape(-1)
        rhs = pybamm.EvaluatorJax(model.concatenated_rhs)

        def fun(y, t, inputs):
            return rhs.evaluate(t=t, y=y, inputs=inputs).reshape(-1)

        h = 0.0001
        rate = 0.1

        @jax.jit
        def solve(rate):
            return pybamm.jax_bdf_integrate(fun, y0, t_eval,
                                            {'rate': rate},
                                            rtol=1e-9, atol=1e-9)

        @jax.jit
        def solve_odeint(rate):
            return jax.experimental.ode.odeint(fun, y0, t_eval,
                                            {'rate': rate},
                                            rtol=1e-9, atol=1e-9)


        grad_solve = jax.jit(jax.jacrev(solve))
        grad = grad_solve(rate)
        print(grad)

        eval_plus = solve(rate + h)
        eval_plus2 = solve_odeint(rate + h)
        print(eval_plus.shape)
        print(eval_plus2.shape)
        eval_neg = solve(rate - h)
        grad_num = (eval_plus - eval_neg) / (2 * h)
        print(grad_num)

        grad_solve = jax.jit(jax.jacrev(solve))
        print('finished calculating jacobian',grad_solve)
        print('5')
        time.sleep(1)
        print('4')
        time.sleep(1)
        print('3')
        time.sleep(1)
        print('2')
        time.sleep(1)
        print('1')
        time.sleep(1)
        print('go')

        grad = grad_solve(rate)
        print('finished executing jacobian')
        print(grad)

        print('5')
        time.sleep(1)
        print('4')
        time.sleep(1)
        print('3')
        time.sleep(1)
        print('2')
        time.sleep(1)
        print('1')
        time.sleep(1)
        print('go')

        grad = grad_solve(rate)
        print(grad)
        print('finished executing jacobian')



        #np.testing.assert_allclose(y[0, :].reshape(-1), np.exp(-0.1 * t_eval))

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
