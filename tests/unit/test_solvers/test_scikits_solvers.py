#
# Tests for the Scikits Solver classes
#
import pybamm
import numpy as np
import unittest
import warnings
from tests import get_mesh_for_testing, get_discretisation_for_testing
import sys


@unittest.skipIf(not pybamm.have_scikits_odes(), "scikits.odes not installed")
class TestScikitsSolvers(unittest.TestCase):
    def test_model_ode_integrate_failure(self):
        # Turn off warnings to ignore sqrt error
        warnings.simplefilter("ignore")

        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: -pybamm.sqrt(var)}
        model.initial_conditions = {var: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.ScikitsOdeSolver()
        # Expect solver to fail when y goes negative
        with self.assertRaises(pybamm.SolverError):
            solver.solve(model, t_eval)

        # Turn warnings back on
        warnings.simplefilter("default")

    def test_model_dae_integrate_failure_bad_ics(self):
        # Force model to fail by providing bad ics
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        # Create custom model so that custom ics
        class Model:
            mass_matrix = pybamm.Matrix([[1.0, 0.0], [0.0, 0.0]])
            y0 = np.array([0.0, 1.0])
            terminate_events_eval = []
            convert_to_format = "python"

            def rhs_algebraic_eval(self, t, y, inputs):
                return np.array([0.5 * np.ones_like(y[0]), 2 * y[0] - y[1]])

            def jac_rhs_algebraic_eval(self, t, y, inputs):
                return np.array([[0.0, 0.0], [2.0, -1.0]])

        model = Model()
        t_eval = np.linspace(0, 1, 100)

        with self.assertRaises(pybamm.SolverError):
            solver._integrate(model, t_eval)

    def test_dae_integrate_bad_ics(self):
        # Make sure that dae solver can fix bad ics automatically
        # Constant
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        var2 = pybamm.Variable("var2")
        model.rhs = {var: 0.5}
        model.algebraic = {var2: 2 * var - var2}
        model.initial_conditions = {var: 0, var2: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 1, 100)
        solver.set_up(model)
        solver._set_initial_conditions(model, 0, {}, True)
        # check y0
        np.testing.assert_array_equal(model.y0.full().flatten(), [0, 0])
        # check dae solutions
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])

    def test_dae_integrate_with_non_unity_mass(self):
        # Constant
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

        # Create custom model so that custom mass matrix can be used
        class Model:
            mass_matrix = pybamm.Matrix([[4.0, 0.0], [0.0, 0.0]])
            y0 = np.array([0.0, 0.0])
            terminate_events_eval = []
            convert_to_format = "python"
            len_rhs_and_alg = 2

            def rhs_algebraic_eval(self, t, y, inputs):
                return np.array([0.5 * np.ones_like(y[0]), 2.0 * y[0] - y[1]])

            def jac_rhs_algebraic_eval(self, t, y, inputs):
                return np.array([[0.0, 0.0], [2.0, -1.0]])

        model = Model()
        t_eval = np.linspace(0, 1, 100)
        solution = solver._integrate(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.125 * solution.t, solution.y[0])
        np.testing.assert_allclose(0.25 * solution.t, solution.y[1])

    def test_model_solver_ode_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsOdeSolver(rtol=1e-9, atol=1e-9)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

    def test_model_solver_ode_events_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = [
            pybamm.Event("2 * var = 2.5", pybamm.min(2.5 - 2 * var)),
            pybamm.Event("var = 1.5", pybamm.min(1.5 - var)),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsOdeSolver(rtol=1e-9, atol=1e-9)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_array_less(solution.y[0, :-1], 1.5)
        np.testing.assert_array_less(solution.y[0, :-1], 1.25)
        np.testing.assert_equal(solution.t_event[0], solution.t[-1])
        np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])

    def test_model_solver_ode_jacobian_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: var1, var2: 1 - var1}
        model.initial_conditions = {var1: 1.0, var2: -1.0}
        model.variables = {"var1": var1, "var2": var2}

        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Add user-supplied Jacobian to model
        mesh = get_mesh_for_testing()
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        N = submesh.npts

        # Solve testing various linear solvers
        linsolvers = [
            "dense",
            # "lapackdense",
            "spgmr",
            "spbcgs",
            "sptfqmr",
        ]

        for linsolver in linsolvers:
            solver = pybamm.ScikitsOdeSolver(
                rtol=1e-9, atol=1e-9, extra_options={"linsolver": linsolver}
            )
            t_eval = np.linspace(0, 1, 100)
            solution = solver.solve(model, t_eval)
            np.testing.assert_array_equal(solution.t, t_eval)

            T, Y = solution.t, solution.y
            np.testing.assert_array_almost_equal(
                model.variables["var1"].evaluate(T, Y),
                np.ones((N, T.size)) * np.exp(T[np.newaxis, :]),
            )
            np.testing.assert_array_almost_equal(
                model.variables["var2"].evaluate(T, Y),
                np.ones((N, T.size)) * (T[np.newaxis, :] - np.exp(T[np.newaxis, :])),
            )

    def test_model_solver_dae_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.use_jacobian = False
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_model_solver_dae_jax(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.use_jacobian = False
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_bad_ics_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 3}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_events_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1)),
            pybamm.Event("var2 = 2.5", pybamm.min(2.5 - var2)),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y[0, :-1], 1.5)
        np.testing.assert_array_less(solution.y[-1, :-1], 2.5)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))
        np.testing.assert_equal(solution.t_event[0], solution.t[-1])
        np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])

    def test_model_solver_dae_nonsmooth_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        discontinuity = 0.6

        def nonsmooth_rate(t):
            return 0.1 * (t < discontinuity) + 0.1

        def nonsmooth_mult(t):
            return (t < discontinuity) + 1.0

        rate = nonsmooth_rate(pybamm.t)
        mult = nonsmooth_mult(pybamm.t)
        # put in an extra heaviside with no time dependence, this should be ignored by
        # the solver i.e. no extra discontinuities added
        model.rhs = {var1: rate * var1 + (var1 < 0)}
        model.algebraic = {var2: mult * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1)),
            pybamm.Event("var2 = 2.5", pybamm.min(2.5 - var2)),
            pybamm.Event(
                "nonsmooth rate",
                pybamm.Scalar(discontinuity),
                pybamm.EventType.DISCONTINUITY,
            ),
            pybamm.Event(
                "nonsmooth mult",
                pybamm.Scalar(discontinuity),
                pybamm.EventType.DISCONTINUITY,
            ),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")

        # create two time series, one without a time point on the discontinuity,
        # and one with
        t_eval1 = np.linspace(0, 5, 10)
        t_eval2 = np.insert(
            t_eval1, np.searchsorted(t_eval1, discontinuity), discontinuity
        )
        solution1 = solver.solve(model, t_eval1)
        solution2 = solver.solve(model, t_eval2)

        # check time vectors
        for solution in [solution1, solution2]:
            # time vectors are ordered
            self.assertTrue(np.all(solution.t[:-1] <= solution.t[1:]))

            # time value before and after discontinuity is an epsilon away
            dindex = np.searchsorted(solution.t, discontinuity)
            value_before = solution.t[dindex - 1]
            value_after = solution.t[dindex]
            self.assertEqual(value_before / (1 - sys.float_info.epsilon), discontinuity)
            self.assertEqual(value_after / (1 + sys.float_info.epsilon), discontinuity)

        # both solution time vectors should have same number of points
        self.assertEqual(len(solution1.t), len(solution2.t))

        # check solution
        for solution in [solution1, solution2]:
            np.testing.assert_array_less(solution.y[0, :-1], 1.5)
            np.testing.assert_array_less(solution.y[-1, :-1], 2.5)
            var1_soln = np.exp(0.2 * solution.t)
            y0 = np.exp(0.2 * discontinuity)
            var1_soln[solution.t > discontinuity] = y0 * np.exp(
                0.1 * (solution.t[solution.t > discontinuity] - discontinuity)
            )
            var2_soln = 2 * var1_soln
            var2_soln[solution.t > discontinuity] = var1_soln[
                solution.t > discontinuity
            ]
            np.testing.assert_allclose(solution.y[0], var1_soln, rtol=1e-06)
            np.testing.assert_allclose(solution.y[-1], var2_soln, rtol=1e-06)

    def test_model_solver_dae_multiple_nonsmooth_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        a = 0.6
        discontinuities = (np.arange(3) + 1) * a

        model.rhs = {var1: pybamm.Modulo(pybamm.t, a)}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 0, var2: 0}
        model.events = [
            pybamm.Event("var1 = 0.55", pybamm.min(0.55 - var1)),
            pybamm.Event("var2 = 1.2", pybamm.min(1.2 - var2)),
        ]
        for discontinuity in discontinuities:
            model.events.append(
                pybamm.Event("nonsmooth rate", pybamm.Scalar(discontinuity))
            )
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")

        # create two time series, one without a time point on the discontinuity,
        # and one with
        t_eval1 = np.linspace(0, 2, 10)
        t_eval2 = np.insert(
            t_eval1, np.searchsorted(t_eval1, discontinuities), discontinuities
        )
        solution1 = solver.solve(model, t_eval1)
        solution2 = solver.solve(model, t_eval2)

        # check time vectors
        for solution in [solution1, solution2]:
            # time vectors are ordered
            self.assertTrue(np.all(solution.t[:-1] <= solution.t[1:]))

            # time value before and after discontinuity is an epsilon away
            for discontinuity in discontinuities:
                dindex = np.searchsorted(solution.t, discontinuity)
                value_before = solution.t[dindex - 1]
                value_after = solution.t[dindex]
                self.assertEqual(
                    value_before / (1 - sys.float_info.epsilon), discontinuity
                )
                self.assertEqual(
                    value_after / (1 + sys.float_info.epsilon), discontinuity
                )

        # both solution time vectors should have same number of points
        self.assertEqual(len(solution1.t), len(solution2.t))

        # check solution
        for solution in [solution1, solution2]:
            np.testing.assert_array_less(solution.y[0, :-1], 0.55)
            np.testing.assert_array_less(solution.y[-1, :-1], 1.2)
            var1_soln = (solution.t % a) ** 2 / 2 + a**2 / 2 * (solution.t // a)
            var2_soln = 2 * var1_soln
            np.testing.assert_allclose(solution.y[0], var1_soln, rtol=1e-06)
            np.testing.assert_allclose(solution.y[-1], var2_soln, rtol=1e-06)

    def test_model_solver_dae_no_nonsmooth_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        discontinuity = 5.6

        def nonsmooth_rate(t):
            return 0.1 * int(t < discontinuity) + 0.1

        def nonsmooth_mult(t):
            return int(t < discontinuity) + 1.0

        # put in an extra heaviside with no time dependence, this should be ignored by
        # the solver i.e. no extra discontinuities added
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-9, atol=1e-9, root_method="lm")

        # create two time series, one without a time point on the discontinuity,
        # and one with
        t_eval = np.linspace(0, 5, 10)
        solution = solver.solve(model, t_eval)

        # test solution, discontinuity should not be triggered
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_with_jacobian_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1.0, var2: 2.0}
        model.initial_conditions_ydot = {var1: 0.1, var2: 0.2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Add user-supplied Jacobian to model
        mesh = get_mesh_for_testing()
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        N = submesh.npts

        def jacobian(t, y):
            return np.block(
                [
                    [0.1 * np.eye(N), np.zeros((N, N))],
                    [2.0 * np.eye(N), -1.0 * np.eye(N)],
                ]
            )

        model.jacobian = jacobian
        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_solve_ode_model_with_dae_solver_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

    def test_model_step_ode_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: -0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        solver = pybamm.ScikitsOdeSolver(rtol=1e-9, atol=1e-9)

        # Step once
        dt = 1
        step_sol = solver.step(None, model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(step_sol.y[0], np.exp(-0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(step_sol, model, dt, npts=5)
        np.testing.assert_array_equal(
            step_sol_2.t, np.concatenate([np.array([0]), np.linspace(dt, 2 * dt, 5)])
        )
        np.testing.assert_allclose(step_sol_2.y[0], np.exp(-0.1 * step_sol_2.t))

        # Check steps give same solution as solve
        t_eval = step_sol.t
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], step_sol.y[0])

    def test_model_step_dae_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.use_jacobian = False
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")

        # Step once
        dt = 1
        step_sol = solver.step(None, model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(step_sol.y[0], np.exp(0.1 * step_sol.t))
        np.testing.assert_allclose(step_sol.y[-1], 2 * np.exp(0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(step_sol, model, dt, npts=5)
        np.testing.assert_array_equal(
            step_sol_2.t, np.concatenate([np.array([0]), np.linspace(dt, 2 * dt, 5)])
        )
        np.testing.assert_allclose(step_sol_2.y[0], np.exp(0.1 * step_sol_2.t))
        np.testing.assert_allclose(step_sol_2.y[-1], 2 * np.exp(0.1 * step_sol_2.t))

        # Check steps give same solution as solve
        t_eval = step_sol.t
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], step_sol.y[0, :])
        np.testing.assert_allclose(solution.y[-1], step_sol.y[-1, :])

    def test_model_solver_ode_events_casadi(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "casadi"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = [
            pybamm.Event("2 * var = 2.5", pybamm.min(2.5 - 2 * var)),
            pybamm.Event("var = 1.5", pybamm.min(1.5 - var)),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsOdeSolver(rtol=1e-9, atol=1e-9)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_array_less(solution.y[0:, -1], 1.5)
        np.testing.assert_array_less(solution.y[0:, -1], 1.25 + 1e-6)
        np.testing.assert_equal(solution.t_event[0], solution.t[-1])
        np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])

    def test_model_solver_dae_events_casadi(self):
        # Create model
        model = pybamm.BaseModel()
        for use_jacobian in [True, False]:
            model.use_jacobian = use_jacobian
            model.convert_to_format = "casadi"
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            var1 = pybamm.Variable("var1", domain=whole_cell)
            var2 = pybamm.Variable("var2", domain=whole_cell)
            model.rhs = {var1: 0.1 * var1}
            model.algebraic = {var2: 2 * var1 - var2}
            model.initial_conditions = {var1: 1, var2: 2}
            model.events = [
                pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1)),
                pybamm.Event("var2 = 2.5", pybamm.min(2.5 - var2)),
            ]
            disc = get_discretisation_for_testing()
            model_disc = disc.process_model(model, inplace=False)

            # Solve
            solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
            t_eval = np.linspace(0, 5, 100)
            solution = solver.solve(model_disc, t_eval)
            np.testing.assert_array_less(solution.y[0, :-1], 1.5)
            np.testing.assert_array_less(solution.y[-1, :-1], 2.5)
            np.testing.assert_equal(solution.t_event[0], solution.t[-1])
            np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])
            np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
            np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_inputs_events(self):
        # Create model
        for form in ["python", "casadi"]:
            model = pybamm.BaseModel()
            model.convert_to_format = form
            whole_cell = ["negative electrode", "separator", "positive electrode"]
            var1 = pybamm.Variable("var1", domain=whole_cell)
            var2 = pybamm.Variable("var2", domain=whole_cell)
            model.rhs = {var1: pybamm.InputParameter("rate 1") * var1}
            model.algebraic = {var2: pybamm.InputParameter("rate 2") * var1 - var2}
            model.initial_conditions = {var1: 1, var2: 2}
            model.events = [
                pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1)),
                pybamm.Event("var2 = 2.5", pybamm.min(2.5 - var2)),
            ]
            disc = get_discretisation_for_testing()
            disc.process_model(model)

            # Solve
            if form == "python":
                solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8, root_method="lm")
            else:
                solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
            t_eval = np.linspace(0, 5, 100)
            solution = solver.solve(model, t_eval, inputs={"rate 1": 0.1, "rate 2": 2})
            np.testing.assert_array_less(solution.y[0, :-1], 1.5)
            np.testing.assert_array_less(solution.y[-1, :-1], 2.5)
            np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])
            np.testing.assert_equal(solution.t_event[0], solution.t[-1])

            np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
            np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_inputs_in_initial_conditions(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: pybamm.InputParameter("rate") * var1}
        model.algebraic = {var2: var1 - var2}
        model.initial_conditions = {
            var1: pybamm.InputParameter("ic 1"),
            var2: pybamm.InputParameter("ic 2"),
        }

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(
            model, t_eval, inputs={"rate": -1, "ic 1": 0.1, "ic 2": 2}
        )
        np.testing.assert_array_almost_equal(
            solution.y[0], 0.1 * np.exp(-solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            solution.y[-1], 0.1 * np.exp(-solution.t), decimal=5
        )

        # Solve again with different initial conditions
        solution = solver.solve(
            model, t_eval, inputs={"rate": -0.1, "ic 1": 1, "ic 2": 3}
        )
        np.testing.assert_array_almost_equal(
            solution.y[0], 1 * np.exp(-0.1 * solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            solution.y[-1], 1 * np.exp(-0.1 * solution.t), decimal=5
        )

    def test_solve_ode_model_with_dae_solver_casadi(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "casadi"
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

    def test_model_step_events(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1)),
            pybamm.Event("var2 = 2.5", pybamm.min(2.5 - var2)),
        ]
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Solve
        step_solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        dt = 0.05
        time = 0
        end_time = 5
        step_solution = None
        while time < end_time:
            step_solution = step_solver.step(step_solution, model, dt=dt, npts=10)
            time += dt
        np.testing.assert_array_less(step_solution.y[0, :-1], 1.5)
        np.testing.assert_array_less(step_solution.y[-1, :-1], 2.5)
        np.testing.assert_equal(step_solution.t_event[0], step_solution.t[-1])
        np.testing.assert_array_equal(
            step_solution.y_event[:, 0], step_solution.y[:, -1]
        )
        np.testing.assert_array_almost_equal(
            step_solution.y[0], np.exp(0.1 * step_solution.t), decimal=5
        )
        np.testing.assert_array_almost_equal(
            step_solution.y[-1], 2 * np.exp(0.1 * step_solution.t), decimal=5
        )

    def test_model_step_nonsmooth_events(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")

        a = 0.6
        discontinuities = (np.arange(3) + 1) * a

        model.rhs = {var1: pybamm.Modulo(pybamm.t, a)}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 0, var2: 0}
        model.events = [
            pybamm.Event("var1 = 0.55", pybamm.min(0.55 - var1)),
            pybamm.Event("var2 = 1.2", pybamm.min(1.2 - var2)),
        ]
        for discontinuity in discontinuities:
            model.events.append(
                pybamm.Event("nonsmooth rate", pybamm.Scalar(discontinuity))
            )
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        step_solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)
        dt = 0.05
        time = 0
        end_time = 3
        step_solution = None
        while time < end_time:
            step_solution = step_solver.step(step_solution, model, dt=dt, npts=10)
            time += dt
        np.testing.assert_array_less(step_solution.y[0, :-1], 0.55)
        np.testing.assert_array_less(step_solution.y[-1, :-1], 1.2)
        np.testing.assert_equal(step_solution.t_event[0], step_solution.t[-1])
        np.testing.assert_array_equal(
            step_solution.y_event[:, 0], step_solution.y[:, -1]
        )
        var1_soln = (step_solution.t % a) ** 2 / 2 + a**2 / 2 * (step_solution.t // a)
        var2_soln = 2 * var1_soln
        np.testing.assert_array_almost_equal(step_solution.y[0], var1_soln, decimal=5)
        np.testing.assert_array_almost_equal(step_solution.y[-1], var2_soln, decimal=5)

    def test_model_solver_dae_nonsmooth(self):
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2")
        discontinuity = 0.6

        # Create three different models with the same solution, each expressing the
        # discontinuity in a different way

        # first model explicitly adds a discontinuity event
        def nonsmooth_rate(t):
            return 0.1 * (t < discontinuity) + 0.1

        rate = pybamm.Function(nonsmooth_rate, pybamm.t)
        model1 = pybamm.BaseModel()
        model1.rhs = {var1: rate * var1}
        model1.algebraic = {var2: var2}
        model1.initial_conditions = {var1: 1, var2: 0}
        model1.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1)),
            pybamm.Event(
                "nonsmooth rate",
                pybamm.Scalar(discontinuity),
                pybamm.EventType.DISCONTINUITY,
            ),
        ]

        # second model implicitly adds a discontinuity event via a heaviside function
        model2 = pybamm.BaseModel()
        model2.rhs = {var1: (0.1 * (pybamm.t < discontinuity) + 0.1) * var1}
        model2.algebraic = {var2: var2}
        model2.initial_conditions = {var1: 1, var2: 0}
        model2.events = [pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1))]

        # third model implicitly adds a discontinuity event via another heaviside
        # function
        model3 = pybamm.BaseModel()
        model3.rhs = {var1: (-0.1 * (discontinuity < pybamm.t) + 0.2) * var1}
        model3.algebraic = {var2: var2}
        model3.initial_conditions = {var1: 1, var2: 0}
        model3.events = [pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1))]

        for model in [model1, model2, model3]:
            disc = get_discretisation_for_testing()
            disc.process_model(model)

            # Solve
            solver = pybamm.ScikitsDaeSolver(rtol=1e-8, atol=1e-8)

            # create two time series, one without a time point on the discontinuity,
            # and one with
            t_eval1 = np.linspace(0, 5, 10)
            t_eval2 = np.insert(
                t_eval1, np.searchsorted(t_eval1, discontinuity), discontinuity
            )
            solution1 = solver.solve(model, t_eval1)
            solution2 = solver.solve(model, t_eval2)

            # check time vectors
            for solution in [solution1, solution2]:
                # time vectors are ordered
                self.assertTrue(np.all(solution.t[:-1] <= solution.t[1:]))

                # time value before and after discontinuity is an epsilon away
                dindex = np.searchsorted(solution.t, discontinuity)
                value_before = solution.t[dindex - 1]
                value_after = solution.t[dindex]
                self.assertEqual(
                    value_before / (1 - sys.float_info.epsilon), discontinuity
                )
                self.assertEqual(
                    value_after / (1 + sys.float_info.epsilon), discontinuity
                )

            # both solution time vectors should have same number of points
            self.assertEqual(len(solution1.t), len(solution2.t))

            # check solution
            for solution in [solution1, solution2]:
                np.testing.assert_array_less(solution.y[0, :-1], 1.5)
                np.testing.assert_array_less(solution.y[-1, :-1], 2.5)
                np.testing.assert_equal(solution.t_event[0], solution.t[-1])
                np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])
                var1_soln = np.exp(0.2 * solution.t)
                y0 = np.exp(0.2 * discontinuity)
                var1_soln[solution.t > discontinuity] = y0 * np.exp(
                    0.1 * (solution.t[solution.t > discontinuity] - discontinuity)
                )
                np.testing.assert_allclose(solution.y[0], var1_soln, rtol=1e-06)

    def test_ode_solver_fail_with_dae(self):
        model = pybamm.BaseModel()
        a = pybamm.Scalar(1)
        model.algebraic = {a: a}
        model.concatenated_initial_conditions = a
        solver = pybamm.ScikitsOdeSolver()
        with self.assertRaisesRegex(pybamm.SolverError, "Cannot use ODE solver"):
            solver.set_up(model)

    def test_dae_solver_algebraic_model(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var + 1}
        model.initial_conditions = {var: 0}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.ScikitsDaeSolver()
        t_eval = np.linspace(0, 1)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.y, -1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
