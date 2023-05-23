#
# Tests for the Base Solver class
#
from tests import TestCase
import casadi
import pybamm
import numpy as np
from scipy.sparse import csr_matrix

import unittest


class TestBaseSolver(TestCase):
    def test_base_solver_init(self):
        solver = pybamm.BaseSolver(rtol=1e-2, atol=1e-4)
        self.assertEqual(solver.rtol, 1e-2)
        self.assertEqual(solver.atol, 1e-4)

        solver.rtol = 1e-5
        self.assertEqual(solver.rtol, 1e-5)
        solver.rtol = 1e-7
        self.assertEqual(solver.rtol, 1e-7)

    def test_root_method_init(self):
        solver = pybamm.BaseSolver(root_method="casadi")
        self.assertIsInstance(solver.root_method, pybamm.CasadiAlgebraicSolver)

        solver = pybamm.BaseSolver(root_method="lm")
        self.assertIsInstance(solver.root_method, pybamm.AlgebraicSolver)
        self.assertEqual(solver.root_method.method, "lm")

        root_solver = pybamm.AlgebraicSolver()
        solver = pybamm.BaseSolver(root_method=root_solver)
        self.assertEqual(solver.root_method, root_solver)

        with self.assertRaisesRegex(
            pybamm.SolverError, "Root method must be an algebraic solver"
        ):
            pybamm.BaseSolver(root_method=pybamm.ScipySolver())

    def test_step_or_solve_empty_model(self):
        model = pybamm.BaseModel()
        solver = pybamm.BaseSolver()
        with self.assertRaisesRegex(pybamm.ModelError, "Cannot step empty model"):
            solver.step(None, model, None)
        with self.assertRaisesRegex(pybamm.ModelError, "Cannot solve empty model"):
            solver.solve(model, None)

    def test_t_eval_none(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: 1}
        model.initial_conditions = {v: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.BaseSolver()
        with self.assertRaisesRegex(ValueError, "t_eval cannot be None"):
            solver.solve(model, None)

    def test_nonmonotonic_teval(self):
        solver = pybamm.BaseSolver(rtol=1e-2, atol=1e-4)
        model = pybamm.BaseModel()
        a = pybamm.Scalar(0)
        model.rhs = {a: a}
        with self.assertRaisesRegex(
            pybamm.SolverError, "t_eval must increase monotonically"
        ):
            solver.solve(model, np.array([1, 2, 3, 2]))

        # Check stepping with step size too small
        dt = 1e-9
        with self.assertRaisesRegex(
            pybamm.SolverError, "Step time must be at least 1.000 ns"
        ):
            solver.step(None, model, dt)

    def test_solution_time_length_fail(self):
        model = pybamm.BaseModel()
        v = pybamm.Scalar(1)
        model.variables = {"v": v}
        solver = pybamm.DummySolver()
        t_eval = np.array([0])
        with self.assertRaisesRegex(
            pybamm.SolverError, "Solution time vector has length 1"
        ):
            solver.solve(model, t_eval)

    def test_block_symbolic_inputs(self):
        solver = pybamm.BaseSolver(rtol=1e-2, atol=1e-4)
        model = pybamm.BaseModel()
        a = pybamm.Variable("a")
        p = pybamm.InputParameter("p")
        model.rhs = {a: a * p}
        with self.assertRaisesRegex(
            pybamm.SolverError, "No value provided for input 'p'"
        ):
            solver.solve(model, np.array([1, 2, 3]))

    def test_ode_solver_fail_with_dae(self):
        model = pybamm.BaseModel()
        a = pybamm.Scalar(1)
        model.algebraic = {a: a}
        model.concatenated_initial_conditions = pybamm.Scalar(0)
        solver = pybamm.ScipySolver()
        with self.assertRaisesRegex(pybamm.SolverError, "Cannot use ODE solver"):
            solver.set_up(model)

    def test_find_consistent_initial_conditions(self):
        # Simple system: a single algebraic equation
        class ScalarModel:
            def __init__(self):
                self.y0 = np.array([2])
                self.rhs = {}
                self.jac_algebraic_eval = None
                t = casadi.MX.sym("t")
                y = casadi.MX.sym("y")
                p = casadi.MX.sym("p")
                self.casadi_algebraic = casadi.Function(
                    "alg", [t, y, p], [self.algebraic_eval(t, y, p)]
                )
                self.convert_to_format = "casadi"
                self.bounds = (np.array([-np.inf]), np.array([np.inf]))
                self.len_rhs_and_alg = 1
                self.events = []

            def rhs_eval(self, t, y, inputs):
                return np.array([])

            def algebraic_eval(self, t, y, inputs):
                return y + 2

        solver = pybamm.BaseSolver(root_method="lm")
        model = ScalarModel()
        init_cond = solver.calculate_consistent_state(model)
        np.testing.assert_array_equal(init_cond, -2)
        # with casadi
        solver_with_casadi = pybamm.BaseSolver(root_method="casadi", root_tol=1e-12)
        model = ScalarModel()
        init_cond = solver_with_casadi.calculate_consistent_state(model)
        np.testing.assert_array_equal(init_cond, -2)

        # More complicated system
        vec = np.array([0.0, 1.0, 1.5, 2.0])

        class VectorModel:
            def __init__(self):
                self.y0 = np.zeros_like(vec)
                self.rhs = {"test": "test"}
                self.concatenated_rhs = np.array([1])
                self.jac_algebraic_eval = None
                t = casadi.MX.sym("t")
                y = casadi.MX.sym("y", vec.size)
                p = casadi.MX.sym("p")
                self.casadi_algebraic = casadi.Function(
                    "alg", [t, y, p], [self.algebraic_eval(t, y, p)]
                )
                self.convert_to_format = "casadi"
                self.bounds = (-np.inf * np.ones(4), np.inf * np.ones(4))
                self.len_rhs = 1
                self.len_rhs_and_alg = 4
                self.events = []

            def rhs_eval(self, t, y, inputs):
                return y[0:1]

            def algebraic_eval(self, t, y, inputs):
                return (y[1:] - vec[1:]) ** 2

        model = VectorModel()
        init_cond = solver.calculate_consistent_state(model)
        np.testing.assert_array_almost_equal(init_cond.flatten(), vec)
        # with casadi
        init_cond = solver_with_casadi.calculate_consistent_state(model)
        np.testing.assert_array_almost_equal(init_cond.full().flatten(), vec)

        # With Jacobian
        def jac_dense(t, y, inputs):
            return 2 * np.hstack([np.zeros((3, 1)), np.diag(y[1:] - vec[1:])])

        model.jac_algebraic_eval = jac_dense
        init_cond = solver.calculate_consistent_state(model)
        np.testing.assert_array_almost_equal(init_cond.flatten(), vec)

        # With sparse Jacobian
        def jac_sparse(t, y, inputs):
            return 2 * csr_matrix(
                np.hstack([np.zeros((3, 1)), np.diag(y[1:] - vec[1:])])
            )

        model.jac_algebraic_eval = jac_sparse
        init_cond = solver.calculate_consistent_state(model)
        np.testing.assert_array_almost_equal(init_cond.flatten(), vec)

    def test_fail_consistent_initial_conditions(self):
        class Model:
            def __init__(self):
                self.y0 = np.array([2])
                self.rhs = {}
                self.jac_algebraic_eval = None
                t = casadi.MX.sym("t")
                y = casadi.MX.sym("y")
                p = casadi.MX.sym("p")
                self.casadi_algebraic = casadi.Function(
                    "alg", [t, y, p], [self.algebraic_eval(t, y, p)]
                )
                self.convert_to_format = "casadi"
                self.bounds = (np.array([-np.inf]), np.array([np.inf]))

            def rhs_eval(self, t, y, inputs):
                return np.array([])

            def algebraic_eval(self, t, y, inputs):
                # algebraic equation has no root
                return y**2 + 1

        solver = pybamm.BaseSolver(root_method="hybr")

        with self.assertRaisesRegex(
            pybamm.SolverError,
            "Could not find acceptable solution: The iteration is not making",
        ):
            solver.calculate_consistent_state(Model())
        solver = pybamm.BaseSolver(root_method="lm")
        with self.assertRaisesRegex(
            pybamm.SolverError, "Could not find acceptable solution: solver terminated"
        ):
            solver.calculate_consistent_state(Model())
        # with casadi
        solver = pybamm.BaseSolver(root_method="casadi")
        with self.assertRaisesRegex(
            pybamm.SolverError, "Could not find acceptable solution: Error in Function"
        ):
            solver.calculate_consistent_state(Model())

    def test_discretise_model(self):
        # Make sure 0D model is automatically discretised
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}

        solver = pybamm.BaseSolver()
        self.assertFalse(model.is_discretised)
        solver.set_up(model, {})
        self.assertTrue(model.is_discretised)

        # 1D model cannot be automatically discretised
        model = pybamm.BaseModel()
        v = pybamm.Variable("v", domain="line")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}

        with self.assertRaisesRegex(
            pybamm.DiscretisationError, "Cannot automatically discretise model"
        ):
            solver.set_up(model, {})

    def test_convert_to_casadi_format(self):
        # Make sure model is converted to casadi format
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}
        model.convert_to_format = "python"

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.BaseSolver(root_method="casadi")
        pybamm.set_logging_level("ERROR")
        solver.set_up(model, {})
        self.assertEqual(model.convert_to_format, "casadi")
        pybamm.set_logging_level("WARNING")

    def test_inputs_step(self):
        # Make sure interpolant inputs are dropped
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}
        x = np.array([0, 1])
        interp = pybamm.Interpolant(x, x, pybamm.t)
        solver = pybamm.CasadiSolver()
        for input_key in ["Current input [A]", "Voltage input [V]", "Power input [W]"]:
            sol = solver.step(
                old_solution=None, model=model, dt=1.0, inputs={input_key: interp}
            )
            self.assertFalse(input_key in sol.all_inputs[0])

    def test_extrapolation_warnings(self):
        # Make sure the extrapolation warnings work
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}
        model.events.append(
            pybamm.Event(
                "Triggered event",
                v - 0.5,
                pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
            )
        )
        model.events.append(
            pybamm.Event(
                "Ignored event",
                v + 10,
                pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
            )
        )
        solver = pybamm.ScipySolver()
        solver.set_up(model)

        with self.assertWarns(pybamm.SolverWarning):
            solver.step(old_solution=None, model=model, dt=1.0)

        with self.assertWarns(pybamm.SolverWarning):
            solver.solve(model, t_eval=[0, 1])

    def test_multiple_models_error(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}
        model2 = pybamm.BaseModel()
        v2 = pybamm.Variable("v")
        model2.rhs = {v2: -1}
        model2.initial_conditions = {v2: 1}

        solver = pybamm.ScipySolver()
        solver.solve(model, t_eval=[0, 1])
        with self.assertRaisesRegex(RuntimeError, "already been initialised"):
            solver.solve(model2, t_eval=[0, 1])

    @unittest.skipIf(not pybamm.have_idaklu(), "idaklu solver is not installed")
    def test_sensitivities(self):
        def exact_diff_a(y, a, b):
            return np.array([[y[0] ** 2 + 2 * a], [y[0]]])

        @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
        def exact_diff_b(y, a, b):
            return np.array([[y[0]], [0]])

        for convert_to_format in ["", "python", "casadi", "jax"]:
            model = pybamm.BaseModel()
            v = pybamm.Variable("v")
            u = pybamm.Variable("u")
            a = pybamm.InputParameter("a")
            b = pybamm.InputParameter("b")
            model.rhs = {v: a * v**2 + b * v + a**2}
            model.algebraic = {u: a * v - u}
            model.initial_conditions = {v: 1, u: a * 1}
            model.convert_to_format = convert_to_format
            solver = pybamm.IDAKLUSolver(root_method="lm")
            model.calculate_sensitivities = ["a", "b"]
            solver.set_up(model, inputs={"a": 0, "b": 0})
            all_inputs = []
            for v_value in [0.1, -0.2, 1.5, 8.4]:
                for u_value in [0.13, -0.23, 1.3, 13.4]:
                    for a_value in [0.12, 1.5]:
                        for b_value in [0.82, 1.9]:
                            y = np.array([v_value, u_value])
                            t = 0
                            inputs = {"a": a_value, "b": b_value}
                            all_inputs.append((t, y, inputs))
            for t, y, inputs in all_inputs:
                if model.convert_to_format == "casadi":
                    use_inputs = casadi.vertcat(*[x for x in inputs.values()])
                else:
                    use_inputs = inputs

                sens = model.jacp_rhs_algebraic_eval(t, y, use_inputs)

                if convert_to_format == "casadi":
                    sens_a = sens[0]
                    sens_b = sens[1]
                else:
                    sens_a = sens["a"]
                    sens_b = sens["b"]

                np.testing.assert_allclose(
                    sens_a, exact_diff_a(y, inputs["a"], inputs["b"])
                )
                np.testing.assert_allclose(
                    sens_b, exact_diff_b(y, inputs["a"], inputs["b"])
                )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
