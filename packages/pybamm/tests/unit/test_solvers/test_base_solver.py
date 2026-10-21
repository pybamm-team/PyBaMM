#
# Tests for the Base Solver class
#

import logging
import multiprocessing as mp
import re

import casadi
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import pybamm
from pybamm.models.base_model import ModelSolutionObservability


class TestBaseSolver:
    def test_base_solver_init(self):
        solver = pybamm.BaseSolver(rtol=1e-2, atol=1e-4)
        assert solver.rtol == 1e-2
        assert solver.atol == 1e-4

        solver.rtol = 1e-5
        assert solver.rtol == 1e-5
        solver.rtol = 1e-7
        assert solver.rtol == 1e-7

    def test_root_method_takes_the_tightest_entry_of_a_per_state_atol(self):
        # The root solver takes a single tolerance, so a per-state atol has to
        # reduce rather than reach the comparison against root_tol as an array.
        solver = pybamm.BaseSolver(
            atol=np.array([1e-6, 1e-9]), root_method="nonlinear_solver", root_tol=1e-3
        )
        assert solver.root_method.atol == 1e-9

    def test_root_method_init(self):
        solver = pybamm.BaseSolver(root_method="nonlinear_solver")
        assert isinstance(solver.root_method, pybamm.NonlinearSolver)

        with pytest.warns(
            DeprecationWarning, match="root_method='casadi' is deprecated"
        ):
            solver = pybamm.BaseSolver(root_method="casadi")
        assert isinstance(solver.root_method, pybamm.CasadiAlgebraicSolver)

        solver = pybamm.BaseSolver(root_method="lm")
        assert isinstance(solver.root_method, pybamm.AlgebraicSolver)
        assert solver.root_method.method == "lm"

        root_solver = pybamm.AlgebraicSolver()
        solver = pybamm.BaseSolver(root_method=root_solver)
        assert solver.root_method == root_solver

        with pytest.raises(
            pybamm.SolverError, match=r"Root method must be an algebraic solver"
        ):
            pybamm.BaseSolver(root_method=pybamm.ScipySolver())

    def test_additional_inputs_provided(self):
        # if additional inputs are provided that are not in the model, this should run as normal
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sol1 = sim.solve([0, 3600])["Voltage [V]"].entries
        sol2 = sim.solve([0, 3600], inputs={"Current function [A]": 1})[
            "Voltage [V]"
        ].entries
        sol3 = sim.solve(
            [0, 3600], inputs=[{"Current function [A]": 1}, {"Current function [A]": 2}]
        )[0]["Voltage [V]"].entries
        # check that the solutions are the same
        np.testing.assert_allclose(sol1, sol2)
        np.testing.assert_allclose(sol2, sol3)

    def test_step_or_solve_empty_model(self):
        model = pybamm.BaseModel()
        solver = pybamm.BaseSolver()
        error = "Cannot simulate an empty model"
        with pytest.raises(pybamm.ModelError, match=error):
            solver.step(None, model, None)
        with pytest.raises(pybamm.ModelError, match=error):
            solver.solve(model, None)

    def test_t_eval_none(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: 1}
        model.initial_conditions = {v: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.BaseSolver()
        with pytest.raises(ValueError, match=r"t_eval cannot be None"):
            solver.solve(model, None)

    def test_nonmonotonic_teval(self):
        solver = pybamm.BaseSolver(rtol=1e-2, atol=1e-4)
        model = pybamm.BaseModel()
        a = pybamm.Scalar(0)
        model.rhs = {a: a}
        with pytest.raises(
            pybamm.SolverError, match=r"t_eval must increase monotonically"
        ):
            solver.solve(model, np.array([1, 2, 3, 2]))

        # Check stepping with step size too small
        dt = -1e-9
        with pytest.raises(pybamm.SolverError, match=r"Step time must be >0"):
            solver.step(None, model, dt)

        # Checking if array t_eval lies within range
        dt = 2
        t_eval = np.array([0, 1])
        with pytest.raises(
            pybamm.SolverError,
            match=re.escape(
                "The final `t_eval` value (1) must be equal to the step time `dt` (2)"
            ),
        ):
            solver.step(None, model, dt, t_eval=t_eval)

        t_eval = np.array([1, dt])
        with pytest.raises(
            pybamm.SolverError,
            match=re.escape("The first `t_eval` value (1) must be 0"),
        ):
            solver.step(None, model, dt, t_eval=t_eval)

    def test_solution_time_length_fail(self):
        model = pybamm.BaseModel()
        v = pybamm.Scalar(1)
        model.variables = {"v": v}
        solver = pybamm.DummySolver()
        t_eval = np.array([0])
        with pytest.raises(
            pybamm.SolverError, match=r"Solution time vector has length 1"
        ):
            solver.solve(model, t_eval)

    def test_block_symbolic_inputs(self):
        solver = pybamm.BaseSolver(rtol=1e-2, atol=1e-4)
        model = pybamm.BaseModel()
        a = pybamm.Variable("a")
        p = pybamm.InputParameter("p")
        model.rhs = {a: a * p}
        with pytest.raises(
            pybamm.SolverError, match=re.escape("No value provided for input: ['p']")
        ):
            solver.solve(model, np.array([1, 2, 3]))

    def test_ode_solver_fail_with_dae(self):
        model = pybamm.BaseModel()
        a = pybamm.Scalar(1)
        model.algebraic = {a: a}
        model.concatenated_initial_conditions = pybamm.Scalar(0)
        solver = pybamm.ScipySolver()
        with pytest.raises(pybamm.SolverError, match=r"Cannot use ODE solver"):
            solver.set_up(model)

    def test_find_consistent_initialization(self):
        # Simple system: a single algebraic equation
        class ScalarModel:
            def __init__(self):
                self.y0_list = [np.array([2])]
                self.y0S_list = None
                self.rhs = {}
                self.jac_algebraic_eval = None
                t = casadi.MX.sym("t")
                y = casadi.MX.sym("y")
                p = casadi.MX.sym("p")
                self.casadi_algebraic = casadi.Function(
                    "alg", [t, y, p], [self.algebraic_eval(t, y, p)]
                )
                self.convert_to_format = "casadi"
                self.uses_stacked_inputs = True
                self.bounds = (np.array([-np.inf]), np.array([np.inf]))
                self.len_rhs_and_alg = 1
                self.events = []
                self.solution_observable = ModelSolutionObservability.DISABLED
                self.algebraic_root_solver = None

            def rhs_eval(self, t, y, inputs):
                return np.array([])

            def algebraic_eval(self, t, y, inputs):
                return y + 2

        solver = pybamm.BaseSolver(root_method="lm")
        model = ScalarModel()
        init_states = solver.calculate_consistent_state(model)
        np.testing.assert_array_equal(init_states, -2)
        # with casadi
        solver_with_casadi = pybamm.BaseSolver(root_method="casadi", root_tol=1e-12)
        model = ScalarModel()
        init_states = solver_with_casadi.calculate_consistent_state(model)
        np.testing.assert_array_equal(init_states, -2)

        # More complicated system
        vec = np.array([0.0, 1.0, 1.5, 2.0])

        class VectorModel:
            def __init__(self):
                self.y0_list = [np.zeros_like(vec)]
                self.y0S_list = None
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
                self.uses_stacked_inputs = True
                self.bounds = (-np.inf * np.ones(4), np.inf * np.ones(4))
                self.len_rhs = 1
                self.len_rhs_and_alg = 4
                self.events = []
                self.solution_observable = ModelSolutionObservability.DISABLED
                self.algebraic_root_solver = None

            def rhs_eval(self, t, y, inputs):
                return y[0:1]

            def algebraic_eval(self, t, y, inputs):
                return (y[1:] - vec[1:]) ** 2

        model = VectorModel()
        [init_states] = solver.calculate_consistent_state(model)
        np.testing.assert_allclose(init_states.flatten(), vec, rtol=1e-7, atol=1e-6)
        # with casadi
        solver_with_casadi.root_method.step_tol = 1e-12
        [init_states] = solver_with_casadi.calculate_consistent_state(model)
        np.testing.assert_allclose(
            init_states.full().flatten(), vec, rtol=1e-7, atol=1e-6
        )

        # With Jacobian
        def jac_dense(t, y, inputs):
            return 2 * np.hstack([np.zeros((3, 1)), np.diag(y[1:] - vec[1:])])

        model.jac_algebraic_eval = jac_dense
        [init_states] = solver.calculate_consistent_state(model)
        np.testing.assert_allclose(init_states.flatten(), vec, rtol=1e-7, atol=1e-6)

        # With sparse Jacobian
        def jac_sparse(t, y, inputs):
            return 2 * csr_matrix(
                np.hstack([np.zeros((3, 1)), np.diag(y[1:] - vec[1:])])
            )

        model.jac_algebraic_eval = jac_sparse
        [init_states] = solver.calculate_consistent_state(model)
        np.testing.assert_allclose(init_states.flatten(), vec, rtol=1e-7, atol=1e-6)

    def test_fail_consistent_initialization(self):
        class Model(pybamm.BaseModel):
            def __init__(self):
                self.y0_list = [np.array([2])]
                self.y0S_list = None
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
                self.algebraic_root_solver = None

            def rhs_eval(self, t, y, inputs):
                return np.array([])

            def algebraic_eval(self, t, y, inputs):
                # algebraic equation has no root
                return y**2 + 1

        solver = pybamm.BaseSolver(root_method="hybr")

        with pytest.raises(
            pybamm.SolverError,
            match=r"Could not find acceptable solution",
        ):
            solver.calculate_consistent_state(Model())
        solver = pybamm.BaseSolver(root_method="lm")
        with pytest.raises(
            pybamm.SolverError,
            match=r"Could not find acceptable solution",
        ):
            solver.calculate_consistent_state(Model())
        # with casadi
        solver = pybamm.BaseSolver(root_method="casadi")
        with pytest.raises(
            pybamm.SolverError,
            match=r"Could not find acceptable solution",
        ):
            solver.calculate_consistent_state(Model())

    def test_discretise_model(self):
        # Make sure 0D model is automatically discretised
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}

        solver = pybamm.BaseSolver()
        assert not model.is_discretised
        solver.set_up(model, {})
        assert model.is_discretised

        # 1D model cannot be automatically discretised
        model = pybamm.BaseModel()
        v = pybamm.Variable("v", domain="line")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}

        with pytest.raises(
            pybamm.DiscretisationError, match=r"Cannot automatically discretise model"
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
        assert model.convert_to_format == "casadi"
        pybamm.set_logging_level("WARNING")

    def test_inputs_step(self):
        # Make sure interpolant inputs are dropped
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}
        x = np.array([0, 1])
        interp = pybamm.Interpolant(x, x, pybamm.t)
        solver = pybamm.IDAKLUSolver()
        for input_key in ["Current input [A]", "Voltage input [V]", "Power input [W]"]:
            sol = solver.step(
                old_solution=None, model=model, dt=1.0, inputs={input_key: interp}
            )
            assert input_key not in sol.all_inputs[0]

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

        with pytest.warns(pybamm.SolverWarning):
            solver.step(old_solution=None, model=model, dt=1.0)

        with pytest.warns(pybamm.SolverWarning):
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
        with pytest.raises(RuntimeError, match=r"already been initialised"):
            solver.solve(model2, t_eval=[0, 1])

    def test_multiprocess_context(self):
        solver = pybamm.BaseSolver()
        assert solver.get_platform_context("Win") == "spawn"
        assert solver.get_platform_context("Darwin") == "spawn"

        if pybamm.has_jax():
            # JAX is incompatible with fork (see https://github.com/jax-ml/jax/issues/1805)
            # Causes issues with tests crashing in CI
            assert solver.get_platform_context("Linux") == "spawn"
        else:
            assert solver.get_platform_context("Linux") == "fork"

    def test_sensitivities(self):
        def exact_diff_a(y, a, b):
            return np.array([[y[0] ** 2 + 2 * a], [y[0]]])

        @pytest.mark.skipif(
            not pybamm.has_jax(), reason="jax or jaxlib is not installed"
        )
        def exact_diff_b(y, a, b):
            return np.array([[y[0]], [0]])

        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        u = pybamm.Variable("u")
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        model.rhs = {v: a * v**2 + b * v + a**2}
        model.algebraic = {u: a * v - u}
        model.initial_conditions = {v: 1, u: a * 1}
        model.convert_to_format = "casadi"
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
            use_inputs = casadi.vertcat(*list(inputs.values()))

            sens = model.jacp_rhs_algebraic_eval(t, y, use_inputs)

            sens_a = sens[0]
            sens_b = sens[1]

            np.testing.assert_allclose(
                sens_a, exact_diff_a(y, inputs["a"], inputs["b"])
            )
            np.testing.assert_allclose(
                sens_b, exact_diff_b(y, inputs["a"], inputs["b"])
            )

    @staticmethod
    def _decay_model():
        # dy/dt = -k y, y(0) = y0, so dy/dk = -t y and dy/dy0 = exp(-k t)
        model = pybamm.BaseModel()
        y = pybamm.Variable("y")
        model.rhs = {y: -pybamm.InputParameter("k") * y}
        model.initial_conditions = {y: pybamm.InputParameter("y0")}
        model.variables = {"y": y}
        pybamm.Discretisation().process_model(model)
        return model

    @pytest.mark.parametrize("names", [["k", "zzz"], ["unused"]])
    def test_calculate_sensitivities_names_must_be_model_inputs(self, names):
        inputs = {"k": 0.3, "y0": 2.0, "unused": 1.0}
        bad = [name for name in names if name != "k"]
        match = re.escape(
            f"Cannot calculate sensitivities with respect to {bad}: the input "
            "parameters of model 'Unnamed model' are ['k', 'y0']."
        )
        with pytest.raises(pybamm.SolverError, match=match):
            pybamm.IDAKLUSolver().solve(
                self._decay_model(),
                [0, 1],
                inputs=inputs,
                calculate_sensitivities=names,
            )
        with pytest.raises(pybamm.SolverError, match=match):
            pybamm.IDAKLUSolver().step(
                None,
                self._decay_model(),
                1,
                inputs=inputs,
                calculate_sensitivities=names,
            )

    def test_calculate_sensitivities_true_logs_unused_inputs(self, caplog):
        model = self._decay_model()
        solver = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12)
        inputs = {"k": 0.3, "y0": 2.0, "unused": 1.0}
        with caplog.at_level(logging.WARNING, logger="pybamm.logger"):
            solution = solver.solve(
                model, [0, 1], inputs=inputs, calculate_sensitivities=True
            )
        assert (
            "Inputs ['unused'] are not input parameters of model 'Unnamed model', "
            "so no sensitivities are calculated for them."
        ) in caplog.text
        assert solution.sensitivity_names == ["k", "y0"]
        sensitivities = solution["y"].sensitivities
        assert "unused" not in sensitivities
        t = solution.t
        np.testing.assert_allclose(
            sensitivities["k"].ravel(), -t * 2.0 * np.exp(-0.3 * t), rtol=1e-6
        )

        # Logged once per set-up, and not without sensitivities
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="pybamm.logger"):
            solver.solve(model, [0, 1], inputs=inputs, calculate_sensitivities=True)
            solver.solve(model, [0, 1], inputs=inputs)
        assert "unused" not in caplog.text

    def test_unused_inputs_logged_once_for_model_without_inputs(self, caplog):
        def no_input_model():
            model = pybamm.BaseModel()
            y = pybamm.Variable("y")
            model.rhs = {y: -y}
            model.initial_conditions = {y: 1.0}
            pybamm.Discretisation().process_model(model)
            return model

        inputs = {"unused": 1.0}
        message = "Inputs ['unused'] are not input parameters of model"
        model = no_input_model()
        solver = pybamm.IDAKLUSolver()
        with caplog.at_level(logging.WARNING, logger="pybamm.logger"):
            for _ in range(3):
                solver.solve(model, [0, 1], inputs=inputs, calculate_sensitivities=True)
        assert caplog.text.count(message) == 1

        caplog.clear()
        model = no_input_model()
        solver = pybamm.IDAKLUSolver()
        solution = None
        with caplog.at_level(logging.WARNING, logger="pybamm.logger"):
            for _ in range(3):
                solution = solver.step(
                    solution, model, 1, inputs=inputs, calculate_sensitivities=True
                )
        assert caplog.text.count(message) == 1

    def test_calculate_sensitivities_list_is_sorted(self):
        model = self._decay_model()
        solution = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12).solve(
            model,
            [0, 1],
            inputs={"y0": 2.0, "k": 0.3},
            calculate_sensitivities=["y0", "k"],
        )
        # The "all" columns follow sensitivity_names, not the requested order
        assert solution.sensitivity_names == ["k", "y0"]
        sensitivities = solution["y"].sensitivities
        np.testing.assert_array_equal(sensitivities["all"][:, 0], sensitivities["k"])
        np.testing.assert_array_equal(sensitivities["all"][:, 1], sensitivities["y0"])
        t = solution.t
        np.testing.assert_allclose(
            sensitivities["y0"].ravel(), np.exp(-0.3 * t), rtol=1e-6
        )

    @pytest.mark.parametrize("calculate_sensitivities", [True, ["a"], ["a", "c"]])
    def test_calculate_sensitivities_rejects_vector_inputs(
        self, calculate_sensitivities
    ):
        model = pybamm.BaseModel()
        y = pybamm.Variable("y")
        model.rhs = {
            y: -pybamm.Index(pybamm.InputParameter("a", expected_size=2), 0)
            * pybamm.InputParameter("c")
            * y
        }
        model.initial_conditions = {y: 1.0}
        pybamm.Discretisation().process_model(model)
        with pytest.raises(
            pybamm.SolverError,
            match=r"vector-valued input\(s\) \['a'\]; only scalar inputs",
        ):
            pybamm.IDAKLUSolver().solve(
                model,
                [0, 1],
                inputs={"a": np.array([[1.0], [2.0]]), "c": 0.5},
                calculate_sensitivities=calculate_sensitivities,
            )

    def test_wrangle_name(self):
        assert pybamm.BaseSolver._wrangle_name("Voltage [V]") == "v_voltage_v"
        assert pybamm.BaseSolver._wrangle_name("SoC [%]") == "v_soc_prc"
        assert (
            pybamm.BaseSolver._wrangle_name("Entropic change [V/K]")
            == "v_entropic_change_v_k"
        )
        assert pybamm.BaseSolver._wrangle_name("a + b * c - (d)") == "v_a_b_c_d"
        assert pybamm.BaseSolver._wrangle_name("Temp. [°C]") == "v_temp_c"
        assert pybamm.BaseSolver._wrangle_name("[]") == "v"

    def test_wrangle_names_are_distinct_valid_casadi_names(self):
        names = ["a/b", "a_b", "A-B", "x", "x jac", "x/jac", "", "%", "jac"]
        wrangled = pybamm.BaseSolver._wrangle_names(names)
        assert wrangled["a/b"] == "v_a_b"
        assert wrangled["a_b"] == "v_a_b_1"
        assert wrangled["A-B"] == "v_a_b_2"
        # "x" derives "v_x_jac" for its Jacobian
        assert wrangled["x jac"] == "v_x_jac_1"
        functions = set()
        for base in wrangled.values():
            for function_name in (
                base,
                f"{base}_jac",
                f"{base}_jac_action",
                f"d{base}_dp",
            ):
                assert function_name not in functions
                functions.add(function_name)
                # CasADi rejects invalid function names
                casadi.Function(function_name, [], [])

    def test_on_extrapolation_and_on_failure_settings(self):
        # Test setting different on_extrapolation and on_failure values on BaseSolver
        base_solver = pybamm.BaseSolver()

        # Test valid on_extrapolation values
        base_solver.on_extrapolation = "warn"
        assert base_solver.on_extrapolation == "warn"
        base_solver.on_extrapolation = "error"
        assert base_solver.on_extrapolation == "error"
        base_solver.on_extrapolation = "ignore"
        assert base_solver.on_extrapolation == "ignore"

        # Test valid on_failure values
        base_solver.on_failure = "warn"
        assert base_solver.on_failure == "warn"
        base_solver.on_failure = "error"
        assert base_solver.on_failure == "error"
        base_solver.on_failure = "ignore"
        assert base_solver.on_failure == "ignore"

        # Test invalid value
        with pytest.raises(
            ValueError, match=r"on_extrapolation must be 'warn', 'error', or 'ignore'"
        ):
            base_solver.on_extrapolation = "invalid"

        with pytest.raises(
            ValueError, match=r"on_failure must be 'warn', 'error', or 'ignore'"
        ):
            base_solver.on_failure = "invalid"

    @pytest.mark.parametrize(
        "format",
        [
            "casadi",
            "python",
            pytest.param(
                None,
                marks=pytest.mark.filterwarnings(
                    "ignore:convert_to_format=None is deprecated:DeprecationWarning"
                ),
            ),
            pytest.param(
                "jax",
                marks=pytest.mark.skipif(
                    not pybamm.has_jax(),
                    reason="jax or jaxlib is not installed",
                ),
            ),
        ],
    )
    def test_events_fail_on_initialisation_multiple_input_params(self, format):
        # Test that events fail on initialisation when multiple input parameters are used
        # if it's not the first set of parameters
        model = pybamm.BaseModel()
        model.convert_to_format = format
        u = pybamm.Variable("u")
        u0 = pybamm.InputParameter("u0")
        model.rhs = {u: -u}
        model.initial_conditions = {u: u0}
        model.events.append(pybamm.Event("test event", u - 0.2))
        model.variables = {"u": u}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.BaseSolver()

        initial_condition_input = [{"u0": 5}, {"u0": 0.1}]
        t_eval = np.array([0, 1])

        with pytest.raises(
            pybamm.SolverError,
            match=re.escape(
                "Events ['test event'] are non-positive at initial conditions with inputs {'u0': 0.1}"
            ),
        ):
            solver.solve(model, t_eval, initial_condition_input)

    def test_integrate_single_error(self):
        solver = pybamm.BaseSolver()
        model = pybamm.BaseModel()

        with pytest.raises(
            NotImplementedError,
            match=re.escape("BaseSolver does not implement _integrate_single."),
        ):
            solver._integrate_single(model, np.array([0, 1]), {}, np.array([1]))

    def test_one_process_solves_input_sets_without_a_pool(self, monkeypatch):
        # calc_ic=False routes consistent initialisation through the root
        # solver's BaseSolver._integrate, once for all input sets with nproc=1.
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        k = pybamm.InputParameter("k")
        model.rhs = {u: -u}
        model.algebraic = {v: v - k * u}
        model.initial_conditions = {u: 1.0, v: 0.0}
        model.variables = {"u": u, "v": v}
        pybamm.Discretisation().process_model(model)

        def no_pools(*args, **kwargs):
            raise AssertionError("a solve with nproc=1 built a process pool")

        monkeypatch.setattr(mp, "get_context", no_pools)

        solver = pybamm.IDAKLUSolver(options={"calc_ic": False})
        t_eval = np.linspace(0, 1, 10)
        solutions = solver.solve(model, t_eval, inputs=[{"k": 1.0}, {"k": 2.0}])
        for solution, k in zip(solutions, [1.0, 2.0], strict=True):
            np.testing.assert_allclose(
                solution["v"](t_eval), k * np.exp(-t_eval), rtol=1e-3, atol=1e-5
            )

    def test_single_input_set_uses_the_first_initial_state(self):
        solver = pybamm.BaseSolver()
        solver._integrate_single = lambda model, t_eval, inputs, y0: y0
        model = pybamm.BaseModel()
        model.y0_list = [np.array([1.0]), np.array([2.0])]
        (y0,) = solver._integrate(model, np.array([0.0]), [{}])
        np.testing.assert_array_equal(y0, [1.0])

    def test_discontinuity_events_different_times_error(self):
        # Test that an error is raised when discontinuity events occur at different
        # times for different input parameter sets
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        t_event = pybamm.InputParameter("t_event")
        model.rhs = {v: pybamm.t > t_event}
        model.initial_conditions = {v: 0}
        model.variables = {"v": v}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.BaseSolver()
        t_eval = np.linspace(0, 10)

        # Different input sets with discontinuities at different times
        inputs_list = [{"t_event": 3.0}, {"t_event": 5.0}]

        with pytest.raises(
            pybamm.SolverError,
            match="Discontinuity events occur at different times between input parameter sets",
        ):
            solver.solve(model, t_eval, inputs=inputs_list)

    @pytest.mark.parametrize(
        "all_sensitivities",
        [
            pytest.param({}, id="no_sensitivities_stored"),
            # IDAKLU stores output-width sensitivities here, which cannot seed
            # a state-width dy0/dp.
            pytest.param(
                {"p": np.zeros((2, 1)), "all": np.zeros((2, 1))}, id="output_width"
            ),
        ],
    )
    def test_check_restart_sensitivities_rejects_outputs_only(self, all_sensitivities):
        solution = pybamm.Solution(
            [np.array([0.0, 1.0])],
            [np.zeros((0, 2))],
            pybamm.BaseModel(),
            [{}],
            variables_returned=True,
        )
        solution._all_sensitivities = all_sensitivities

        with pytest.raises(
            pybamm.SolverError,
            match=r"Cannot continue a sensitivity solve from a solution that "
            r"returned output variables only",
        ):
            pybamm.BaseSolver._check_restart_sensitivities(solution, ["p"])

    def test_step_restarts_sensitivities_from_output_variables(self):
        # Equal output and state widths, so seeding from the outputs is silently wrong
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -pybamm.InputParameter("k") * u, v: u - 2 * v}
        model.initial_conditions = {u: 1.0, v: 0.5}
        model.variables = {"2u": 2 * u, "v": v}
        pybamm.Discretisation().process_model(model)
        solver = pybamm.IDAKLUSolver(
            output_variables=["2u", "v"], rtol=1e-8, atol=1e-10
        )

        solution = solver.step(
            None, model, 1.0, inputs={"k": 1.3}, calculate_sensitivities=True
        )
        solution = solver.step(
            solution, model, 1.0, inputs={"k": 1.3}, calculate_sensitivities=True
        )

        # u = exp(-k t), so d(2u)/dk = -2 t exp(-k t) across both steps
        t = solution.t
        np.testing.assert_allclose(
            solution["2u"].sensitivities["k"].ravel(),
            -2 * t * np.exp(-1.3 * t),
            rtol=1e-5,
            atol=1e-9,
        )

    def test_check_restart_sensitivities_allows_full_state(self):
        solution = pybamm.Solution(
            [np.array([0.0, 1.0])],
            [np.zeros((3, 2))],
            pybamm.BaseModel(),
            [{}],
            variables_returned=False,
        )
        solution._all_sensitivities = {"p": np.zeros((6, 1))}

        pybamm.BaseSolver._check_restart_sensitivities(solution, ["p"])

    def test_check_restart_sensitivities_allows_empty_solution(self):
        pybamm.BaseSolver._check_restart_sensitivities(pybamm.EmptySolution(), ["p"])


def _independent_decays():
    """u' = -a u and v' = -b v, with u(0) = v(0) = 1."""
    model = pybamm.BaseModel()
    u = pybamm.Variable("u")
    v = pybamm.Variable("v")
    model.rhs = {
        u: -pybamm.InputParameter("a") * u,
        v: -pybamm.InputParameter("b") * v,
    }
    model.initial_conditions = {u: 1.0, v: 1.0}
    model.variables = {"u": u, "v": v}
    pybamm.Discretisation().process_model(model)
    return model


def _scaled_decay(rate_factor, reference, scale):
    """u' = -rate_factor * k * u, u(0) = 1, with u stored as (u - reference) / scale."""
    model = pybamm.BaseModel()
    u = pybamm.Variable("u", reference=reference, scale=scale)
    model.rhs = {u: -rate_factor * pybamm.InputParameter("k") * u}
    model.initial_conditions = {u: 1.0}
    model.variables = {"u": u}
    pybamm.Discretisation().process_model(model)
    return model


def _switched_decay(algebraic=False):
    """y' = -k y + 2 H(t - t_switch), y(0) = 1, and optionally z = 2 y + k H."""
    model = pybamm.BaseModel()
    y = pybamm.Variable("y")
    k = pybamm.InputParameter("k")
    switch = pybamm.t > pybamm.InputParameter("t_switch")
    model.rhs = {y: -k * y + 2.0 * switch}
    model.initial_conditions = {y: 1.0}
    model.variables = {"y": y}
    if algebraic:
        z = pybamm.Variable("z")
        model.algebraic = {z: z - 2 * y - k * switch}
        model.initial_conditions[z] = 2.0
        model.variables["z"] = z
    return model


def _switched_decay_solution(t, k, t_switch):
    """y and dy/dk of `_switched_decay`, so z = 2 y + k H and dz/dk = 2 dy/dk + H."""
    after = t > t_switch
    decay = np.exp(-k * (t - t_switch))
    y = np.exp(-k * t) + after * (2 / k) * (1 - decay)
    dy_dk = -t * np.exp(-k * t) + after * (
        -(2 / k**2) * (1 - decay) + (2 / k) * (t - t_switch) * decay
    )
    return y, dy_dk


def _switched_decay_with_experiment_event():
    """y' = -k y + c H(t - t_switch), y(0) = 1, stopping when y falls to y_min."""
    model = pybamm.BaseModel()
    y = pybamm.Variable("y")
    switch = pybamm.t > pybamm.InputParameter("t_switch")
    model.rhs = {
        y: -pybamm.InputParameter("k") * y + pybamm.InputParameter("c") * switch
    }
    model.initial_conditions = {y: 1.0}
    model.variables = {"y": y}
    # step() continues past an event only if it is an experiment event
    y_min = pybamm.InputParameter("y_min")
    model.events = [pybamm.Event("y > y_min [experiment]", y - y_min)]
    return model


def _modulo_ramp(period=None):
    """y' = a (t mod P), y(0) = 0, with an input P unless `period` is given."""
    model = pybamm.BaseModel()
    y = pybamm.Variable("y")
    if period is None:
        period = pybamm.InputParameter("P")
    model.rhs = {y: pybamm.InputParameter("a") * pybamm.Modulo(pybamm.t, period)}
    model.initial_conditions = {y: 0.0}
    model.variables = {"y": y}
    return model


def _modulo_ramp_sensitivity(t, period):
    """dy/da of `_modulo_ramp`, so y = a dy/da, from y = a (n P^2 + r^2) / 2."""
    n = np.floor(t / period)
    return (n * period**2 + (t - n * period) ** 2) / 2


def _modulo_algebraic(period=None):
    """x' = z, x(0) = 0, with z = t mod P, and an input P unless `period` is given."""
    model = pybamm.BaseModel()
    x = pybamm.Variable("x")
    z = pybamm.Variable("z")
    if period is None:
        period = pybamm.InputParameter("P")
    model.rhs = {x: z}
    model.algebraic = {z: z - pybamm.Modulo(pybamm.t, period)}
    model.initial_conditions = {x: 0.0, z: 0.0}
    model.variables = {"x": x, "z": z}
    return model


class TestBaseSolverSensitivityRestarts:
    @staticmethod
    def _solver():
        return pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12)

    @staticmethod
    def _modulo_algebraic_solver():
        # Tighter tolerances fail to start where z is near zero, as after each reset
        return pybamm.IDAKLUSolver()

    @pytest.mark.parametrize("output_variables", [False, True])
    @pytest.mark.parametrize("algebraic", [False, True])
    @pytest.mark.parametrize(
        "t_eval", [[0, 3], np.linspace(0, 3, 31)], ids=["integer_list", "array"]
    )
    def test_input_dependent_discontinuity_restarts_each_input_set(
        self, algebraic, t_eval, output_variables
    ):
        # Each segment must restart from the end of the previous one, for the
        # states and their sensitivities
        model = _switched_decay(algebraic)
        inputs = [{"k": 0.5, "t_switch": 1.0}, {"k": 0.8, "t_switch": 1.0}]
        t_eval_before = np.array(t_eval)
        solver = pybamm.IDAKLUSolver(
            rtol=1e-10,
            atol=1e-12,
            output_variables=list(model.variables) if output_variables else None,
        )

        solutions = solver.solve(
            model, t_eval, inputs=inputs, calculate_sensitivities=["k"]
        )

        np.testing.assert_array_equal(t_eval, t_eval_before)
        for solution, solution_inputs in zip(solutions, inputs, strict=True):
            assert len(solution.sub_solutions) == 2
            t = solution.t
            exact, dexact_dk = _switched_decay_solution(t, solution_inputs["k"], 1.0)
            np.testing.assert_allclose(solution["y"](t), exact, rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(
                solution["y"].sensitivities["k"].ravel(),
                dexact_dk,
                rtol=1e-6,
                atol=1e-8,
            )
            if algebraic:
                np.testing.assert_allclose(
                    solution["z"].sensitivities["k"].ravel(),
                    2 * dexact_dk + (t > 1.0),
                    rtol=1e-6,
                    atol=1e-8,
                )

    @pytest.mark.parametrize("stopping_set", [0, 1])
    def test_input_dependent_discontinuity_restarts_running_input_sets(
        self, stopping_set
    ):
        # A set that stops on an event before the switch at t = 1 is not restarted,
        # and does not stop the other set
        model = _switched_decay()
        y_min = pybamm.InputParameter("y_min")
        model.events = [pybamm.Event("y > y_min", model.variables["y"] - y_min)]
        inputs = [{"k": 0.5, "t_switch": 1.0, "y_min": 0.0}] * 2
        inputs[stopping_set] = {"k": 3.0, "t_switch": 1.0, "y_min": 0.5}

        solutions = self._solver().solve(
            model, [0, 3], inputs=inputs, calculate_sensitivities=["k"]
        )

        for index, solution in enumerate(solutions):
            if index == stopping_set:
                assert solution.termination == "event: y > y_min"
                assert len(solution.sub_solutions) == 1
                assert solution.t[-1] == pytest.approx(np.log(2) / 3, rel=1e-6)
            else:
                assert solution.termination == "final time"
                assert len(solution.sub_solutions) == 2
                assert solution.t[-1] == 3
            t = solution.t
            exact, dexact_dk = _switched_decay_solution(t, inputs[index]["k"], 1.0)
            np.testing.assert_allclose(solution["y"](t), exact, rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(
                solution["y"].sensitivities["k"].ravel(),
                dexact_dk,
                rtol=1e-6,
                atol=1e-8,
            )

    @pytest.mark.parametrize("store_first_last", [False, True])
    def test_input_dependent_discontinuity_with_t_interp(self, store_first_last):
        # Switches at t = 1 and 2 split the solve into three segments, each storing
        # its end points and the t_interp values inside it
        solver = pybamm.IDAKLUSolver(
            rtol=1e-10, atol=1e-12, store_first_last=store_first_last
        )
        t_interp = None if store_first_last else [0, 0.5, 2.5, 3]

        solution = solver.solve(
            _modulo_ramp(),
            [0, 3],
            inputs={"a": 2.0, "P": 1.0},
            t_interp=t_interp,
            calculate_sensitivities=["a"],
        )

        expected_t = [0, 1, 1, 2, 2, 3]
        if t_interp is not None:
            expected_t = [0, 0.5, 1, 1, 2, 2, 2.5, 3]
        np.testing.assert_allclose(solution.t, expected_t, rtol=1e-12)
        dexact_da = _modulo_ramp_sensitivity(solution.t, 1.0)
        np.testing.assert_allclose(
            solution["y"](solution.t), 2.0 * dexact_da, rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(
            solution["y"].sensitivities["a"].ravel(), dexact_da, rtol=1e-7, atol=1e-9
        )

    @pytest.mark.parametrize(
        "t_interp, message",
        [([-1, 3], "greater than the smallest"), ([0, 4], "less than the greatest")],
    )
    def test_input_dependent_discontinuity_rejects_t_interp_outside_t_eval(
        self, t_interp, message
    ):
        with pytest.raises(pybamm.SolverError, match=message):
            self._solver().solve(
                _modulo_ramp(), [0, 3], inputs={"a": 2.0, "P": 1.0}, t_interp=t_interp
            )

    @pytest.mark.parametrize("t_switch", [-0.5, 0.0], ids=["negative", "zero"])
    def test_input_dependent_discontinuity_at_non_positive_time(self, t_switch):
        inputs = {"k": 0.5, "t_switch": t_switch}
        solution = self._solver().solve(
            _switched_decay(), [-1, 1], inputs=inputs, calculate_sensitivities=["k"]
        )

        before, after = solution.sub_solutions
        assert before.t[-1] < t_switch < after.t[0]
        t = solution.t
        # Starting from y(-1) = 1 shifts the solution from y(0) = 1 by one
        exact, dexact_dk = _switched_decay_solution(t + 1, inputs["k"], t_switch + 1)
        np.testing.assert_allclose(solution["y"](t), exact, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            solution["y"].sensitivities["k"].ravel(), dexact_dk, rtol=1e-6, atol=1e-8
        )

    @pytest.mark.parametrize(
        "t_point",
        [np.nextafter(1.0, -np.inf), 1.0, np.nextafter(1.0, np.inf)],
        ids=["below", "at", "above"],
    )
    def test_input_dependent_discontinuity_next_to_t_eval_point(self, t_point):
        # A t_eval point within one float of the switch is replaced, not repeated
        model = _switched_decay()
        inputs = {"k": 0.5, "t_switch": 1.0}
        solver = self._solver()
        solver.set_up(model, [inputs])

        start_indices, end_indices, t_eval = (
            solver._get_discontinuity_start_end_indices(
                model, model.y0_list[0], inputs, np.array([0.0, t_point, 2.0])
            )
        )

        assert np.all(np.diff(t_eval) > 0)
        assert start_indices[1] == end_indices[0]
        assert t_eval[end_indices[0] - 1] < 1.0 < t_eval[start_indices[1]]

    def test_input_dependent_discontinuity_within_one_float_of_t_eval_end(self):
        # There is no time to restart in, so the switch is integrated across
        t_eval = [0, 0.1 + 0.2]
        inputs = {"k": 0.5, "t_switch": 0.3}
        solution = self._solver().solve(
            _switched_decay(), t_eval, inputs=inputs, calculate_sensitivities=["k"]
        )

        assert len(solution.sub_solutions) == 1
        np.testing.assert_array_equal(solution.t[[0, -1]], t_eval)
        t = solution.t
        t_start = t_eval[0]
        exact, dexact_dk = _switched_decay_solution(
            t - t_start, inputs["k"], inputs["t_switch"] - t_start
        )
        np.testing.assert_allclose(solution["y"](t), exact, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            solution["y"].sensitivities["k"].ravel(), dexact_dk, rtol=1e-6, atol=1e-8
        )

    def test_input_dependent_discontinuity_one_float_after_t_eval_start(self):
        # There is no time to stop in before z jumps, so the solve starts after it
        t_switch = np.nextafter(1.0, np.inf)
        inputs = {"k": 0.5, "t_switch": t_switch}
        solution = self._solver().solve(
            _switched_decay(algebraic=True),
            [1.0, 3.0],
            inputs=inputs,
            calculate_sensitivities=["k"],
            t_interp=[1.0, 2.0, 3.0],
        )

        np.testing.assert_array_equal(
            solution.t, [np.nextafter(t_switch, np.inf), 2.0, 3.0]
        )
        t = solution.t
        exact, dexact_dk = _switched_decay_solution(t - 1.0, inputs["k"], 0.0)
        np.testing.assert_allclose(
            solution["z"](t), 2 * exact + inputs["k"], rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(
            solution["z"].sensitivities["k"].ravel(),
            2 * dexact_dk + 1,
            rtol=1e-6,
            atol=1e-8,
        )

    @pytest.mark.parametrize("discontinuity", ["switch", "modulo"])
    def test_input_dependent_discontinuity_times_match_casadi(self, discontinuity):
        # numpy's tanh can round differently from CasADi's, which the residual uses
        time = pybamm.tanh(pybamm.InputParameter("b")) + 1.5
        if discontinuity == "switch":
            model = _switched_decay()
            model.rhs = {
                y: rhs.create_copy() + (pybamm.t > time) for y, rhs in model.rhs.items()
            }
        else:
            model = _modulo_ramp(time)
        inputs = {"a": 2.0, "k": 0.5, "t_switch": 100.0, "b": 0.0}
        solver = self._solver()
        solver.set_up(model, [inputs])
        b = casadi.MX.sym("b")
        time_casadi = casadi.Function("time", [b], [time.to_casadi(inputs={"b": b})])

        for value in np.linspace(-3, 3, 101):
            inputs["b"] = value
            start_indices, end_indices, t_eval = (
                solver._get_discontinuity_start_end_indices(
                    model, model.y0_list[0], inputs, np.array([0.0, 10.0])
                )
            )

            stops = t_eval[np.array(end_indices[:-1]) - 1]
            restarts = t_eval[start_indices[1:]]
            time_value = float(time_casadi(value))
            if discontinuity == "switch":
                np.testing.assert_array_less(stops, time_value)
                np.testing.assert_array_less(time_value, restarts)
            else:
                drops = np.fmod(stops, time_value) - np.fmod(restarts, time_value)
                assert np.all(drops > time_value / 2)

    def test_input_dependent_discontinuities_with_python_format(self):
        # The python format evaluates the times with numpy, as its residual does
        model = _modulo_ramp()
        model.convert_to_format = "python"
        pybamm.Discretisation().process_model(model)
        inputs = {"a": 2.0, "P": 0.7}
        solver = pybamm.BaseSolver()
        solver.set_up(model, [inputs])

        start_indices, end_indices, t_eval = (
            solver._get_discontinuity_start_end_indices(
                model, model.y0_list[0], inputs, np.array([0.0, 3.0])
            )
        )

        restarts = t_eval[start_indices[1:]]
        np.testing.assert_allclose(restarts, [0.7, 1.4, 2.1, 2.8], rtol=1e-15)
        stops = t_eval[np.array(end_indices[:-1]) - 1]
        np.testing.assert_array_equal(stops, np.nextafter(restarts, -np.inf))

    @pytest.mark.parametrize("floats_apart", [1, 2])
    def test_input_dependent_discontinuities_within_one_float(self, floats_apart):
        # z jumps at both switches, so the solver must restart after the second
        model = _switched_decay(algebraic=True)
        z = model.variables["z"]
        model.algebraic[z] -= pybamm.t > pybamm.InputParameter("t_second")
        t_second = 1.0
        for _ in range(floats_apart):
            t_second = np.nextafter(t_second, np.inf)
        inputs = {"k": 0.5, "t_switch": 1.0, "t_second": t_second}

        solution = self._solver().solve(
            model, [0, 3], inputs=inputs, calculate_sensitivities=["k"]
        )

        before, after = solution.sub_solutions
        assert before.t[-1] < 1.0 and after.t[0] > t_second
        t = solution.t
        exact, dexact_dk = _switched_decay_solution(t, inputs["k"], 1.0)
        np.testing.assert_allclose(
            solution["z"](t),
            2 * exact + inputs["k"] * (t > 1.0) + (t > t_second),
            rtol=1e-7,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            solution["z"].sensitivities["k"].ravel(),
            2 * dexact_dk + (t > 1.0),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_step_restarts_at_input_dependent_discontinuity(self):
        # The second step crosses the switch at t = 1, where z jumps
        model = _switched_decay(algebraic=True)
        inputs = {"k": 0.5, "t_switch": 1.0}
        solver = self._solver()
        solution = solver.step(
            None, model, 0.5, inputs=inputs, calculate_sensitivities=["k"]
        )

        solution = solver.step(
            solution, model, 2.5, inputs=inputs, calculate_sensitivities=["k"]
        )

        assert len(solution.sub_solutions) == 3
        t = solution.t
        after = t > 1.0
        exact, dexact_dk = _switched_decay_solution(t, inputs["k"], 1.0)
        np.testing.assert_allclose(solution["y"](t), exact, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            solution["z"](t), 2 * exact + inputs["k"] * after, rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(
            solution["z"].sensitivities["k"].ravel(),
            2 * dexact_dk + after,
            rtol=1e-6,
            atol=1e-8,
        )

    def _steps_after_event(self, c, t_switch):
        # The first step stops at y = 0.6, at a time that depends on k, so the
        # two steps after it start at times that move with k
        model = _switched_decay_with_experiment_event()
        solver = self._solver()
        inputs = {"k": 0.5, "c": c, "t_switch": t_switch, "y_min": 0.6}
        solution = solver.step(
            None, model, 1.2, inputs=inputs, calculate_sensitivities=["k"]
        )
        assert solution.termination == "event: y > y_min [experiment]"

        inputs = {**inputs, "y_min": 0.0}
        steps = []
        for t_eval in (np.linspace(0, 1, 11), None):
            solution = solver.step(
                solution,
                model,
                1.0,
                t_eval=t_eval,
                inputs=inputs,
                calculate_sensitivities=["k"],
                save=False,
            )
            steps.append(solution)
        return steps

    def test_step_after_event_keeps_sensitivities_continuous_at_discontinuity(self):
        # The switch at t = 1.5 is fixed even though the step's start time moves,
        # so the sensitivities must not jump where the step restarts
        step, _ = self._steps_after_event(c=2.0, t_switch=1.5)

        assert len(step.sub_solutions) == 2
        after = np.searchsorted(step.t, 1.5)
        assert step.t[after - 1] == pytest.approx(1.5) == step.t[after]
        sensitivities = step["y"].sensitivities["k"].ravel()
        np.testing.assert_allclose(
            sensitivities[after], sensitivities[after - 1], rtol=1e-8
        )

    def test_step_after_event_matches_unsplit_step(self):
        # A switch that leaves the equations unchanged must not change either step,
        # whether it splits the first at t = 1.5 or is never reached
        split = self._steps_after_event(c=0.0, t_switch=1.5)
        unsplit = self._steps_after_event(c=0.0, t_switch=100.0)

        assert [len(step.sub_solutions) for step in split] == [2, 1]
        assert [len(step.sub_solutions) for step in unsplit] == [1, 1]
        for split_step, unsplit_step in zip(split, unsplit, strict=True):
            t = np.intersect1d(split_step.t, unsplit_step.t)
            assert len(t) >= 2
            np.testing.assert_allclose(
                split_step["y"](t), unsplit_step["y"](t), rtol=1e-7, atol=1e-9
            )
            np.testing.assert_allclose(
                split_step["y"].sensitivities["k"][np.isin(split_step.t, t)],
                unsplit_step["y"].sensitivities["k"][np.isin(unsplit_step.t, t)],
                rtol=1e-7,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                split_step.last_state["y"].sensitivities["k"],
                unsplit_step.last_state["y"].sensitivities["k"],
                rtol=1e-7,
                atol=1e-9,
            )

    def test_set_up_adds_discontinuity_events_once(self):
        # Changing the sensitivities sets the solver up again on the same model
        model = _switched_decay()
        solver = self._solver()
        for calculate_sensitivities in [False, True, ["k"], False]:
            solver.solve(
                model,
                [0, 3],
                inputs={"k": 0.5, "t_switch": 1.0},
                calculate_sensitivities=calculate_sensitivities,
            )
            assert len(model.events) == 1

    def test_modulo_with_input_period(self):
        model = _modulo_ramp()
        solver = self._solver()

        # A re-solve with a shorter period has more discontinuities in t_eval
        for period, number_of_segments in [(1.0, 3), (0.4, 8)]:
            solution = solver.solve(
                model,
                [0, 3],
                inputs={"a": 2.0, "P": period},
                calculate_sensitivities=["a"],
            )
            assert len(solution.sub_solutions) == number_of_segments
            t = solution.t
            dexact_da = _modulo_ramp_sensitivity(t, period)
            np.testing.assert_allclose(
                solution["y"](t), 2.0 * dexact_da, rtol=1e-7, atol=1e-9
            )
            np.testing.assert_allclose(
                solution["y"].sensitivities["a"].ravel(),
                dexact_da,
                rtol=1e-7,
                atol=1e-9,
            )

        with pytest.raises(pybamm.SolverError, match=r"must be positive"):
            solver.solve(model, [0, 3], inputs={"a": 2.0, "P": 0.0})

    @pytest.mark.parametrize("input_period", [False, True], ids=["constant", "input"])
    def test_step_stops_at_every_modulo_reset(self, input_period):
        # Each step spans 300 resets, and the second starts away from t = 0
        period = 0.01
        inputs = {"a": 2.0}
        if input_period:
            model = _modulo_ramp()
            inputs["P"] = period
        else:
            model = _modulo_ramp(period)
        solver = self._solver()

        solution = solver.step(None, model, 3.0, inputs=inputs)
        solution = solver.step(solution, model, 3.0, inputs=inputs)

        t = solution.t
        resets = period * np.arange(1, 600)
        first_after_reset = t[np.searchsorted(t, resets)]
        np.testing.assert_allclose(first_after_reset, resets, rtol=1e-12)
        np.testing.assert_allclose(
            solution["y"](t),
            2.0 * _modulo_ramp_sensitivity(t, period),
            rtol=1e-7,
            atol=1e-9,
        )

    @pytest.mark.parametrize("input_period", [False, True], ids=["constant", "input"])
    def test_modulo_before_zero(self, input_period):
        period = 0.5
        inputs = {"a": 2.0}
        if input_period:
            model = _modulo_ramp()
            inputs["P"] = period
        else:
            model = _modulo_ramp(period)

        solution = self._solver().solve(
            model, [-1.2, 0.8], inputs=inputs, calculate_sensitivities=["a"]
        )

        t = solution.t
        # t mod P is continuous at t = 0, so need not stop there
        resets = np.array([-1.0, -0.5, 0.5])
        first_after_reset = t[np.searchsorted(t, resets)]
        np.testing.assert_allclose(first_after_reset, resets, rtol=1e-12)
        # CasADi's fmod is odd in t, so its integral from t = 0 is even
        dexact_da_from_zero = _modulo_ramp_sensitivity(np.abs(t), period)
        dexact_da = dexact_da_from_zero - _modulo_ramp_sensitivity(1.2, period)
        np.testing.assert_allclose(
            solution["y"](t), 2.0 * dexact_da, rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(
            solution["y"].sensitivities["a"].ravel(), dexact_da, rtol=1e-7, atol=1e-9
        )

    @pytest.mark.parametrize("k", [5, 9, 10])
    def test_modulo_restarts_where_fmod_resets(self, k):
        # k * 0.1 rounds to just short of these resets, where fmod has not reset, so a
        # restart there would cross the jump in z straight away
        t_start = 0.1 * k - 0.05
        solution = self._modulo_algebraic_solver().solve(
            _modulo_algebraic(0.1), np.linspace(t_start, t_start + 0.1, 5)
        )

        t = solution.t
        np.testing.assert_allclose(
            solution["x"](t),
            _modulo_ramp_sensitivity(t, 0.1) - _modulo_ramp_sensitivity(t_start, 0.1),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_step_one_float_before_modulo_reset(self):
        # The first step ends two floats before the reset at t = 0.5, so the second
        # starts one float before it, where fmod has not reset
        model = _modulo_algebraic()
        inputs = {"P": 0.1}
        reset = np.nextafter(0.5, np.inf)
        t_end = np.nextafter(np.nextafter(reset, -np.inf), -np.inf)
        solver = self._modulo_algebraic_solver()
        solution = solver.step(None, model, t_end, inputs=inputs)

        solution = solver.step(solution, model, 0.1, inputs=inputs)

        assert solution.sub_solutions[-1].t[0] == reset
        t = solution.t
        np.testing.assert_allclose(
            solution["z"](t), np.fmod(t, 0.1), rtol=1e-5, atol=1e-5
        )

    # The suite ignores RuntimeWarning, but the resets raised one from inf * 0
    @pytest.mark.filterwarnings("error::RuntimeWarning")
    @pytest.mark.parametrize("input_period", [False, True], ids=["constant", "input"])
    def test_modulo_with_infinite_period(self, input_period):
        # t mod inf is t, which never resets
        inputs = {"a": 2.0}
        if input_period:
            model = _modulo_ramp()
            inputs["P"] = np.inf
        else:
            model = _modulo_ramp(np.inf)

        solution = self._solver().solve(model, [0, 3], inputs=inputs)

        assert len(solution.sub_solutions) == 1
        t = solution.t
        np.testing.assert_allclose(solution["y"](t), t**2, rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize(
        "period, message",
        [
            (1 + 0.1 * pybamm.t, "depends only on input parameters"),
            (1 + pybamm.StateVector(slice(0, 1)), "depends only on input parameters"),
            (pybamm.InputParameter("P", expected_size=2), "must be a scalar"),
            (pybamm.Scalar(0), "must be positive, but is 0.0"),
            (pybamm.Scalar(-1), "must be positive, but is -1.0"),
        ],
        ids=["time", "state", "vector", "zero", "negative"],
    )
    def test_check_modulo_period_rejects_invalid_period(self, period, message):
        with pytest.raises(pybamm.SolverError, match=message):
            pybamm.BaseSolver._check_modulo_period(period)

    def test_modulo_with_time_dependent_period_raises(self):
        # The resets of t mod (1 + 0.1 t) are not multiples of its initial period
        with pytest.raises(pybamm.SolverError, match=r"depends only on input"):
            self._solver().solve(
                _modulo_ramp(1 + 0.1 * pybamm.t), [0, 3], inputs={"a": 2.0}
            )

    @pytest.mark.parametrize("use_state_mapper", [False, True])
    @pytest.mark.parametrize(
        "reference, scale",
        [(0.5, 2.0), (pybamm.InputParameter("r"), pybamm.InputParameter("s"))],
        ids=["constant", "input"],
    )
    def test_step_to_new_model_maps_sensitivities(
        self, use_state_mapper, reference, scale
    ):
        # u = exp(-k t) then exp(-k) exp(-2 k (t - 1)) after the switch at t = 1,
        # independent of how u is stored, so du/dr = du/ds = 0
        inputs = {"k": 0.7}
        if isinstance(reference, pybamm.InputParameter):
            inputs.update({"r": 0.5, "s": 2.0})
        first = _scaled_decay(1.0, reference, scale)
        second = _scaled_decay(2.0, reference, scale)

        solution = self._solver().step(
            None, first, 1.0, inputs=inputs, calculate_sensitivities=True
        )
        state_mapper = None
        if use_state_mapper:
            vars_for_processing = pybamm.BaseSolver._get_vars_for_processing(
                first, dict(sorted(inputs.items()))
            )
            state_mapper = pybamm.solvers.base_solver.process(
                second.build_initial_state_mapper(first),
                "mapper",
                vars_for_processing,
            )[:3]
        solution = self._solver().step(
            solution,
            second,
            1.0,
            inputs=inputs,
            calculate_sensitivities=True,
            state_mapper=state_mapper,
            save=False,
        )

        t = solution.t
        k = inputs["k"]
        exact = np.exp(-k * (2 * t - 1))
        np.testing.assert_allclose(solution["u"](t), exact, rtol=1e-7)
        sensitivities = solution["u"].sensitivities
        np.testing.assert_allclose(
            sensitivities["k"].ravel(), -(2 * t - 1) * exact, rtol=1e-6, atol=1e-9
        )
        for name in set(inputs) - {"k"}:
            np.testing.assert_allclose(sensitivities[name].ravel(), 0, atol=1e-9)

    def test_step_seeds_sensitivities_by_name(self):
        model = _independent_decays()
        inputs = {"a": 0.3, "b": 0.6}
        solution = self._solver().step(
            None, model, 1.0, inputs=inputs, calculate_sensitivities=["a", "b"]
        )

        solution = self._solver().step(
            solution,
            model,
            1.0,
            inputs=inputs,
            calculate_sensitivities=["b"],
            save=False,
        )

        t = solution.t
        np.testing.assert_allclose(
            solution["v"].sensitivities["b"].ravel(),
            -t * np.exp(-0.6 * t),
            rtol=1e-6,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            solution["u"].sensitivities["b"].ravel(), 0, atol=1e-9
        )

    @pytest.mark.parametrize(
        "first_sensitivities, second_sensitivities",
        [(["a", "b"], ["b"]), (["a"], False)],
        ids=["fewer", "none"],
    )
    def test_saved_step_rejects_changed_sensitivities(
        self, first_sensitivities, second_sensitivities
    ):
        # The joined solution would label the second step's columns by the first's
        model = _independent_decays()
        inputs = {"a": 0.3, "b": 0.6}
        solver = self._solver()
        solution = solver.step(
            None, model, 1.0, inputs=inputs, calculate_sensitivities=first_sensitivities
        )

        with pytest.raises(pybamm.SolverError, match=r"Step with 'save=False'"):
            solver.step(
                solution,
                model,
                1.0,
                inputs=inputs,
                calculate_sensitivities=second_sensitivities,
            )

        assert model.calculate_sensitivities == first_sensitivities
        solution = solver.step(
            solution,
            model,
            1.0,
            inputs=inputs,
            calculate_sensitivities=first_sensitivities,
        )
        t = solution.t
        np.testing.assert_allclose(
            solution["u"].sensitivities["a"].ravel(),
            -t * np.exp(-0.3 * t),
            rtol=1e-6,
            atol=1e-9,
        )

    @pytest.mark.parametrize("same_solver", [True, False])
    def test_step_rejects_restart_without_requested_sensitivities(self, same_solver):
        model = _independent_decays()
        inputs = {"a": 0.3, "b": 0.6}
        solver = self._solver()
        solution = solver.step(
            None, model, 1.0, inputs=inputs, calculate_sensitivities=["a"]
        )
        next_solver = solver if same_solver else self._solver()

        # dv/db at the end of the first step is unknown, not dv/da
        with pytest.raises(pybamm.SolverError, match=r"no sensitivities for \['b'\]"):
            next_solver.step(
                solution, model, 1.0, inputs=inputs, calculate_sensitivities=["b"]
            )

        assert model.calculate_sensitivities == ["a"]
        solution = solver.step(
            solution,
            model,
            1.0,
            inputs=inputs,
            calculate_sensitivities=["a"],
            save=False,
        )
        t = solution.t
        np.testing.assert_allclose(
            solution["u"].sensitivities["a"].ravel(),
            -t * np.exp(-0.3 * t),
            rtol=1e-6,
            atol=1e-9,
        )

    @pytest.mark.parametrize(
        "first_sensitivities", [False, ["a"]], ids=["none", "fewer"]
    )
    def test_step_to_new_model_rejects_missing_sensitivities(self, first_sensitivities):
        inputs = {"a": 0.3, "b": 0.6}
        solution = self._solver().step(
            None,
            _independent_decays(),
            1.0,
            inputs=inputs,
            calculate_sensitivities=first_sensitivities,
        )

        with pytest.raises(pybamm.SolverError, match=r"no sensitivities for \[.*'b'\]"):
            self._solver().step(
                solution,
                _independent_decays(),
                1.0,
                inputs=inputs,
                calculate_sensitivities=True,
            )

    def test_step_after_changing_sensitivities(self):
        model = _independent_decays()
        inputs = {"a": 0.3, "b": 0.6}
        solver = self._solver()
        solver.solve(model, [0, 1], inputs=inputs, calculate_sensitivities=True)

        solution = solver.step(None, model, 1.0, inputs=inputs)
        assert not solution.has_sensitivities()

        solution = solver.step(
            None, model, 1.0, inputs=inputs, calculate_sensitivities=["b"]
        )
        solution = solver.step(
            solution, model, 1.0, inputs=inputs, calculate_sensitivities=["b"]
        )
        t = solution.t
        np.testing.assert_allclose(
            solution["v"].sensitivities["b"].ravel(),
            -t * np.exp(-0.6 * t),
            rtol=1e-6,
            atol=1e-9,
        )

    @pytest.mark.parametrize("method", ["solve", "step"])
    def test_second_model_error_keeps_sensitivity_setting(self, method):
        inputs = {"a": 0.3, "b": 0.6}
        solver = self._solver()
        solver.step(
            None,
            _independent_decays(),
            1.0,
            inputs=inputs,
            calculate_sensitivities=["a"],
        )
        model = _independent_decays()
        self._solver().solve(
            model, [0, 1], inputs=inputs, calculate_sensitivities=["b"]
        )

        with pytest.raises(RuntimeError, match=r"already been initialised"):
            if method == "solve":
                solver.solve(model, [0, 1], inputs=inputs, calculate_sensitivities=True)
            else:
                solver.step(
                    None, model, 1.0, inputs=inputs, calculate_sensitivities=True
                )

        assert model.calculate_sensitivities == ["b"]
