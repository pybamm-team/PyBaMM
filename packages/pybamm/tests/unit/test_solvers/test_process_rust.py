import numpy as np
import pytest
import scipy.sparse

import pybamm
from tests import get_discretisation_for_testing

casadi = pytest.importorskip("casadi")


class TestCompiledFunctionADContract:
    def _cf(self, n_states=2):
        # f(t, y, a) = [y0*a - y1, y0 + y1*y1] over a 2-state system, 1 input.
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        a = graph.input_parameter("a")
        y0 = graph.state_vector(0, 1)
        y1 = graph.state_vector(1, 2)
        expr = graph.concat([y0 * a - y1, y0 + y1 * y1])
        return graph.compile(expr, name="rhs", n_states=n_states)

    def test_primal_eval(self):
        cf = self._cf()
        out = cf(0.0, np.array([1.5, -2.0]), np.array([3.0]))
        np.testing.assert_allclose(np.asarray(out).ravel(), [6.5, 5.5])

    def test_p_accepts_dict_or_array(self):
        cf = self._cf()
        y = np.array([1.5, -2.0])
        np.testing.assert_allclose(
            np.asarray(cf(0.0, y, np.array([3.0]))).ravel(),
            np.asarray(cf(0.0, y, {"a": 3.0})).ravel(),
        )

    def test_jacobian_y_is_scipy_csc(self):
        cf = self._cf()
        jac = cf.jacobian("y")(0.0, np.array([1.5, -2.0]), np.array([3.0]))
        assert scipy.sparse.issparse(jac)
        np.testing.assert_allclose(jac.toarray(), [[3.0, -1.0], [1.0, -4.0]])

    def test_jvp_matches_jac_matvec(self):
        cf = self._cf()
        y, p, v = np.array([1.5, -2.0]), np.array([3.0]), np.array([0.3, 0.7])
        jv = np.asarray(cf.jvp(0.0, y, p, v)).ravel()
        full = cf.jacobian("y")(0.0, y, p).toarray() @ v
        np.testing.assert_allclose(jv, full)

    def test_jacobian_p_columns(self):
        cf = self._cf()
        # df/da = [y0 = 1.5, 0.0]
        jp = cf.jacobian("p")(0.0, np.array([1.5, -2.0]), np.array([3.0]))
        np.testing.assert_allclose(jp.toarray()[:, 0], [1.5, 0.0])

    def test_rectangular_group_jacobian(self):
        # output_len (1) != n_states (2): the rhs sub-block of a DAE.
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        y0 = graph.state_vector(0, 1)
        y1 = graph.state_vector(1, 2)
        cf = graph.compile(graph.concat([y0 * y1]), name="RHS", n_states=2)
        jac = cf.jacobian("y")(0.0, np.array([1.5, -2.0]), np.empty(0))
        assert jac.shape == (1, 2)
        np.testing.assert_allclose(jac.toarray(), [[-2.0, 1.5]])


class TestTrajectoryFFILooseDtypes:
    """The 1-D time params on the trajectory entry points coerce loose dtypes
    (lists, float32, non-contiguous slices) rather than rejecting anything that
    is not a strict contiguous float64 ndarray."""

    def _cf(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        expr = (2 * pybamm.StateVector(slice(0, 1))).to_rust(graph, {})
        return graph.compile(expr, name="f", n_states=1)

    def _group(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        expr = (2 * pybamm.StateVector(slice(0, 1))).to_rust(graph, {})
        return graph.compile_group({"double": expr})

    def test_eval_trajectory_hermite_accepts_loose_t_query(self):
        cf = self._cf()
        ts = np.array([0.0, 1.0])
        ys = np.array([[1.0, 2.0]])
        yps = np.array([[1.0, 1.0]])
        query = [0.25, 0.75]
        expected = np.asarray(
            cf.eval_trajectory_hermite(np.array(query), ts, ys, yps, np.array([]))
        )

        as_list = query
        as_float32 = np.array(query, dtype=np.float32)
        # genuinely non-contiguous: a length-1 strided slice is trivially
        # contiguous, so use two elements to force a real stride gap.
        strided = np.array([0.25, -1.0, 0.75, -1.0])[::2]
        assert not strided.flags["C_CONTIGUOUS"]

        for t_query in (as_list, as_float32, strided):
            out = np.asarray(
                cf.eval_trajectory_hermite(t_query, ts, ys, yps, np.array([]))
            )
            np.testing.assert_allclose(out, expected)

    def test_eval_trajectory_hermite_accepts_loose_ts(self):
        cf = self._cf()
        ys = np.array([[1.0, 2.0]])
        yps = np.array([[1.0, 1.0]])
        tq = np.array([0.5])
        expected = np.asarray(
            cf.eval_trajectory_hermite(tq, np.array([0.0, 1.0]), ys, yps, np.array([]))
        )
        for ts in ([0.0, 1.0], np.array([0.0, 1.0], dtype=np.float32)):
            out = np.asarray(cf.eval_trajectory_hermite(tq, ts, ys, yps, np.array([])))
            np.testing.assert_allclose(out, expected)

    def test_eval_trajectory_accepts_loose_ts(self):
        cf = self._cf()
        y_traj = np.array([[1.0, 2.0, 3.0]])
        p = np.array([])
        expected = np.asarray(cf.eval_trajectory(np.array([0.0, 0.5, 1.0]), y_traj, p))
        for ts in ([0.0, 0.5, 1.0], np.array([0.0, 0.5, 1.0], dtype=np.float32)):
            out = np.asarray(cf.eval_trajectory(ts, y_traj, p))
            np.testing.assert_allclose(out, expected)

    def test_jvp_trajectory_accepts_loose_ts(self):
        cf = self._cf()
        y_traj = np.array([[1.0, 2.0, 3.0]])
        vy_traj = np.array([[0.1, 0.2, 0.3]])
        p = np.array([])
        expected = np.asarray(
            cf.jvp_trajectory(np.array([0.0, 0.5, 1.0]), y_traj, p, vy_traj)
        )
        for ts in ([0.0, 0.5, 1.0], np.array([0.0, 0.5, 1.0], dtype=np.float32)):
            out = np.asarray(cf.jvp_trajectory(ts, y_traj, p, vy_traj))
            np.testing.assert_allclose(out, expected)

    def test_group_eval_trajectory_accepts_loose_ts(self):
        group = self._group()
        y_traj = np.array([[1.0, 2.0, 3.0]])
        p = np.array([])
        (expected,) = group.eval_trajectory(np.array([0.0, 0.5, 1.0]), y_traj, p)
        for ts in ([0.0, 0.5, 1.0], np.array([0.0, 0.5, 1.0], dtype=np.float32)):
            (out,) = group.eval_trajectory(ts, y_traj, p)
            np.testing.assert_allclose(np.asarray(out), np.asarray(expected))

    def test_group_eval_trajectory_hermite_accepts_loose_t_query(self):
        group = self._group()
        ts = np.array([0.0, 1.0])
        ys = np.array([[1.0, 2.0]])
        yps = np.array([[1.0, 1.0]])
        p = np.array([])
        (expected,) = group.eval_trajectory_hermite(np.array([0.5]), ts, ys, yps, p)
        for t_query in ([0.5], np.array([0.5], dtype=np.float32)):
            (out,) = group.eval_trajectory_hermite(t_query, ts, ys, yps, p)
            np.testing.assert_allclose(np.asarray(out), np.asarray(expected))


class TestDenseMatrixToRust:
    def test_dense_matrix_matmul_to_rust(self):
        from pybamm.rust import ExprGraph
        from pybamm.solvers.rust_evaluator import RustEvaluator

        A = np.arange(6.0).reshape(2, 3)
        y = pybamm.StateVector(slice(0, 3))
        expr = pybamm.Matrix(A) @ y
        graph = ExprGraph()
        cf = graph.compile(expr.to_rust(graph, {}), name="f", n_states=3)
        yv = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            np.asarray(cf(0.0, yv, np.array([]))).ravel(), A @ yv
        )
        # jacobian path: d(A @ y)/dy == A, exercising sparsity/tangent on a dense LHS
        ev = RustEvaluator(cf, "jac")
        np.testing.assert_allclose(ev(0.0, yv, np.array([])).toarray(), A)


class TestMatMulLhsValidation:
    def test_matmul_non_constant_lhs_raises_clean_error(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        lhs = graph.state_vector(0, 2)  # computed (non-constant) LHS
        rhs = graph.state_vector(0, 2)
        mm = graph.matmul(lhs, rhs)
        with pytest.raises(NotImplementedError, match=r"MatMul left operand"):
            graph.compile(mm, name="f", n_states=2)

    def test_matmul_non_constant_lhs_raises_on_eval_to_array(self):
        # eval_to_array bypassed check_supported (direct CompiledExpr::new),
        # so a non-constant MatMul LHS panicked instead of raising cleanly.
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        sv = graph.state_vector(0, 2)
        mm = graph.matmul(sv, sv)
        with pytest.raises(NotImplementedError, match=r"MatMul left operand"):
            graph.eval_to_array(mm, 0.0, np.array([1.0, 2.0]), np.array([]), [])


class TestExpressionShapeValidation:
    def test_incompatible_binary_widths_raise_before_evaluation(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        short = graph.state_vector(0, 2)
        long = graph.state_vector(2, 5)
        invalid = short + long

        with pytest.raises(ValueError, match=r"incompatible operand widths 2 and 3"):
            graph.compile(invalid, name="invalid", n_states=5)


class TestRustEvaluatorWrappers:
    def _cf(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        a = graph.input_parameter("a")
        y0 = graph.state_vector(0, 1)
        y1 = graph.state_vector(1, 2)
        expr = graph.concat([y0 * a - y1, y0 + y1 * y1])
        return graph.compile(expr, name="rhs", n_states=2)

    def test_func_returns_column(self):
        from pybamm.solvers.rust_evaluator import RustEvaluator

        cf = self._cf()
        out = RustEvaluator(cf, "func")(0.0, np.array([[1.5], [-2.0]]), np.array([3.0]))
        assert out.shape == (2, 1)
        np.testing.assert_allclose(out.ravel(), [6.5, 5.5])

    def test_jac_returns_scipy_csc(self):
        from pybamm.solvers.rust_evaluator import RustEvaluator

        cf = self._cf()
        jac = RustEvaluator(cf, "jac")(0.0, np.array([1.5, -2.0]), np.array([3.0]))
        assert scipy.sparse.issparse(jac)
        np.testing.assert_allclose(jac.toarray(), [[3.0, -1.0], [1.0, -4.0]])

    def test_jac_action_matches_jac_matvec(self):
        from pybamm.solvers.rust_evaluator import RustEvaluator

        cf = self._cf()
        y, p, v = np.array([1.5, -2.0]), np.array([3.0]), np.array([0.3, 0.7])
        jv = RustEvaluator(cf, "jac_action")(0.0, y, p, v)
        full = RustEvaluator(cf, "jac")(0.0, y, p).toarray() @ v
        np.testing.assert_allclose(jv.ravel(), full)

    def test_jacp_tuple_per_param(self):
        from pybamm.solvers.rust_evaluator import RustEvaluator

        cf = self._cf()
        # sens_indices=[0] selects the "a" column of df/dp
        out = RustEvaluator(cf, "jacp", sens_indices=[0])(
            0.0, np.array([1.5, -2.0]), np.array([3.0])
        )
        assert isinstance(out, tuple) and len(out) == 1
        np.testing.assert_allclose(out[0].ravel(), [1.5, 0.0])

    def test_jacobian_is_derived_lazily(self):
        from pybamm.solvers.rust_evaluator import RustEvaluator

        cf = self._cf()
        ev = RustEvaluator(cf, "jac")
        assert ev._jac is None  # not derived at construction
        ev(0.0, np.array([1.5, -2.0]), np.array([3.0]))
        assert ev._jac is not None  # derived + cached on first call

    def test_func_accepts_eval_only(self):
        from pybamm.solvers.rust_evaluator import RustEvaluator

        cf = self._cf()
        out = RustEvaluator(cf, "func")(0.0, np.array([1.5, -2.0]), np.array([3.0]))
        assert out.shape == (2, 1)
        np.testing.assert_allclose(out.ravel(), [6.5, 5.5])


def _jacp_as_list(result):
    """Normalise a jacp call result to a list of 1-D arrays.

    casadi returns a single DM (not a tuple) when there is exactly one output,
    while RustEvaluator always returns a tuple.
    """
    if isinstance(result, casadi.DM):
        result = (result,)
    return [np.asarray(j).ravel() for j in result]


def _toy_dae(convert_to_format):
    model = pybamm.BaseModel()
    u = pybamm.Variable("u")
    v = pybamm.Variable("v")
    a = pybamm.InputParameter("a")
    model.rhs = {u: -2 * u + a * v}
    model.algebraic = {v: 2 * u - v}
    model.initial_conditions = {u: 1.0, v: 2.0}
    model.events = [pybamm.Event("u-0.05", u - 0.05)]
    model.variables = {"u": u, "v": v}
    disc = pybamm.Discretisation()
    disc.process_model(model)
    model.convert_to_format = convert_to_format
    return model


class TestProcessRustParity:
    @pytest.fixture
    def pair(self):
        from pybamm.solvers.base_solver import BaseSolver

        inputs = {"a": 0.5}
        out = {}
        for fmt in ("casadi", "rust"):
            model = _toy_dae(fmt)
            model.calculate_sensitivities = ["a"]
            vars_ = BaseSolver._get_vars_for_processing(model, inputs)
            rhs_alg = pybamm.numpy_concatenation(
                model.concatenated_rhs, model.concatenated_algebraic
            )
            out[fmt] = (model, vars_, rhs_alg)
        return out

    def _stack(self, fmt, inputs):
        if fmt == "casadi":
            return casadi.vertcat(*inputs.values())
        return np.array(list(inputs.values()), dtype=np.float64)

    def test_rhs_algebraic_func_jac_action_jacp(self, pair):
        from pybamm.solvers.base_solver import process

        t, y, inputs = 0.3, np.array([1.1, 0.4]), {"a": 0.5}
        results = {}
        for fmt, (_model, vars_, rhs_alg) in pair.items():
            func, jac, jacp, jac_action = process(rhs_alg, "rhs_algebraic", vars_)
            p = self._stack(fmt, inputs)
            jac_mat = jac(t, y, p)
            results[fmt] = {
                "f": np.asarray(func(t, y, p)).ravel(),
                "jac": np.asarray(
                    jac_mat.toarray() if hasattr(jac_mat, "toarray") else jac_mat
                ),
                "jv": np.asarray(jac_action(t, y, p, np.array([0.2, -0.3]))).ravel(),
                "jacp": _jacp_as_list(jacp(t, y, p)),
            }
        for key in ("f", "jac", "jv"):
            np.testing.assert_allclose(
                results["rust"][key], results["casadi"][key], rtol=1e-12, atol=1e-14
            )
        for r, c in zip(
            results["rust"]["jacp"], results["casadi"]["jacp"], strict=True
        ):
            np.testing.assert_allclose(r, c, rtol=1e-12, atol=1e-14)

    def test_event_and_ic_eval_only(self, pair):
        from pybamm.solvers.base_solver import process

        t, y, inputs = 0.0, np.array([1.0, 2.0]), {"a": 0.5}
        vals = {}
        for fmt, (model, vars_, _) in pair.items():
            ev = process(
                model.events[0].expression, "event_0", vars_, use_jacobian=False
            )[0]
            ic = process(
                model.concatenated_initial_conditions,
                "initial_conditions",
                vars_,
                use_jacobian=False,
            )[0]
            p = self._stack(fmt, inputs)
            vals[fmt] = (
                float(np.asarray(ev(t, y, p)).item()),
                np.asarray(ic(t, np.zeros((2, 1)), p)).ravel(),
            )
        assert vals["rust"][0] == pytest.approx(vals["casadi"][0], rel=1e-12)
        np.testing.assert_allclose(vals["rust"][1], vals["casadi"][1], rtol=1e-12)

    def test_rectangular_group_jacobian_matches_casadi(self, pair):
        from pybamm.solvers.base_solver import process

        t, y = 0.3, np.array([1.1, 0.4])
        mats = {}
        for fmt, (model, vars_, _) in pair.items():
            _, jac, _, jac_action = process(model.concatenated_rhs, "RHS", vars_)
            assert jac is not None and jac_action is not None
            p = self._stack(fmt, inputs={"a": 0.5})
            jm = jac(t, y, p)
            mats[fmt] = np.asarray(jm.toarray() if hasattr(jm, "toarray") else jm)
        np.testing.assert_allclose(mats["rust"], mats["casadi"], rtol=1e-12, atol=1e-14)


class TestProcessRustDuplication:
    # These exercise the BaseSolver.set_up skip logic in isolation, via a minimal
    # whole-model-artifact solver rather than the real IDAKLUSolver rust path.
    def test_lazy_jac_not_built_for_uncalled_groups(self):
        from pybamm.solvers.base_solver import BaseSolver, process
        from pybamm.solvers.rust_evaluator import RustEvaluator

        model = _toy_dae("rust")
        model.calculate_sensitivities = []
        vars_ = BaseSolver._get_vars_for_processing(model, {"a": 0.5})
        _, jac, _, jac_action = process(model.concatenated_rhs, "RHS", vars_)
        assert (
            isinstance(jac, RustEvaluator)
            and jac._jac is None
            and jac_action is not None
        )  # not yet derived

    def test_whole_model_solver_skips_every_per_group_lowering(self, monkeypatch):
        from pybamm.solvers.base_solver import BaseSolver

        # Minimal concrete subclass with the whole-model-artifact flag
        class _WholeModelSolver(BaseSolver):
            _integrates_via_compiled_model = True

            def _run(self, *a, **kw):  # pragma: no cover
                raise NotImplementedError

        model = _toy_dae("rust")
        solver = _WholeModelSolver()

        # Patch _set_initial_conditions so base set_up can be called without the
        # full initial-condition plumbing.
        monkeypatch.setattr(
            BaseSolver, "_set_initial_conditions", lambda *a, **kw: None
        )
        # Patch _set_up_events to return empty structures
        monkeypatch.setattr(
            BaseSolver,
            "_set_up_events",
            lambda *a, **kw: ([], [], [], {}, []),
        )

        solver.set_up(model, inputs=[{"a": 0.5}])

        # Every group is served from the solver's own shared lowering, which fills
        # rhs_eval via RustModelLowering.bind_generic_evaluators.
        assert model.rhs_eval is None
        assert model.jac_rhs_eval is None
        assert model.jac_rhs_action_eval is None
        assert model.jacp_rhs_eval is None
        assert model.algebraic_eval is None
        assert model.rhs_algebraic_eval is None

    def test_rust_output_block_skipped(self, monkeypatch):
        from pybamm.solvers.base_solver import BaseSolver

        class _WholeModelSolver(BaseSolver):
            _integrates_via_compiled_model = True

            def _run(self, *a, **kw):  # pragma: no cover
                raise NotImplementedError

        model = _toy_dae("rust")
        solver = _WholeModelSolver(output_variables=["u"])

        monkeypatch.setattr(
            BaseSolver, "_set_initial_conditions", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            BaseSolver,
            "_set_up_events",
            lambda *a, **kw: ([], [], [], {}, []),
        )

        solver.set_up(model, inputs=[{"a": 0.5}])

        assert solver.computed_var_fcns == {}


class TestStackedInputPredicate:
    def test_uses_stacked_inputs(self):
        model = pybamm.BaseModel()
        for fmt, expected in [
            ("casadi", True),
            ("rust", True),
            ("python", False),
            ("jax", False),
            (None, False),
        ]:
            model.convert_to_format = fmt
            assert model.uses_stacked_inputs is expected

    def test_stack_inputs_rust_is_ndarray(self):
        from pybamm.solvers.base_solver import stack_inputs

        out = stack_inputs({"a": 1.0, "b": np.array([2.0, 3.0])}, "rust")
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
        assert stack_inputs({}, "rust").size == 0

    def test_set_initial_conditions_and_event_check_rust(self):
        from pybamm.solvers.base_solver import BaseSolver

        class _DaeSolver(BaseSolver):
            def _run(self, *a, **kw):  # pragma: no cover
                raise NotImplementedError

        model = _toy_dae("rust")
        solver = _DaeSolver()
        # set_up exercises _set_initial_conditions and the rust IC evaluator
        solver.set_up(model, inputs=[{"a": 0.5}], ics_only=True)
        y0 = np.asarray(model.y0_list[0]).ravel()
        np.testing.assert_allclose(y0, [1.0, 2.0])
        # event check must not raise for positive events
        model.terminate_events_eval = []
        solver._check_event_violation([0.0], model, y0, {"a": 0.5})


class TestRustSolverGuards:
    def test_casadi_solver_rejects_rust_model(self):
        model = _toy_dae("rust")
        with pytest.raises(pybamm.SolverError, match="convert_to_format='rust'"):
            pybamm.CasadiSolver()._check_and_prepare_model_inplace(model)
        assert model.convert_to_format == "rust"  # never silently forced

    def test_casadi_root_method_switches_to_rust_newton(self):
        class _DaeSolver(pybamm.BaseSolver):
            def _run(self, *a, **kw):  # pragma: no cover
                raise NotImplementedError

        model = _toy_dae("rust")
        solver = _DaeSolver()
        solver.root_method = "casadi"
        solver._check_and_prepare_model_inplace(model)
        assert isinstance(solver.root_method, pybamm.NonlinearSolver)
        assert model.convert_to_format == "rust"  # never silently forced


class TestRustAlgebraicMinimize:
    def test_minimize_converges_on_rust_model(self):
        # scipy minimize needs the exact gradient of sum(f**2), and jac_norm's
        # broadcasting must hold for numpy jacobians as well as casadi DM.
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.algebraic = {var1: var1 - 3, var2: 2 * var1 - var2}
        model.initial_conditions = {var1: pybamm.Scalar(1), var2: pybamm.Scalar(4)}
        model.variables = {"var1": var1, "var2": var2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)
        model.convert_to_format = "rust"

        solution = pybamm.AlgebraicSolver("minimize", tol=1e-8).solve(model)
        np.testing.assert_allclose(
            model.get_processed_variable("var1").evaluate(t=None, y=solution.y),
            3,
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            model.get_processed_variable("var2").evaluate(t=None, y=solution.y),
            6,
            rtol=1e-7,
            atol=1e-6,
        )


class TestVectorWidthInputParameter:
    """`ExprGraph.input_parameter(name, width)` — packed-offset support for
    vector-valued (`expected_size > 1`) inputs."""

    def test_default_width_is_scalar(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        graph.input_parameter("a")
        assert graph.n_inputs() == 1

    def test_n_inputs_is_total_packed_width_not_name_count(self):
        # n_inputs() sizes the packed `p` buffer at the FFI boundary, so it must be
        # the total width, not the count of distinct names (2 names, 3 values).
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        graph.input_parameter("a")
        graph.input_parameter("b", 2)
        assert graph.n_inputs() == 3

    def test_vector_input_indexes_into_packed_offset(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        a = graph.input_parameter("a")  # width 1, offset 0
        b = graph.input_parameter("b", 2)  # width 2, offset 1
        y0 = graph.state_vector(0, 1)
        b0 = graph.index(b, 0, 1)
        b1 = graph.index(b, 1, 2)
        expr = graph.concat([(b0 + b1) * a * y0])
        cf = graph.compile(expr, name="f", n_states=1)
        out = np.asarray(cf(0.0, np.array([2.0]), np.array([3.0, 0.2, 0.3])))
        np.testing.assert_allclose(out, [3.0 * (0.2 + 0.3) * 2.0])

    def test_check_p_reports_total_width_not_name_count(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        a = graph.input_parameter("a")
        b = graph.input_parameter("b", 2)
        expr = graph.concat([a + graph.index(b, 0, 1) + graph.index(b, 1, 2)])
        cf = graph.compile(expr, name="f", n_states=0)
        with pytest.raises(
            ValueError,
            match=r"f: expected 3 input values \(2 parameters\), got 2",
        ):
            cf(0.0, np.empty(0), np.array([1.0, 2.0]))

    def test_pack_dict_path_validates_vector_length(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        a = graph.input_parameter("a")
        b = graph.input_parameter("b", 2)
        expr = graph.concat([a + graph.index(b, 0, 1) + graph.index(b, 1, 2)])
        cf = graph.compile(expr, name="f", n_states=0)
        # correct-length dict path matches the array path
        via_array = np.asarray(cf(0.0, np.empty(0), np.array([1.0, 0.2, 0.3])))
        via_dict = np.asarray(
            cf(0.0, np.empty(0), {"a": 1.0, "b": np.array([[0.2], [0.3]])})
        )
        np.testing.assert_allclose(via_array, via_dict)
        # wrong-length value for 'b' must be rejected, not silently truncated
        with pytest.raises(
            ValueError, match=r"input 'b' must have 2 value\(s\), got 3"
        ):
            cf(0.0, np.empty(0), {"a": 1.0, "b": np.array([0.2, 0.3, 0.4])})

    def test_reregistering_input_with_different_width_raises(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        graph.input_parameter("a")
        with pytest.raises(ValueError, match=r"'a'.*width 2.*width 1"):
            graph.input_parameter("a", 2)


class TestPickle:
    def test_expr_graph_pickle_roundtrip(self):
        import pickle

        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        (pybamm.InputParameter("a") * pybamm.StateVector(slice(0, 2))).to_rust(
            graph, {}
        )
        g2 = pickle.loads(pickle.dumps(graph))
        # same expression converts and compiles identically on the restored graph
        expr2 = (pybamm.InputParameter("a") * pybamm.StateVector(slice(0, 2))).to_rust(
            g2, {}
        )
        cf = g2.compile(expr2, name="f", n_states=2)
        np.testing.assert_allclose(
            np.asarray(cf(0.0, np.array([1.0, 2.0]), np.array([3.0]))).ravel(),
            [3.0, 6.0],
        )

    def test_pickle_preserves_input_registration_indices(self):
        import pickle

        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        graph.input_parameter("a")
        graph.input_parameter("b")
        g2 = pickle.loads(pickle.dumps(graph))
        # "b" must still resolve to index 1 on the restored graph; if the
        # input map were lost, re-registration would give "b" index 0 ("a").
        expr_b = (pybamm.InputParameter("b") * pybamm.StateVector(slice(0, 2))).to_rust(
            g2, {}
        )
        cf = g2.compile(expr_b, name="fb", n_states=2)
        np.testing.assert_allclose(
            np.asarray(cf(0.0, np.array([1.0, 2.0]), np.array([5.0, 7.0]))).ravel(),
            [7.0, 14.0],
        )

    def test_pickle_preserves_input_widths(self):
        import pickle

        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        graph.input_parameter("a")
        graph.input_parameter("b", 2)
        g2 = pickle.loads(pickle.dumps(graph))
        # re-registering "b" with a different width on the restored graph
        # must still be rejected — the width table round-tripped, not lost.
        with pytest.raises(ValueError, match=r"'b'.*width 1.*width 2"):
            g2.input_parameter("b", 1)
        # "b" must still occupy packed offset 1 (after "a"'s width-1 slot)
        # and keep its width-2 slot on the restored graph.
        b = pybamm.InputParameter("b", expected_size=2)
        expr_b = (pybamm.Index(b, 0) + pybamm.Index(b, 1)).to_rust(g2, {})
        cf = g2.compile(expr_b, name="fb", n_states=0)
        np.testing.assert_allclose(
            np.asarray(cf(0.0, np.empty(0), np.array([5.0, 7.0, 11.0]))),
            [18.0],
        )

    def test_compiled_function_pickle_roundtrip(self):
        import pickle

        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        expr = (pybamm.InputParameter("a") * pybamm.StateVector(slice(0, 2))).to_rust(
            graph, {}
        )
        cf = graph.compile(expr, name="f", n_states=2)
        cf2 = pickle.loads(pickle.dumps(cf))
        y, p = np.array([1.0, 2.0]), np.array([3.0])
        np.testing.assert_allclose(
            np.asarray(cf2(0.0, y, p)), np.asarray(cf(0.0, y, p))
        )
        # jacobian still derivable after the round-trip
        np.testing.assert_allclose(
            cf2.jacobian("y")(0.0, y, p).toarray(),
            cf.jacobian("y")(0.0, y, p).toarray(),
        )

    def test_rebuild_rejects_out_of_range_root(self):
        from pybamm.rust import CompiledFunction, ExprGraph

        graph = ExprGraph()
        graph.state_vector(0, 2)
        with pytest.raises(ValueError, match=r"out of range"):
            CompiledFunction._rebuild(graph, 2**31, None, None)

    def test_rebuild_rejects_unsupported_root(self):
        from pybamm.rust import CompiledFunction, ExprGraph

        graph = ExprGraph()
        # non-constant MatMul LHS is a lowering blocker; _rebuild must apply
        # the same check_supported gate as graph.compile
        mm = graph.matmul(graph.state_vector(0, 2), graph.state_vector(0, 2))
        with pytest.raises(NotImplementedError, match=r"MatMul left operand"):
            CompiledFunction._rebuild(graph, mm.id, None, None)

    def test_compiled_model_pickle_roundtrip(self):
        import pickle

        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 0.1 * v}
        model.algebraic = {v: 2 * u - v}
        model.initial_conditions = {u: 0, v: 0}
        model.convert_to_format = "rust"
        disc = pybamm.Discretisation()
        disc.process_model(model)
        solver = pybamm.IDAKLUSolver()
        solver.solve(model, np.array([0.0, 1.0]))
        rm = solver._setup["rust_model"]
        rm2 = pickle.loads(pickle.dumps(rm))
        assert rm2.nnz == rm.nnz
        np.testing.assert_array_equal(
            rm2.csc_sparsity_pattern()[0], rm.csc_sparsity_pattern()[0]
        )

    def test_rust_evaluator_pickle_rederives_jacobian(self):
        import pickle

        from pybamm.rust import ExprGraph
        from pybamm.solvers.rust_evaluator import RustEvaluator

        graph = ExprGraph()
        expr = (3 * pybamm.StateVector(slice(0, 2))).to_rust(graph, {})
        ev = RustEvaluator(graph.compile(expr, name="f", n_states=2), "jac")
        y, p = np.array([1.0, 2.0]), np.array([])
        expected = ev(0.0, y, p).toarray()  # populates the _jac cache
        ev2 = pickle.loads(pickle.dumps(ev))
        np.testing.assert_allclose(ev2(0.0, y, p).toarray(), expected)
