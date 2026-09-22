import numpy as np
import pytest
import scipy.sparse

import pybamm

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
    """Trajectory time params coerce lists, float32 and non-contiguous slices."""

    def _cf(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        expr = (2 * pybamm.StateVector(slice(0, 1))).to_rust(graph, {})
        return graph.compile(expr, name="f", n_states=1)

    def _named(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        expr = (2 * pybamm.StateVector(slice(0, 1))).to_rust(graph, {})
        return graph.compile_outputs({"double": expr})

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
        # Two elements: a length-1 strided slice is trivially contiguous.
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

    def test_named_outputs_eval_trajectory_accepts_loose_ts(self):
        fn = self._named()
        y_traj = np.array([[1.0, 2.0, 3.0]])
        p = np.array([])
        (expected,) = fn.split(fn.eval_trajectory(np.array([0.0, 0.5, 1.0]), y_traj, p))
        for ts in ([0.0, 0.5, 1.0], np.array([0.0, 0.5, 1.0], dtype=np.float32)):
            (out,) = fn.split(fn.eval_trajectory(ts, y_traj, p))
            np.testing.assert_allclose(np.asarray(out), np.asarray(expected))

    def test_named_outputs_eval_trajectory_hermite_accepts_loose_t_query(self):
        fn = self._named()
        ts = np.array([0.0, 1.0])
        ys = np.array([[1.0, 2.0]])
        yps = np.array([[1.0, 1.0]])
        p = np.array([])
        (expected,) = fn.split(
            fn.eval_trajectory_hermite(np.array([0.5]), ts, ys, yps, p)
        )
        for t_query in ([0.5], np.array([0.5], dtype=np.float32)):
            (out,) = fn.split(fn.eval_trajectory_hermite(t_query, ts, ys, yps, p))
            np.testing.assert_allclose(np.asarray(out), np.asarray(expected))


class TestMatMulLhsValidation:
    def test_matmul_non_constant_lhs_raises_clean_error(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        lhs = graph.state_vector(0, 2)
        rhs = graph.state_vector(0, 2)
        mm = graph.matmul(lhs, rhs)
        with pytest.raises(NotImplementedError, match=r"MatMul left operand"):
            graph.compile(mm, name="f", n_states=2)

    def test_matmul_non_constant_lhs_raises_on_eval_to_array(self):
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


class TestVectorWidthInputParameter:
    def test_default_width_is_scalar(self):
        from pybamm.rust import ExprGraph

        graph = ExprGraph()
        graph.input_parameter("a")
        assert graph.n_inputs() == 1

    def test_n_inputs_is_total_packed_width_not_name_count(self):
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
        via_array = np.asarray(cf(0.0, np.empty(0), np.array([1.0, 0.2, 0.3])))
        via_dict = np.asarray(
            cf(0.0, np.empty(0), {"a": 1.0, "b": np.array([[0.2], [0.3]])})
        )
        np.testing.assert_allclose(via_array, via_dict)
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
        # "b" must still resolve to index 1, not 0, on the restored graph.
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
        with pytest.raises(ValueError, match=r"'b'.*width 1.*width 2"):
            g2.input_parameter("b", 1)
        # "b" must keep packed offset 1 and width 2 on the restored graph.
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
        # _rebuild must apply the same check_supported gate as graph.compile.
        mm = graph.matmul(graph.state_vector(0, 2), graph.state_vector(0, 2))
        with pytest.raises(NotImplementedError, match=r"MatMul left operand"):
            CompiledFunction._rebuild(graph, mm.id, None, None)
