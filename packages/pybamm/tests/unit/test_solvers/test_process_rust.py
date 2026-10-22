import numpy as np
import pytest
import scipy.sparse

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
