"""Boundary validation for the Rust ExprGraph constructors."""

import pickle

import numpy as np
import pytest

from pybamm.rust import CompiledModel, ExprGraph


class TestStateVectorValidation:
    def test_state_vector_inverted_range_raises(self):
        g = ExprGraph()
        with pytest.raises(ValueError, match=r"start"):
            g.state_vector(2, 1)

    def test_state_vector_dot_inverted_range_raises(self):
        g = ExprGraph()
        with pytest.raises(ValueError, match=r"start"):
            g.state_vector_dot(2, 1)

    def test_index_inverted_range_raises(self):
        g = ExprGraph()
        y = g.state_vector(0, 4)
        with pytest.raises(ValueError, match=r"start"):
            g.index(y, 3, 1)

    def test_valid_ranges_still_work(self):
        g = ExprGraph()
        y = g.state_vector(0, 4)
        g.state_vector(1, 1)
        g.index(y, 1, 3)

    def test_a_state_extent_past_the_tape_index_range_raises(self):
        g = ExprGraph()
        y = g.state_vector(2**32, 2**32 + 1)
        with pytest.raises(ValueError, match=r"u32 index range"):
            g.compile(y)

    def test_a_state_count_past_the_tape_index_range_raises(self):
        g = ExprGraph()
        y = g.state_vector(0, 1)
        with pytest.raises(ValueError, match=r"largest state count"):
            g.compile(y, n_states=2**45).jacobian()


class TestSparseMatrixValidation:
    def test_non_monotonic_indptr_raises(self):
        g = ExprGraph()
        with pytest.raises(ValueError, match=r"CSR|indptr"):
            g.sparse_matrix([0, 2, 1], [0, 1], np.array([1.0, 2.0]), 2, 2)

    def test_out_of_range_column_raises(self):
        g = ExprGraph()
        with pytest.raises(ValueError, match=r"CSR|column"):
            g.sparse_matrix([0, 1, 2], [0, 5], np.array([1.0, 2.0]), 2, 2)

    def test_indptr_wrong_length_raises(self):
        g = ExprGraph()
        with pytest.raises(ValueError, match=r"CSR|indptr"):
            g.sparse_matrix([0, 2], [0, 1], np.array([1.0, 2.0]), 2, 2)

    def test_valid_sparse_matrix_still_works(self):
        g = ExprGraph()
        g.sparse_matrix([0, 1, 2], [0, 1], np.array([1.0, 2.0]), 2, 2)


class TestInterpolantValidation:
    def test_empty_grid_raises(self):
        g = ExprGraph()
        y = g.state_vector(0, 1)
        with pytest.raises(ValueError, match=r"interpolant|non-empty"):
            g.interpolant_1d_linear([], [], y)

    def test_length_mismatch_raises(self):
        g = ExprGraph()
        y = g.state_vector(0, 1)
        with pytest.raises(ValueError, match=r"interpolant|length"):
            g.interpolant_1d_linear([0.0, 1.0], [1.0], y)

    def test_non_increasing_grid_raises(self):
        g = ExprGraph()
        y = g.state_vector(0, 1)
        with pytest.raises(ValueError, match=r"interpolant|increasing"):
            g.interpolant_1d_linear([0.0, 2.0, 1.0], [1.0, 2.0, 3.0], y)

    def test_valid_interpolant_still_works(self):
        g = ExprGraph()
        y = g.state_vector(0, 1)
        g.interpolant_1d_linear([0.0, 1.0, 2.0], [10.0, 20.0, 30.0], y)

    def test_cubic_coeffs_with_a_short_tail_raise(self):
        g = ExprGraph()
        y = g.state_vector(0, 1)
        with pytest.raises(ValueError, match=r"4-tuples"):
            g.interpolant_1d_cubic([0.0, 1.0, 2.0], [1.0] * 7, y)

    def test_nd_interpolant_child_count_must_match_the_axes(self):
        g = ExprGraph()
        y = g.state_vector(0, 1)
        with pytest.raises(ValueError, match=r"expected 2 children"):
            g.interpolant_nd([[0.0, 1.0], [0.0, 1.0]], [0.0] * 4, 2, [y])


class TestConditionalValidation:
    def test_conditional_without_branches_raises(self):
        g = ExprGraph()
        selector = g.state_vector(0, 1)
        with pytest.raises(ValueError, match=r"at least one branch"):
            g.conditional(selector, [])

    def test_conditional_with_one_branch_still_works(self):
        g = ExprGraph()
        selector = g.state_vector(0, 1)
        g.conditional(selector, [g.scalar(1.0)])


def _evaluate(graph, method, expr, y, y_dot, inputs):
    if method == "eval_to_float":
        return graph.eval_to_float(expr, 0.0, y, y_dot, inputs)
    return graph.eval_to_array(expr, 0.0, np.array(y), np.array(y_dot), inputs)


class TestEvalHelperValidation:
    @pytest.mark.parametrize("method", ["eval_to_float", "eval_to_array"])
    @pytest.mark.parametrize("n_padding_nodes", [0, 10])
    def test_a_foreign_expression_is_rejected(self, method, n_padding_nodes):
        # Padding pushes the foreign node id past the evaluating graph's arena.
        g = ExprGraph()
        g.scalar(11.0)
        other = ExprGraph()
        for _ in range(n_padding_nodes):
            other.scalar(0.0)
        foreign = other.scalar(99.0)
        with pytest.raises(ValueError, match=r"different ExprGraph"):
            _evaluate(g, method, foreign, [], [], [])

    def test_dump_dag_rejects_a_foreign_expression(self, tmp_path):
        foreign = ExprGraph().scalar(1.0)
        with pytest.raises(ValueError, match=r"different ExprGraph"):
            ExprGraph().dump_dag(foreign, str(tmp_path / "dag.bin"), "model", 0, 0)

    @pytest.mark.parametrize(
        ("y", "y_dot", "inputs", "name"),
        [
            ([1.0], [1.0, 1.0], [1.0], "y"),
            ([1.0, 1.0], [1.0], [1.0], "y_dot"),
            ([1.0, 1.0], [1.0, 1.0], [], "inputs"),
        ],
    )
    @pytest.mark.parametrize("method", ["eval_to_float", "eval_to_array"])
    def test_a_buffer_shorter_than_the_expression_reads_is_rejected(
        self, method, y, y_dot, inputs, name
    ):
        g = ExprGraph()
        k = g.input_parameter("k")
        second_state = g.index(g.state_vector(0, 2), 1, 2)
        second_rate = g.index(g.state_vector_dot(0, 2), 1, 2)
        expr = g.mul(k, g.add(second_state, second_rate))
        with pytest.raises(ValueError, match=rf"reads \d+ value\(s\) of {name}, got"):
            _evaluate(g, method, expr, y, y_dot, inputs)

    @pytest.mark.parametrize("method", ["eval_to_float", "eval_to_array"])
    def test_buffers_longer_than_the_expression_reads_are_accepted(self, method):
        g = ExprGraph()
        expr = g.index(g.state_vector(0, 2), 1, 2)
        result = _evaluate(g, method, expr, [1.0, 2.0, 3.0], [], [4.0])
        np.testing.assert_array_equal(np.ravel(result), [2.0])

    def test_eval_to_float_rejects_a_vector_expression(self):
        g = ExprGraph()
        with pytest.raises(ValueError, match=r"evaluates to 2 values"):
            g.eval_to_float(g.state_vector(0, 2), 0.0, [3.0, 4.0], [], [])

    def test_eval_to_array_non_contiguous_y_raises(self):
        g = ExprGraph()
        expr = g.state_vector(0, 3)
        strided = np.arange(6.0)[::2]
        with pytest.raises(TypeError, match=r"contiguous"):
            g.eval_to_array(expr, 0.0, strided, np.array([]), [])

    def test_eval_to_float_empty_result_raises(self):
        g = ExprGraph()
        y = g.state_vector(0, 4)
        empty = g.index(y, 1, 1)
        with pytest.raises(ValueError):
            g.eval_to_float(empty, 0.0, [0.0, 0.0, 0.0, 0.0], [], [])


def _identity_mass(n):
    return (
        np.ones(n),
        np.arange(n + 1, dtype=np.int64),
        np.arange(n, dtype=np.int64),
    )


def _algebraic_mass(n):
    """CSR triple for the ``n x n`` zero matrix: every state algebraic."""
    return (
        np.zeros(0),
        np.zeros(n + 1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )


class TestCompiledModelWidthValidation:
    """``sens_param_indices`` are registration indices; ``n_inputs`` is packed width."""

    @staticmethod
    def _graph_with_a_vector_input():
        # Registration indices 0 ("wide", width 3) and 1 ("k"): packed width 4.
        g = ExprGraph()
        y = g.state_vector(0, 2)
        g.input_parameter("wide", 3)
        k = g.input_parameter("k")
        return g, g.mul(k, y)

    def test_sens_index_past_the_last_input_raises(self):
        g, rhs = self._graph_with_a_vector_input()
        data, indptr, indices = _identity_mass(2)
        # Inside the packed width (4), past the last registration index (1).
        for bad in (2, 3):
            with pytest.raises(ValueError, match=r"names no input parameter"):
                CompiledModel.from_expr(
                    g,
                    rhs,
                    data,
                    indptr,
                    indices,
                    n_inputs=4,
                    sens_param_indices=[bad],
                )

    def test_sens_index_naming_a_real_input_still_works(self):
        g, rhs = self._graph_with_a_vector_input()
        data, indptr, indices = _identity_mass(2)
        model = CompiledModel.from_expr(
            g, rhs, data, indptr, indices, n_inputs=4, sens_param_indices=[1]
        )
        assert model.n_sens_params == 1

    def test_sens_index_naming_a_vector_input_raises(self):
        g, rhs = self._graph_with_a_vector_input()
        data, indptr, indices = _identity_mass(2)
        with pytest.raises(ValueError, match=r"names an input of width 3"):
            CompiledModel.from_expr(
                g, rhs, data, indptr, indices, n_inputs=4, sens_param_indices=[0]
            )

    def test_repeated_sens_index_raises(self):
        g, rhs = self._graph_with_a_vector_input()
        data, indptr, indices = _identity_mass(2)
        with pytest.raises(ValueError, match=r"more than once"):
            CompiledModel.from_expr(
                g, rhs, data, indptr, indices, n_inputs=4, sens_param_indices=[1, 1]
            )

    def test_under_declared_n_inputs_raises(self):
        # "wide" is registered but never read: the residual alone would accept 1.
        g = ExprGraph()
        y = g.state_vector(0, 2)
        k = g.input_parameter("k")
        g.input_parameter("wide", 3)
        data, indptr, indices = _identity_mass(2)
        with pytest.raises(ValueError, match=r"n_inputs=1 is below the packed width 4"):
            CompiledModel.from_expr(g, g.mul(k, y), data, indptr, indices, n_inputs=1)

    def test_omitted_n_inputs_adopts_the_registered_width(self):
        g, rhs = self._graph_with_a_vector_input()
        data, indptr, indices = _identity_mass(2)
        model = CompiledModel.from_expr(g, rhs, data, indptr, indices)
        assert model.n_inputs == 4
        wider = CompiledModel.from_expr(g, rhs, data, indptr, indices, n_inputs=6)
        assert wider.n_inputs == 6

    def test_n_states_below_the_residual_extent_raises(self):
        # n_states is inferred from the mass matrix, so a 2x2 mass declares 2.
        g = ExprGraph()
        wide = g.state_vector(0, 8)
        data, indptr, indices = _identity_mass(2)
        with pytest.raises(ValueError, match=r"below the .* state"):
            CompiledModel.from_expr(g, g.neg(wide), data, indptr, indices)

    def test_output_below_the_declared_state_count_raises(self):
        g = ExprGraph()
        narrow = g.state_vector(0, 2)
        wide = g.state_vector(0, 8)
        data, indptr, indices = _identity_mass(2)
        with pytest.raises(ValueError, match=r"below the .* state"):
            CompiledModel.from_expr(
                g, g.neg(narrow), data, indptr, indices, output_exprs=[wide]
            )

    def test_algebraic_variable_index_outside_the_state_count_raises(self):
        g = ExprGraph()
        y = g.state_vector(0, 2)
        data, indptr, indices = _algebraic_mass(2)
        with pytest.raises(ValueError, match=r"algebraic variable index 2"):
            CompiledModel.from_expr(
                g,
                g.neg(y),
                data,
                indptr,
                indices,
                algebraic_expr=g.neg(y),
                algebraic_variable_indices=[0, 2],
            )

    @pytest.mark.parametrize("given", [[1, 0], [1, 1], [0, 0, 1]])
    def test_unordered_or_repeated_algebraic_variable_indices_raise(self, given):
        g = ExprGraph()
        y = g.state_vector(0, 2)
        data, indptr, indices = _algebraic_mass(2)
        with pytest.raises(ValueError, match=r"strictly ascending"):
            CompiledModel.from_expr(
                g,
                g.neg(y),
                data,
                indptr,
                indices,
                algebraic_expr=g.neg(y),
                algebraic_variable_indices=given,
            )

    def test_non_square_algebraic_block_raises(self):
        # The C++ Newton drivers factorise the block as n_alg x n_alg.
        g = ExprGraph()
        y = g.state_vector(0, 2)
        data, indptr, indices = _algebraic_mass(2)
        with pytest.raises(ValueError, match=r"2 residual rows but 1 variable"):
            CompiledModel.from_expr(
                g,
                g.neg(y),
                data,
                indptr,
                indices,
                algebraic_expr=g.neg(y),
                algebraic_variable_indices=[0],
            )

    def test_algebraic_variable_index_naming_a_differential_state_raises(self):
        g = ExprGraph()
        y = g.state_vector(0, 2)
        data, indptr, indices = _identity_mass(2)
        with pytest.raises(ValueError, match=r"index 0 names a differential state"):
            CompiledModel.from_expr(
                g,
                g.neg(y),
                data,
                indptr,
                indices,
                algebraic_expr=g.neg(y),
                algebraic_variable_indices=[0, 1],
            )

    def test_algebraic_variable_indices_without_an_algebraic_expr_raise(self):
        g = ExprGraph()
        y = g.state_vector(0, 2)
        data, indptr, indices = _algebraic_mass(2)
        with pytest.raises(ValueError, match=r"need an algebraic_expr"):
            CompiledModel.from_expr(
                g, g.neg(y), data, indptr, indices, algebraic_variable_indices=[0, 1]
            )

    def test_event_below_the_declared_state_count_raises(self):
        g = ExprGraph()
        narrow = g.state_vector(0, 2)
        wide = g.state_vector(0, 8)
        data, indptr, indices = _identity_mass(2)
        with pytest.raises(ValueError, match=r"below the .* state"):
            CompiledModel.from_expr(
                g, g.neg(narrow), data, indptr, indices, event_exprs=[wide]
            )

    def test_a_residual_shorter_than_the_state_count_raises(self):
        g = ExprGraph()
        data, indptr, indices = _identity_mass(2)
        with pytest.raises(ValueError, match=r"residual has 1 rows but n_states=2"):
            CompiledModel.from_expr(g, g.scalar(1.0), data, indptr, indices)


class TestCompiledModelEvalValidation:
    """The direct evaluators reject lengths that evaluation would index past."""

    @staticmethod
    def _model():
        g = ExprGraph()
        k = g.input_parameter("k")
        return CompiledModel.from_expr(
            g, g.mul(k, g.state_vector(0, 2)), *_identity_mass(2)
        )

    @pytest.mark.parametrize(
        ("y", "yp", "inputs", "name"),
        [
            (np.zeros(0), np.zeros(2), np.ones(1), "y"),
            (np.zeros(3), np.zeros(2), np.ones(1), "y"),
            (np.zeros(2), np.zeros(1), np.ones(1), "yp"),
            (np.zeros(2), np.zeros(2), np.ones(0), "inputs"),
            (np.zeros(2), np.zeros(2), np.ones(2), "inputs"),
        ],
    )
    def test_residual_rejects_a_wrong_length(self, y, yp, inputs, name):
        model = self._model()
        with pytest.raises(ValueError, match=rf"expected {name} of length"):
            model.eval_residual(0.0, y, yp, inputs)
        with pytest.raises(ValueError, match=rf"expected {name} of length"):
            model.eval_residual_into(0.0, y, yp, inputs, np.zeros(2))

    @pytest.mark.parametrize(
        ("y", "inputs", "name"),
        [
            (np.zeros(1), np.ones(1), "y"),
            (np.zeros(2), np.ones(0), "inputs"),
        ],
    )
    def test_jacobian_assembly_rejects_a_wrong_length(self, y, inputs, name):
        model = self._model()
        with pytest.raises(ValueError, match=rf"expected {name} of length"):
            model.assemble_jacobian_csc_into(0.0, y, 1.0, inputs, np.zeros(model.nnz))

    def test_a_non_contiguous_state_raises(self):
        model = self._model()
        with pytest.raises(TypeError, match=r"contiguous"):
            model.eval_residual(0.0, np.arange(4.0)[::2], np.zeros(2), np.ones(1))

    def test_matching_lengths_still_evaluate(self):
        model = self._model()
        residual = model.eval_residual(
            0.0, np.array([1.0, 2.0]), np.zeros(2), np.array([3.0])
        )
        np.testing.assert_allclose(residual, [-3.0, -6.0])


class TestInputRegistration:
    def test_re_registering_a_name_reuses_its_slot(self):
        g = ExprGraph()
        g.input_parameter("wide", 3)
        g.input_parameter("k")
        g.input_parameter("wide", 3)
        assert g.n_inputs() == 4

    def test_re_registering_a_name_at_another_width_raises(self):
        g = ExprGraph()
        g.input_parameter("k", 2)
        with pytest.raises(ValueError, match=r"'k' re-registered with width 3"):
            g.input_parameter("k", 3)

    def test_a_pickled_graph_rebuilds_its_registry_from_the_arena(self):
        g = ExprGraph()
        y = g.state_vector(0, 2)
        g.input_parameter("wide", 3)
        k = g.input_parameter("k")
        data, indptr, indices = _identity_mass(2)
        model = CompiledModel.from_expr(g, g.mul(k, y), data, indptr, indices)

        restored = pickle.loads(pickle.dumps(g))
        assert restored.n_inputs() == g.n_inputs() == model.n_inputs
        # The restored registry still knows its names rather than restarting at index 0.
        restored.input_parameter("wide", 3)
        assert restored.n_inputs() == 4
