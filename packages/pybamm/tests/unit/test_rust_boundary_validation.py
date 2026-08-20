"""Boundary validation for the Rust ExprGraph constructors.

Malformed ranges, matrices, and interpolation tables supplied at the Python
boundary must raise ordinary ``ValueError``s rather than panicking, wrapping in
release arithmetic, or reading out of bounds inside evaluation.
"""

import numpy as np
import pytest

from pybamm.rust import ExprGraph


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
        # end == start (empty) and normal ranges must not raise
        g.state_vector(1, 1)
        g.index(y, 1, 3)


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


class TestEvalHelperValidation:
    def test_eval_to_array_non_contiguous_y_raises(self):
        # TypeError (not PanicException): the layout error every other eval
        # path raises for a strided array.
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
