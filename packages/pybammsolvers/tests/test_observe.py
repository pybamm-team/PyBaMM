from __future__ import annotations

import casadi
import numpy as np
import pytest
import pybammsolvers.idaklu as idaklu

NY = 8
NT = 20
N_INTERP = 100


@pytest.fixture(scope="session")
def test_data():
    rng = np.random.default_rng(0)
    ts = np.linspace(0, 1, NT, dtype=np.float64)
    ys = np.asfortranarray(rng.standard_normal((NY, NT)).astype(np.float64))
    yps = np.asfortranarray(rng.standard_normal((NY, NT)).astype(np.float64))
    inputs = np.array([2.0, 0.5], dtype=np.float64)
    t_interp = np.linspace(0, 1, N_INTERP, dtype=np.float64)
    return ts, ys, yps, inputs, t_interp


def _make_vecs(ts, ys, yps, inputs):
    return (
        idaklu.VectorRealtypeNdArray([ts]),
        idaklu.VectorRealtypeNdArray([ys]),
        idaklu.VectorRealtypeNdArray([yps]),
        idaklu.VectorRealtypeNdArray([inputs]),
    )


def _make_dense_and_sparse_pair(name, inputs_sym, outputs_expr):
    n_in = len(inputs_sym)
    f_sparse = casadi.Function(name, inputs_sym, [outputs_expr])
    diff_flags = [True] * n_in
    diff_flags[1] = False
    f_dense = casadi.Function(
        name, inputs_sym, [outputs_expr], {"is_diff_in": diff_flags}
    )

    assert f_sparse.is_diff_in(1) is True
    assert f_dense.is_diff_in(1) is False

    return f_sparse.serialize(), f_dense.serialize()


def _casadi_symbols():
    t = casadi.SX.sym("t")
    y = casadi.SX.sym("y", NY)
    p = casadi.SX.sym("p", 2)
    return t, y, p


class TestObserve:
    pytestmark = pytest.mark.unit

    def test_hermite_accuracy_against_analytic_function(self, test_data):
        t_knots = np.linspace(0, 2 * np.pi, N_INTERP, dtype=np.float64)
        ys = np.zeros((NY, len(t_knots)), dtype=np.float64, order="F")
        yps = np.zeros((NY, len(t_knots)), dtype=np.float64, order="F")
        ys[0, :] = np.sin(t_knots)
        yps[0, :] = np.cos(t_knots)

        inputs = np.array([1.0, 0.0], dtype=np.float64)
        t_interp = np.linspace(0, 2 * np.pi, 10 * N_INTERP, dtype=np.float64)

        t, y, p = _casadi_symbols()
        expr = y[0]
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ts_v, ys_v, yps_v, inputs_v = _make_vecs(t_knots, ys, yps, inputs)
        shape = [1, len(t_interp)]

        result_sparse = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [sparse_str], shape
        )
        result_dense = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [dense_str], shape
        )

        exact = np.sin(t_interp)

        np.testing.assert_array_equal(result_sparse, result_dense)
        np.testing.assert_allclose(result_sparse.flatten(), exact, atol=1e-6)
        np.testing.assert_allclose(result_dense.flatten(), exact, atol=1e-6)


class TestObserveSparsity:
    pytestmark = pytest.mark.unit

    def test_scalar_sparse_y(self, test_data):
        ts, ys, _, inputs, _ = test_data
        t, y, p = _casadi_symbols()

        expr = y[1] * p[0] + y[5] * p[1] + t
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ts_v, ys_v, _, inputs_v = _make_vecs(ts, ys, None, inputs)
        shape = [1, NT]

        result_sparse = idaklu.observe(ts_v, ys_v, inputs_v, [sparse_str], True, shape)
        result_dense = idaklu.observe(ts_v, ys_v, inputs_v, [dense_str], True, shape)

        np.testing.assert_array_equal(result_sparse, result_dense)

    def test_scalar_all_y(self, test_data):
        ts, ys, _, inputs, _ = test_data
        t, y, p = _casadi_symbols()

        expr = casadi.sum1(y) * p[0] + t
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ts_v, ys_v, _, inputs_v = _make_vecs(ts, ys, None, inputs)
        shape = [1, NT]

        result_sparse = idaklu.observe(ts_v, ys_v, inputs_v, [sparse_str], True, shape)
        result_dense = idaklu.observe(ts_v, ys_v, inputs_v, [dense_str], True, shape)

        np.testing.assert_array_equal(result_sparse, result_dense)

    def test_scalar_no_y(self, test_data):
        ts, ys, _, inputs, _ = test_data
        t, y, p = _casadi_symbols()

        expr = t * p[0] + p[1]
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ts_v, ys_v, _, inputs_v = _make_vecs(ts, ys, None, inputs)
        shape = [1, NT]

        result_sparse = idaklu.observe(ts_v, ys_v, inputs_v, [sparse_str], True, shape)
        result_dense = idaklu.observe(ts_v, ys_v, inputs_v, [dense_str], True, shape)

        np.testing.assert_array_equal(result_sparse, result_dense)

    def test_vector_sparse_y(self, test_data):
        ts, ys, _, inputs, _ = test_data
        t, y, p = _casadi_symbols()

        expr = casadi.vertcat(y[0] + p[0], y[3] * t, y[0] * y[3] + p[1])
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ts_v, ys_v, _, inputs_v = _make_vecs(ts, ys, None, inputs)
        shape = [3, NT]

        result_sparse = idaklu.observe(ts_v, ys_v, inputs_v, [sparse_str], True, shape)
        result_dense = idaklu.observe(ts_v, ys_v, inputs_v, [dense_str], True, shape)

        np.testing.assert_array_equal(result_sparse, result_dense)

    def test_c_contiguous_y(self, test_data):
        ts, ys, _, inputs, _ = test_data
        t, y, p = _casadi_symbols()

        expr = y[2] * p[0] + y[6]
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ys_c = np.ascontiguousarray(ys)
        ts_v, ys_v, _, inputs_v = _make_vecs(ts, ys_c, None, inputs)
        shape = [1, NT]

        result_sparse = idaklu.observe(ts_v, ys_v, inputs_v, [sparse_str], False, shape)
        result_dense = idaklu.observe(ts_v, ys_v, inputs_v, [dense_str], False, shape)

        np.testing.assert_array_equal(result_sparse, result_dense)


class TestObserveHermiteInterpSparsity:
    pytestmark = pytest.mark.unit

    def test_scalar_sparse_y(self, test_data):
        ts, ys, yps, inputs, t_interp = test_data
        t, y, p = _casadi_symbols()

        expr = y[1] * p[0] + y[5] * p[1] + t
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ts_v, ys_v, yps_v, inputs_v = _make_vecs(ts, ys, yps, inputs)
        shape = [1, N_INTERP]

        result_sparse = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [sparse_str], shape
        )
        result_dense = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [dense_str], shape
        )

        np.testing.assert_array_equal(result_sparse, result_dense)

    def test_scalar_all_y(self, test_data):
        ts, ys, yps, inputs, t_interp = test_data
        t, y, p = _casadi_symbols()

        expr = casadi.sum1(y) * p[0] + t
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ts_v, ys_v, yps_v, inputs_v = _make_vecs(ts, ys, yps, inputs)
        shape = [1, N_INTERP]

        result_sparse = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [sparse_str], shape
        )
        result_dense = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [dense_str], shape
        )

        np.testing.assert_array_equal(result_sparse, result_dense)

    def test_scalar_no_y(self, test_data):
        ts, ys, yps, inputs, t_interp = test_data
        t, y, p = _casadi_symbols()

        expr = t * p[0] + p[1]
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ts_v, ys_v, yps_v, inputs_v = _make_vecs(ts, ys, yps, inputs)
        shape = [1, N_INTERP]

        result_sparse = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [sparse_str], shape
        )
        result_dense = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [dense_str], shape
        )

        np.testing.assert_array_equal(result_sparse, result_dense)

    def test_vector_sparse_y(self, test_data):
        ts, ys, yps, inputs, t_interp = test_data
        t, y, p = _casadi_symbols()

        expr = casadi.vertcat(y[0] + p[0], y[3] * t, y[0] * y[3] + p[1])
        sparse_str, dense_str = _make_dense_and_sparse_pair("f", [t, y, p], expr)

        ts_v, ys_v, yps_v, inputs_v = _make_vecs(ts, ys, yps, inputs)
        shape = [3, N_INTERP]

        result_sparse = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [sparse_str], shape
        )
        result_dense = idaklu.observe_hermite_interp(
            t_interp, ts_v, ys_v, yps_v, inputs_v, [dense_str], shape
        )

        np.testing.assert_array_equal(result_sparse, result_dense)


def _sine_segment(t_start, t_end, n):
    """A segment whose first state is sin(t), with its exact derivative."""
    ts = np.linspace(t_start, t_end, n, dtype=np.float64)
    ys = np.zeros((NY, n), dtype=np.float64, order="F")
    yps = np.zeros((NY, n), dtype=np.float64, order="F")
    ys[0, :] = np.sin(ts)
    yps[0, :] = np.cos(ts)
    return ts, ys, yps


def _observe_segments(t_interp, segments):
    t, y, p = _casadi_symbols()
    function = casadi.Function("f", [t, y, p], [y[0]]).serialize()
    inputs = np.array([1.0, 0.0], dtype=np.float64)
    return idaklu.observe_hermite_interp(
        np.asarray(t_interp, dtype=np.float64),
        idaklu.VectorRealtypeNdArray([ts for ts, _, _ in segments]),
        idaklu.VectorRealtypeNdArray([ys for _, ys, _ in segments]),
        idaklu.VectorRealtypeNdArray([yps for _, _, yps in segments]),
        idaklu.VectorRealtypeNdArray([inputs] * len(segments)),
        [function] * len(segments),
        [1, len(t_interp)],
    ).reshape(-1)


class TestObserveSegments:
    pytestmark = pytest.mark.unit

    def test_a_time_between_segments_is_nan_and_later_times_are_unaffected(self):
        segments = [_sine_segment(0, 1, 21), _sine_segment(2, 3, 21)]

        with_gap = _observe_segments([1.5, 2.25, 2.5], segments)
        without_gap = _observe_segments([2.25, 2.5], segments)

        assert np.isnan(with_gap[0])
        np.testing.assert_array_equal(with_gap[1:], without_gap)
        np.testing.assert_allclose(without_gap, np.sin([2.25, 2.5]), atol=1e-6)

    def test_a_single_point_segment(self):
        segments = [_sine_segment(0, 1, 21), _sine_segment(1.5, 1.5, 1)]

        result = _observe_segments([0.5, 1.5, 1.6], segments)

        np.testing.assert_allclose(result[:2], np.sin([0.5, 1.5]), atol=1e-6)
        # Past its only point, extrapolate linearly from it
        np.testing.assert_allclose(result[2], np.sin(1.5) + 0.1 * np.cos(1.5))

    def test_unsorted_times_are_rejected(self):
        with pytest.raises(ValueError, match=r"sorted in non-decreasing order"):
            _observe_segments([0.5, 0.25], [_sine_segment(0, 1, 21)])

    def test_no_segment_with_data_is_rejected(self):
        empty = _sine_segment(0, 1, 0)
        with pytest.raises(ValueError, match=r"no time points"):
            _observe_segments([0.5], [empty])
