"""Unit tests for the prep-artifact binding API (CompiledFunction)."""

import collections
import concurrent.futures
import pickle
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

import pybamm
from pybamm.rust import ExprGraph


def _make_fn(name=None):
    """f(t, y, p) = [p_a * y0 * y1, sin(y0) * p_b]; 2 states, 2 inputs."""
    g = ExprGraph()
    pa = g.input_parameter("a")
    pb = g.input_parameter("b")
    y0 = g.state_vector(0, 1)
    y1 = g.state_vector(1, 2)
    r0 = g.mul(g.mul(pa, y0), y1)
    r1 = g.mul(g.sin(y0), pb)
    expr = g.concat([r0, r1])
    return g, g.compile(expr, name=name)


def _full_residual_symbol(built_model):
    """rhs concatenated with algebraic, matching the solver's residual."""
    if built_model.len_alg > 0:
        return pybamm.numpy_concatenation(
            built_model.concatenated_rhs,
            built_model.concatenated_algebraic,
        )
    return built_model.concatenated_rhs


class TestSignature:
    def test_properties(self):
        _, f = _make_fn(name="RHS")
        assert f.input_names == ("a", "b")
        assert f.vp_len == 2
        assert f.n_states == 2
        assert f.output_len == 2
        assert f.uses_y_dot is False
        assert f.name == "RHS"
        assert "RHS" in repr(f)

    def test_explicit_n_states_override(self):
        g = ExprGraph()
        y0 = g.state_vector(0, 1)
        f = g.compile(g.mul(y0, y0), n_states=3)
        assert f.n_states == 3

    def test_n_states_override_below_extent_rejected(self):
        g = ExprGraph()
        y = g.state_vector(0, 5)
        with pytest.raises(ValueError, match=r"n_states=2.*state extent 5"):
            g.compile(g.mul(y, y), n_states=2)


class TestCompilePrep:
    def test_compile_runs_simplify_pipeline(self):
        g = ExprGraph()
        y0 = g.state_vector(0, 1)
        base = g.mul(g.sin(y0), g.input_parameter("a"))
        decorated = g.add(g.mul(base, g.scalar(1.0)), g.scalar(0.0))
        assert g.compile(decorated).n_instructions == g.compile(base).n_instructions

    def test_signature_scans_pre_simplify_extent(self):
        # y1 - y1 cancels, but a length-2 y must stay valid at the call site.
        g = ExprGraph()
        y0 = g.state_vector(0, 1)
        y1 = g.state_vector(1, 2)
        f = g.compile(g.add(y0, g.sub(y1, y1)))
        assert f.n_states == 2
        f(0.0, np.array([1.0, 5.0]), np.array([]))

    def test_derivatives_follow_the_simplified_graph(self):
        # (y0 / y0) * y0 simplifies to y0; differentiated unsimplified, it is NaN at y0 = 0.
        g = ExprGraph()
        y0 = g.state_vector(0, 1)
        f = g.compile(g.mul(g.div(y0, y0), y0))
        y, p = np.zeros(1), np.zeros(0)
        np.testing.assert_array_equal(f(0.0, y, p), [0.0])
        np.testing.assert_array_equal(f.jvp(0.0, y, p, np.ones(1)), [1.0])
        np.testing.assert_array_equal(f.jacobian()(0.0, y, p).toarray(), [[1.0]])


class TestCall:
    def test_call_stacked(self):
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        p = np.array([3.0, 4.0])
        out = f(0.0, y, p)
        np.testing.assert_allclose(out, [3.0 * 0.5 * 2.0, np.sin(0.5) * 4.0])

    def test_call_dict(self):
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        out = f(0.0, y, {"a": 3.0, "b": 4.0})
        np.testing.assert_allclose(out, [3.0, np.sin(0.5) * 4.0], rtol=1e-12)

    def test_call_dict_with_1element_array_values(self):
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        out = f(0.0, y, {"a": np.array([3.0]), "b": np.array([4.0])})
        np.testing.assert_allclose(out, [3.0, np.sin(0.5) * 4.0], rtol=1e-12)

    def test_call_dict_with_nonfloat64_values(self):
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        expected = [3.0, np.sin(0.5) * 4.0]
        cases = [
            {"a": np.array([3], dtype=np.int64), "b": np.array([4], dtype=np.int64)},
            {"a": np.array([3.0], dtype=np.float32), "b": np.float32(4.0)},
            {"a": np.int64(3), "b": 4},
        ]
        for inputs in cases:
            np.testing.assert_allclose(f(0.0, y, inputs), expected, rtol=1e-6)

    def test_call_any_mapping(self):
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        expected = f(0.0, y, {"a": 3.0, "b": 4.0})
        inputs = types.MappingProxyType({"a": 3.0, "b": 4.0})
        np.testing.assert_array_equal(f(0.0, y, inputs), expected)
        np.testing.assert_array_equal(f.pack(inputs), [3.0, 4.0])

    @pytest.mark.parametrize(
        "value",
        [
            [1.0, 3.0],
            (1.0, 3.0),
            [[1.0], [3.0]],
            np.array([[1.0], [3.0]]),
            np.array([1.0, 9.0, 3.0])[::2],
            np.array([1, 3], dtype=np.int32),
            [1, 3],
            [np.float32(1.0), np.float32(3.0)],
        ],
        ids=[
            "list",
            "tuple",
            "nested-list",
            "column",
            "strided",
            "int-array",
            "int-list",
            "float32-list",
        ],
    )
    def test_vector_input_accepts_any_real_container(self, value):
        g = ExprGraph()
        b = g.input_parameter("b", 2)
        f = g.compile(g.mul(b, g.state_vector(0, 2)))
        y = np.array([2.0, 3.0])
        np.testing.assert_array_equal(f(0.0, y, {"b": value}), [2.0, 9.0])

    def test_dict_and_stacked_bitwise_equal(self):
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        a = f(0.0, y, np.array([3.0, 4.0]))
        b = f(0.0, y, {"a": 3.0, "b": 4.0})
        np.testing.assert_array_equal(a, b)

    def test_pack(self):
        _, f = _make_fn()
        np.testing.assert_array_equal(f.pack({"a": 3.0, "b": 4.0}), [3.0, 4.0])

    def test_eval_into(self):
        _, f = _make_fn()
        out = np.zeros(2)
        f.eval_into(0.0, np.array([0.5, 2.0]), np.array([3.0, 4.0]), out)
        np.testing.assert_allclose(out, [3.0, np.sin(0.5) * 4.0])

    def test_shared_function_is_reusable(self):
        _, f = _make_fn()
        y1, y2 = np.array([0.5, 2.0]), np.array([1.0, 1.0])
        p = np.array([1.0, 1.0])
        a1 = f(0.0, y1, p)
        b1 = f(0.0, y2, p)
        a2 = f(0.0, y1, p)
        np.testing.assert_array_equal(a1, a2)
        assert not np.array_equal(a1, b1)


class TestValidation:
    def test_wrong_y_length(self):
        _, f = _make_fn(name="RHS")
        with pytest.raises(ValueError, match=r"RHS.*expected y of length 2.*got 3"):
            f(0.0, np.zeros(3), np.zeros(2))

    def test_wrong_p_length(self):
        _, f = _make_fn(name="RHS")
        with pytest.raises(
            ValueError, match=r"expected 2 input values.*2 parameters.*got 1"
        ):
            f(0.0, np.zeros(2), np.zeros(1))

    def test_wrong_y_dot_length(self):
        g = ExprGraph()
        f = g.compile(g.state_vector_dot(0, 2), name="resid")
        with pytest.raises(
            ValueError, match=r"resid.*expected y_dot of length 2.*got 1"
        ):
            f(0.0, np.zeros(2), np.array([]), y_dot=np.zeros(1))

    def test_dict_missing_key(self):
        _, f = _make_fn()
        with pytest.raises(ValueError, match=r"missing input 'b'"):
            f(0.0, np.zeros(2), {"a": 1.0})

    def test_dict_unknown_key(self):
        _, f = _make_fn()
        with pytest.raises(ValueError, match=r"unknown input 'c'"):
            f(0.0, np.zeros(2), {"a": 1.0, "b": 2.0, "c": 3.0})

    def test_dict_rejects_multielement_array(self):
        _, f = _make_fn()
        with pytest.raises(ValueError, match=r"input 'a'.*scalar or length-1.*got 2"):
            f(0.0, np.zeros(2), {"a": np.array([1.0, 2.0]), "b": 3.0})

    @pytest.mark.parametrize(
        ("value", "received"),
        [
            (None, r"got NoneType"),
            ("3.0", r"got str"),
            (np.array([1j]), r"got a 1-D array of dtype complex128"),
            (np.complex128(1 + 2j), r"got complex128"),
            (1 + 2j, r"got complex"),
            ([1.0, None], r"got list"),
        ],
    )
    def test_dict_rejects_a_non_real_value_naming_it(self, value, received):
        _, f = _make_fn()
        # Converting a complex to float only warns, so the value must be refused unconverted.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
            with pytest.raises(
                ValueError, match=rf"input 'a' must be a real number.*{received}"
            ):
                f(0.0, np.zeros(2), {"a": value, "b": 3.0})

    def test_a_defaultdict_does_not_fill_in_a_missing_input(self):
        _, f = _make_fn()
        inputs = collections.defaultdict(float, a=1.0)
        with pytest.raises(ValueError, match=r"missing input 'b'"):
            f(0.0, np.zeros(2), inputs)
        assert dict(inputs) == {"a": 1.0}

    @pytest.mark.parametrize(
        ("p", "received"),
        [
            ([3.0, 4.0], r"got list"),
            (np.array([3, 4]), r"got a 1-D array of dtype int64"),
            (np.zeros((2, 1)), r"got a 2-D array of dtype float64"),
        ],
    )
    def test_p_of_another_type_is_named(self, p, received):
        _, f = _make_fn()
        with pytest.raises(ValueError, match=rf"p must be .* mapping, {received}"):
            f(0.0, np.zeros(2), p)

    def test_eval_into_rejects_aliased_out(self):
        _, f = _make_fn()
        arr = np.array([0.5, 2.0])
        with pytest.raises(ValueError, match=r"already borrowed"):
            f.eval_into(0.0, arr, np.array([3.0, 4.0]), arr)

    def test_eval_into_rejects_the_wrong_out_length(self):
        _, f = _make_fn(name="RHS")
        with pytest.raises(ValueError, match=r"RHS: expected out of length 2, got 3"):
            f.eval_into(0.0, np.zeros(2), np.zeros(2), np.zeros(3))

    def test_eval_into_rejects_a_strided_out(self):
        _, f = _make_fn(name="RHS")
        with pytest.raises(ValueError, match=r"RHS: out must be contiguous"):
            f.eval_into(0.0, np.zeros(2), np.zeros(2), np.zeros(4)[::2])


def _strided(values):
    """``values`` as a view with stride 2, which is not contiguous."""
    padded = np.zeros(2 * len(values))
    padded[::2] = values
    view = padded[::2]
    assert not view.flags["C_CONTIGUOUS"]
    return view


class TestStridedArguments:
    def test_strided_arguments_match_contiguous_ones(self):
        _, f = _make_fn()
        y, p = np.array([0.5, 2.0]), np.array([3.0, 4.0])
        vy, vp = np.array([1.0, -0.5]), np.array([0.25, -1.0])
        np.testing.assert_array_equal(f(0.0, _strided(y), _strided(p)), f(0.0, y, p))
        np.testing.assert_array_equal(
            f.jvp(0.0, _strided(y), _strided(p), _strided(vy), vp=_strided(vp)),
            f.jvp(0.0, y, p, vy, vp=vp),
        )
        np.testing.assert_array_equal(
            f.jacobian()(0.0, _strided(y), p).toarray(),
            f.jacobian()(0.0, y, p).toarray(),
        )
        out = np.zeros(2)
        f.eval_into(0.0, _strided(y), _strided(p), out)
        np.testing.assert_array_equal(out, f(0.0, y, p))

    def test_strided_y_dot_matches_contiguous(self):
        g = ExprGraph()
        f = g.compile(g.sub(g.state_vector_dot(0, 2), g.state_vector(0, 2)))
        y, y_dot = np.array([1.0, 2.0]), np.array([5.0, 7.0])
        np.testing.assert_array_equal(
            f(0.0, y, np.zeros(0), y_dot=_strided(y_dot)),
            f(0.0, y, np.zeros(0), y_dot=y_dot),
        )


class TestJvp:
    def test_jvp_wrt_y_matches_fd(self):
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        p = np.array([3.0, 4.0])
        vy = np.array([1.0, -0.5])
        eps = 1e-7
        fd = (f(0.0, y + eps * vy, p) - f(0.0, y - eps * vy, p)) / (2 * eps)
        np.testing.assert_allclose(f.jvp(0.0, y, p, vy), fd, rtol=1e-6)

    def test_jvp_with_vp_sums_contributions(self):
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        p = np.array([3.0, 4.0])
        vy = np.array([1.0, -0.5])
        vp = np.array([0.25, -1.0])
        eps = 1e-7
        fd = (
            f(0.0, y + eps * vy, p + eps * vp) - f(0.0, y - eps * vy, p - eps * vp)
        ) / (2 * eps)
        np.testing.assert_allclose(f.jvp(0.0, y, p, vy, vp=vp), fd, rtol=1e-6)

    def test_concurrent_first_calls_agree(self):
        # Every thread races to derive the tangent tapes, which must be built once.
        _, reference = _make_fn()
        y, p = np.array([0.5, 2.0]), np.array([3.0, 4.0])
        vy, vp = np.array([1.0, -0.5]), np.array([0.25, -1.0])
        expected = reference.jvp(0.0, y, p, vy, vp=vp)
        _, f = _make_fn()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: f.jvp(0.0, y, p, vy, vp=vp), range(32)))
        for result in results:
            np.testing.assert_array_equal(result, expected)

    def test_jvp_rejects_y_dot_expressions(self):
        # LoadStateVectorDot slices y_dot unconditionally: would panic across PyO3.
        g = ExprGraph()
        f = g.compile(g.state_vector_dot(0, 1), name="resid")
        with pytest.raises(ValueError, match=r"resid.*y_dot"):
            f.jvp(0.0, np.zeros(1), np.zeros(0), np.zeros(1))

    def test_jvp_vp_broadcast_param_keeps_full_width(self):
        # df/dp @ [1] must be [1, 1, 1], not collapsed to length 1 as [1, 0, 0].
        g = ExprGraph()
        sv = g.state_vector(0, 3)
        f = g.compile(g.add(sv, g.input_parameter("a")), name="vecp")
        assert f.output_len == 3
        y = np.array([10.0, 20.0, 30.0])
        out = f.jvp(0.0, y, np.array([5.0]), np.zeros(3), vp=np.array([1.0]))
        np.testing.assert_array_equal(out, [1.0, 1.0, 1.0])

    def test_jvp_zero_dfdy_returns_full_width_zeros(self):
        g = ExprGraph()
        cv = g.array(np.array([1.0, 2.0, 3.0]))
        f = g.compile(g.add(cv, g.input_parameter("a")), name="zeroy", n_states=2)
        assert f.output_len == 3
        out = f.jvp(0.0, np.zeros(2), np.array([5.0]), np.zeros(2))
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    def test_jvp_rejects_wrong_vy_length(self):
        _, f = _make_fn(name="rhs")
        y, p = np.array([0.5, 2.0]), np.array([3.0, 4.0])
        with pytest.raises(ValueError, match=r"rhs.*vy.*length"):
            f.jvp(0.0, y, p, np.zeros(3))

    def test_jvp_rejects_wrong_vp_length(self):
        _, f = _make_fn(name="rhs")
        y, p, vy = np.array([0.5, 2.0]), np.array([3.0, 4.0]), np.zeros(2)
        with pytest.raises(ValueError, match=r"rhs: expected vp of length 2.*got 3"):
            f.jvp(0.0, y, p, vy, vp=np.zeros(3))


class TestJacobian:
    def test_wrt_y_matches_fd(self):
        _, f = _make_fn()
        J = f.jacobian()
        y = np.array([0.5, 2.0])
        p = np.array([3.0, 4.0])
        mat = J(0.0, y, p)
        assert isinstance(mat, scipy.sparse.csc_matrix)
        assert mat.shape == (2, 2)
        eps = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1.0
            fd = (f(0.0, y + eps * e, p) - f(0.0, y - eps * e, p)) / (2 * eps)
            np.testing.assert_allclose(mat.toarray()[:, j], fd, rtol=1e-6, atol=1e-10)

    def test_wrt_p_matches_fd(self):
        _, f = _make_fn()
        Jp = f.jacobian(wrt="p")
        y = np.array([0.5, 2.0])
        p = np.array([3.0, 4.0])
        mat = Jp(0.0, y, p)
        assert mat.shape == (2, 2)
        eps = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1.0
            fd = (f(0.0, y, p + eps * e) - f(0.0, y, p - eps * e)) / (2 * eps)
            np.testing.assert_allclose(mat.toarray()[:, j], fd, rtol=1e-6, atol=1e-10)

    def test_cached_per_wrt(self):
        _, f = _make_fn()
        assert f.jacobian() is f.jacobian()
        assert f.jacobian(wrt="p") is f.jacobian(wrt="p")
        assert f.jacobian() is not f.jacobian(wrt="p")

    def test_introspection(self):
        _, f = _make_fn()
        J = f.jacobian()
        assert J.shape == (2, 2)
        assert J.wrt == "y"
        assert J.nnz >= 3
        assert J.n_colors >= 1
        indptr, _indices = J.sparsity()
        assert len(indptr) == 3  # CSC: n_cols + 1

    def test_rectangular_partial_group(self):
        g = ExprGraph()
        y0 = g.state_vector(0, 1)
        y1 = g.state_vector(1, 2)
        f = g.compile(g.mul(y0, y1), n_states=2)
        assert f.jacobian().shape == (1, 2)

    def test_invalid_wrt(self):
        _, f = _make_fn()
        with pytest.raises(ValueError, match="wrt"):
            f.jacobian(wrt="t")

    def test_rejects_y_dot_expressions(self):
        g = ExprGraph()
        f = g.compile(g.state_vector_dot(0, 1), name="resid")
        with pytest.raises(ValueError, match=r"resid.*y_dot"):
            f.jacobian()

    def test_csc_matrices_own_their_arrays(self):
        _, f = _make_fn()
        J = f.jacobian()
        y, p = np.array([0.5, 2.0]), np.array([0.0, 4.0])
        m1, m2 = J(0.0, y, p), J(0.0, y, p)
        assert m1.indices.dtype == m1.indptr.dtype == np.int32
        for arrays in ((m1.data, m2.data), (m1.indices, m2.indices)):
            assert not np.shares_memory(*arrays)
        # p_a = 0 makes the y-column entries of the first row explicit zeros.
        m1.eliminate_zeros()
        assert m1.nnz < m2.nnz
        indptr, indices = J.sparsity()
        np.testing.assert_array_equal(indptr, m2.indptr)
        np.testing.assert_array_equal(indices, m2.indices)

    def test_spme_vaas_dense_row_coloring(self):
        # The voltage row is dense; without the split, coloring needs ~65 colors.
        model = pybamm.lithium_ion.SPMe(options={"voltage as a state": "true"})
        model.events = []
        sim = pybamm.Simulation(model)
        sim.build()
        built = sim.built_model
        g = ExprGraph()
        fn = g.compile(
            _full_residual_symbol(built).to_rust(g, {}),
            name="spme",
            n_states=built.len_rhs_and_alg,
        )
        jac = fn.jacobian()
        assert jac.n_dense_rows == 1
        assert jac.n_colors <= 8

        y0 = built.concatenated_initial_conditions.evaluate().flatten()
        J = jac(0.0, y0, np.array([])).toarray()
        h = 1e-7
        for col in [0, built.len_rhs_and_alg // 2, built.len_rhs_and_alg - 1]:
            yp, ym = y0.copy(), y0.copy()
            yp[col] += h
            ym[col] -= h
            fd = (fn(0.0, yp, np.array([])) - fn(0.0, ym, np.array([]))) / (2 * h)
            np.testing.assert_allclose(
                J[:, col], np.asarray(fd).flatten(), rtol=1e-4, atol=1e-6
            )


class TestEvalTrajectory:
    def _setup(self, n_t=50):
        _, f = _make_fn()
        ts = np.linspace(0.0, 1.0, n_t)
        Y = np.vstack([np.linspace(0.1, 1.0, n_t), np.linspace(2.0, 3.0, n_t)])
        p = np.array([3.0, 4.0])
        return f, ts, Y, p

    # 65 and 200 points span more than one 64-lane tile, the last one partial.
    @pytest.mark.parametrize("n_t", [50, 65, 200])
    def test_matches_per_column_loop(self, n_t):
        f, ts, Y, p = self._setup(n_t)
        out = f.eval_trajectory(ts, Y, p)
        assert out.shape == (2, len(ts))
        for j, t in enumerate(ts):
            np.testing.assert_array_equal(out[:, j], f(t, Y[:, j].copy(), p))

    def test_accepts_c_and_f_order(self):
        f, ts, Y, p = self._setup()
        a = f.eval_trajectory(ts, np.ascontiguousarray(Y), p)
        b = f.eval_trajectory(ts, np.asfortranarray(Y), p)
        np.testing.assert_array_equal(a, b)

    def test_f_order_matches_per_column_loop(self):
        f, ts, Y, p = self._setup()
        Yf = np.asfortranarray(Y)
        out = f.eval_trajectory(ts, Yf, p)
        for j, t in enumerate(ts):
            np.testing.assert_array_equal(out[:, j], f(t, Yf[:, j].copy(), p))

    def test_reversed_view_matches_per_column_loop(self):
        f, ts, Y, p = self._setup()
        Yr = np.asfortranarray(Y)[:, ::-1]
        out = f.eval_trajectory(ts, Yr, p)
        for j, t in enumerate(ts):
            np.testing.assert_array_equal(out[:, j], f(t, Yr[:, j].copy(), p))

    def test_shape_validation(self):
        f, ts, Y, p = self._setup()
        with pytest.raises(ValueError, match=r"y_traj.shape\[0\] must equal n_states"):
            f.eval_trajectory(ts, Y[:1, :], p)
        with pytest.raises(ValueError, match=r"len\(ts\) \(49\) must equal y_traj"):
            f.eval_trajectory(ts[:-1], Y, p)

    def test_rejects_y_dot_before_reading_its_arguments(self):
        g = ExprGraph()
        f = g.compile(g.state_vector_dot(0, 1), name="resid")
        with pytest.raises(ValueError, match=r"resid: eval_trajectory.*y_dot"):
            f.eval_trajectory(np.zeros(3), np.zeros((5, 2)), "not p")


class TestNamedOutputs:
    def _setup(self):
        """Two outputs sharing the subexpression sin(y0)*a."""
        g = ExprGraph()
        a = g.input_parameter("a")
        y0 = g.state_vector(0, 1)
        shared = g.mul(g.sin(y0), a)
        out1 = g.add(shared, g.scalar(1.0))
        out2 = g.mul(shared, g.scalar(2.0))
        fn = g.compile_outputs({"plus_one": out1, "doubled": out2})
        f1 = g.compile(out1)
        f2 = g.compile(out2)
        return g, fn, f1, f2

    def test_outputs_match_siso_compiles(self):
        _, fn, f1, f2 = self._setup()
        y = np.array([0.7])
        p = np.array([2.0])
        r1, r2 = fn.split(fn(0.0, y, p))
        np.testing.assert_array_equal(r1, f1(0.0, y, p))
        np.testing.assert_array_equal(r2, f2(0.0, y, p))

    def test_names_and_lens(self):
        _, fn, _, _ = self._setup()
        assert fn.output_names == ("plus_one", "doubled")
        assert fn.output_lens == (1, 1)

    def test_a_single_expression_carries_no_layout(self):
        g = ExprGraph()
        f = g.compile(g.state_vector(0, 1), name="solo")
        assert f.output_names is None
        assert f.output_lens is None
        with pytest.raises(ValueError, match=r"solo: split needs named outputs"):
            f.split(np.zeros(1))

    def test_split_rejects_the_wrong_row_count(self):
        _, fn, _, _ = self._setup()
        with pytest.raises(ValueError, match=r"split expects 2 row\(s\), got 5"):
            fn.split(np.zeros(5))

    def test_split_rejects_a_value_without_a_shape(self):
        _, fn, _, _ = self._setup()
        with pytest.raises(
            ValueError, match=r"split expects an array with a leading axis"
        ):
            fn.split([1.0, 2.0])

    def test_split_returns_views_not_copies(self):
        _, fn, _, _ = self._setup()
        n_t = 4
        ts = np.linspace(0.0, 1.0, n_t)
        Y = np.linspace(0.1, 1.0, n_t).reshape(1, n_t)
        rows = fn.eval_trajectory(ts, Y, np.array([2.0]))
        assert all(part.base is rows for part in fn.split(rows))

    def test_signature_surface(self):
        _, fn, _, _ = self._setup()
        assert fn.input_names == ("a",)
        assert fn.vp_len == 1
        assert fn.output_len == 2
        assert fn.uses_y_dot is False
        np.testing.assert_array_equal(fn.pack({"a": 2.0}), [2.0])

    def test_cse_observable_in_tape_length(self):
        _, fn, f1, f2 = self._setup()
        assert fn.n_instructions < f1.n_instructions + f2.n_instructions

    def test_cse_dedupes_structurally_identical_outputs(self):
        g = ExprGraph()
        a = g.input_parameter("a")
        s1 = g.mul(g.sin(g.state_vector(0, 1)), a)
        s2 = g.mul(g.sin(g.state_vector(0, 1)), a)
        twin = g.compile_outputs(
            {"x": g.add(s1, g.scalar(1.0)), "y": g.mul(s2, g.scalar(2.0))}
        )
        shared = g.compile_outputs(
            {"x": g.add(s1, g.scalar(1.0)), "y": g.mul(s1, g.scalar(2.0))}
        )
        assert twin.n_instructions == shared.n_instructions

    def test_eval_trajectory(self):
        _, fn, f1, f2 = self._setup()
        n_t = 20
        ts = np.linspace(0.0, 1.0, n_t)
        Y = np.linspace(0.1, 1.0, n_t).reshape(1, n_t)
        p = np.array([2.0])
        r1, r2 = fn.split(fn.eval_trajectory(ts, Y, p))
        assert r1.shape == (1, n_t) and r2.shape == (1, n_t)
        for j, t in enumerate(ts):
            np.testing.assert_array_equal(r1[:, j], f1(t, Y[:, j].copy(), p))
            np.testing.assert_array_equal(r2[:, j], f2(t, Y[:, j].copy(), p))

    def test_eval_trajectory_hermite_matches_per_output_hermite(self):
        _, fn, f1, f2 = self._setup()
        n_knots = 10
        ts = np.linspace(0.0, 1.0, n_knots)
        Y = np.sin(ts).reshape(1, n_knots)
        YP = np.cos(ts).reshape(1, n_knots)
        p = np.array([2.0])
        tq = np.linspace(0.0, 1.0, 31)
        r1, r2 = fn.split(fn.eval_trajectory_hermite(tq, ts, Y, YP, p))
        np.testing.assert_allclose(
            r1, f1.eval_trajectory_hermite(tq, ts, Y, YP, p), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            r2, f2.eval_trajectory_hermite(tq, ts, Y, YP, p), rtol=1e-12, atol=1e-12
        )

    def test_named_outputs_carry_the_whole_function_api(self):
        _, fn, _, _ = self._setup()
        t, y, p = 0.0, np.array([0.7]), np.array([2.0])
        expected = fn(t, y, p)

        assert fn.jacobian("y")(t, y, p).shape == (2, 1)
        out = np.zeros(fn.output_len)
        fn.eval_into(t, y, p, out)
        np.testing.assert_array_equal(out, expected)

        restored = pickle.loads(pickle.dumps(fn))
        assert restored.output_names == fn.output_names
        assert restored.output_lens == fn.output_lens
        np.testing.assert_array_equal(restored(t, y, p), expected)

    def test_rejects_y_dot_at_call(self):
        g = ExprGraph()
        fn = g.compile_outputs({"r": g.state_vector_dot(0, 1)}, name="resid")
        with pytest.raises(ValueError, match=r"resid.*y_dot"):
            fn(0.0, np.zeros(1), np.array([]))

    def test_n_states_override_below_extent_rejected(self):
        g = ExprGraph()
        e = g.mul(g.state_vector(0, 3), g.state_vector(0, 3))
        before = g.n_nodes
        with pytest.raises(ValueError, match=r"n_states=1.*state extent 3"):
            g.compile_outputs({"out": e}, n_states=1)
        assert g.n_nodes == before, "a refused call must leave the graph as it was"

    def test_requires_at_least_one_output(self):
        g = ExprGraph()
        with pytest.raises(ValueError, match=r"at least one output"):
            g.compile_outputs({})

    def test_the_concat_root_is_kept_in_the_graph(self):
        g = ExprGraph()
        e = g.mul(g.sin(g.state_vector(0, 1)), g.input_parameter("a"))
        before = g.n_nodes
        fn = g.compile_outputs({"out": e})
        assert g.n_nodes == before + 1
        assert fn.jacobian("y")(0.0, np.array([0.5]), np.array([2.0])).shape == (1, 1)

    def test_repeat_compiles_reuse_the_concat_root(self):
        g = ExprGraph()
        y = g.state_vector(0, 1)
        outputs = {"sin": g.sin(y), "cos": g.cos(y)}
        first = g.compile_outputs(outputs)
        after_first = g.n_nodes
        second = g.compile_outputs(outputs)
        assert g.n_nodes == after_first
        np.testing.assert_array_equal(
            second(0.0, np.array([0.3]), np.zeros(0)),
            first(0.0, np.array([0.3]), np.zeros(0)),
        )


class TestPickle:
    def _setup(self):
        """`early` compiles before `later` registers a second input on the same graph."""
        g = ExprGraph()
        y = g.state_vector(0, 2)
        early = g.compile(g.mul(y, g.input_parameter("x", 2)), name="early")
        later = g.compile(g.add(y, g.input_parameter("p", 2)), name="later")
        return early, later

    def test_keeps_the_inputs_registered_at_compile(self):
        early, _ = self._setup()
        restored = pickle.loads(pickle.dumps(early))
        assert restored.input_names == early.input_names == ("x",)
        # Width-2 inputs keep the index extent (1) apart from the packed width (2).
        assert restored.vp_len == early.vp_len == 1

        y, x = np.array([2.0, 5.0]), np.array([3.0, 4.0])
        expected = early(0.0, y, {"x": x})
        np.testing.assert_array_equal(restored(0.0, y, {"x": x}), expected)
        np.testing.assert_array_equal(restored(0.0, y, x), expected)

    def test_functions_sharing_a_graph_keep_their_own_inputs(self):
        early, later = self._setup()
        restored_early, restored_later = pickle.loads(pickle.dumps((early, later)))
        assert restored_early.input_names == ("x",)
        assert restored_later.input_names == ("x", "p")


class TestInstructionCounts:
    def test_n_instructions_excludes_conditional_branch_blocks(self):
        # `n_instructions` is the common tape plus one dispatch, like CasADi's Switch.
        g = ExprGraph()
        y0 = g.state_vector(0, 1)
        selector = g.input_parameter("step")
        small = g.sin(y0)
        big = y0
        for _ in range(12):
            big = g.exp(big)
        f = g.compile(g.conditional(selector, [small, big]))

        assert f.branch_block_lens == (1, 12)
        assert f.n_instructions_total == f.n_instructions + 13
        # common = y0 + selector + dispatch + conditional
        assert f.n_instructions == 4
        assert f.n_dispatches == 1

    def test_n_instructions_flat_in_branch_count(self):
        def reported(n_branches):
            g = ExprGraph()
            y0 = g.state_vector(0, 1)
            selector = g.input_parameter("step")
            branches = []
            for i in range(n_branches):
                node = y0
                for _ in range(3 + i):
                    node = g.exp(node)
                branches.append(node)
            return g.compile(g.conditional(selector, branches)).n_instructions

        assert reported(2) == reported(6)


class TestCompiledModelBundle:
    def _setup(self, n_inputs=1, sens_param_indices=()):
        from pybamm.rust import CompiledModel

        g = ExprGraph()
        a = g.input_parameter("a")
        y0 = g.state_vector(0, 1)
        y1 = g.state_vector(1, 2)
        rhs = g.concat([g.mul(a, y1), g.neg(y0)])
        out = g.mul(y0, y1)
        event = g.sub(y0, g.scalar(10.0))
        mass = np.array([1.0, 1.0])
        indptr = np.array([0, 1, 2], dtype=np.int64)
        indices = np.array([0, 1], dtype=np.int64)
        model = CompiledModel.from_expr(
            g,
            rhs,
            mass,
            indptr,
            indices,
            n_inputs=n_inputs,
            sens_param_indices=list(sens_param_indices),
            output_exprs=[out],
            event_exprs=[event],
        )
        return g, model

    def _check_views(self, model, p):
        y = np.array([1.0, 2.0])
        np.testing.assert_array_equal(model.rhs(0.0, y, p), [6.0, -1.0])
        np.testing.assert_array_equal(model.outputs[0](0.0, y, p), [2.0])
        np.testing.assert_array_equal(model.events[0](0.0, y, p), [-9.0])
        np.testing.assert_allclose(
            model.jacobian(0.0, y, p).toarray(), [[0.0, 3.0], [-1.0, 0.0]]
        )

    def test_views_keep_the_inputs_the_model_was_compiled_with(self):
        # A later lowering, such as Solution.observe, can register inputs on the graph.
        g, model = self._setup()
        g.input_parameter("late")
        self._check_views(model, np.array([3.0]))
        assert model.rhs.input_names == ("a",)

    def test_a_late_input_survives_a_pickle_round_trip(self):
        g, model = self._setup(sens_param_indices=[0])
        g.input_parameter("late")
        restored = pickle.loads(pickle.dumps(model))
        assert restored.n_inputs == 1
        assert restored.n_sens_params == 1
        assert restored.__reduce__()[1][1:] == model.__reduce__()[1][1:]
        self._check_views(restored, np.array([3.0]))

    def test_rebuild_refuses_a_node_count_that_misses_a_root_or_the_graph(self):
        g, model = self._setup()
        rebuild, args = model.__reduce__()
        before, after = args[:9], args[10:]
        with pytest.raises(ValueError, match=r"root \d+ is not among the 1 nodes"):
            rebuild(*before, 1, *after)
        too_many = g.n_nodes + 1
        with pytest.raises(
            ValueError, match=rf"n_nodes={too_many} exceeds the graph's {g.n_nodes}"
        ):
            rebuild(*before, too_many, *after)

    def test_views_of_a_widened_model_take_its_width(self):
        _, model = self._setup(n_inputs=3)
        wide = np.array([3.0, 0.0, 0.0])
        self._check_views(model, wide)
        restored = pickle.loads(pickle.dumps(model.rhs))
        np.testing.assert_array_equal(
            restored(0.0, np.array([1.0, 2.0]), wide), [6.0, -1.0]
        )
        with pytest.raises(ValueError, match=r"rhs: expected 3 input values"):
            model.rhs(0.0, np.array([1.0, 2.0]), np.array([3.0]))

    def test_rhs_view(self):
        _, model = self._setup()
        f = model.rhs
        y = np.array([1.0, 2.0])
        p = np.array([3.0])
        np.testing.assert_array_equal(f(0.0, y, p), [6.0, -1.0])
        assert f.n_states == 2

    def test_jacobian_view_is_pure_df_dy(self):
        _, model = self._setup()
        J = model.jacobian
        mat = J(0.0, np.array([1.0, 2.0]), np.array([3.0]))
        # df/dy = [[0, a], [-1, 0]] — no mass, no cj
        np.testing.assert_allclose(mat.toarray(), [[0.0, 3.0], [-1.0, 0.0]])

    def test_outputs_and_events_views(self):
        _, model = self._setup()
        (out_fn,) = model.outputs
        (event_fn,) = model.events
        y = np.array([1.0, 2.0])
        p = np.array([3.0])
        np.testing.assert_array_equal(out_fn(0.0, y, p), [2.0])
        np.testing.assert_array_equal(event_fn(0.0, y, p), [-9.0])

    def test_algebraic_views_none_for_ode(self):
        _, model = self._setup()
        assert model.algebraic_residual is None
        assert model.algebraic_jacobian is None

    def test_algebraic_jacobian_view_is_the_compiled_artifact(self):
        model = pybamm.lithium_ion.DFN()
        model.convert_to_format = "rust"
        sim = pybamm.Simulation(model)
        sim.build()
        sim.solver.set_up(sim.built_model, inputs=[{}])
        rust_model = sim.solver._setup["rust_model"]
        assert rust_model.has_algebraic

        view = rust_model.algebraic_jacobian
        n_algebraic = rust_model.n_algebraic
        assert view.shape == (n_algebraic, n_algebraic)
        assert view.nnz == rust_model.algebraic_jacobian_nnz
        rows, _ = rust_model.algebraic_jacobian_sparsity_pattern()
        assert sorted(np.asarray(rows).tolist()) == sorted(
            np.asarray(view.sparsity()[1]).tolist()
        )

    def test_views_are_cached_objects(self):
        _, model = self._setup()
        assert model.rhs is model.rhs
        assert model.jacobian is model.jacobian
        assert model.outputs[0] is model.outputs[0]
        assert model.events[0] is model.events[0]

    def test_kept_solver_surface(self):
        _, model = self._setup()
        assert model.n_states == 2
        assert model.nnz >= 2
        assert model.n_colors >= 1
        assert repr(model.evaluator()).startswith(
            '<capsule object "pybamm.rust.evaluator"'
        )
        model.csc_sparsity_pattern()
        stats = model.jacobian_stats()
        assert stats["n_dense_rows"] == 0
        assert stats["dense_row_entries"] == 0
        assert stats["dense_row_tape_instructions"] == 0


class TestJvpTrajectory:
    def _setup(self):
        _, f = _make_fn()
        n_t = 20
        ts = np.linspace(0.0, 1.0, n_t)
        Y = np.vstack([np.linspace(0.1, 1.0, n_t), np.linspace(2.0, 3.0, n_t)])
        VY = np.vstack([np.linspace(0.5, 1.5, n_t), np.linspace(-1.0, 1.0, n_t)])
        p = np.array([3.0, 4.0])
        return f, ts, Y, VY, p

    def test_matches_per_column_jvp(self):
        f, ts, Y, VY, p = self._setup()
        out = f.jvp_trajectory(ts, Y, p, VY)
        assert out.shape == (2, len(ts))
        for j, t in enumerate(ts):
            expected = f.jvp(t, Y[:, j].copy(), p, VY[:, j].copy())
            np.testing.assert_array_equal(out[:, j], expected)

    def test_with_vp_matches_per_column_jvp(self):
        f, ts, Y, VY, p = self._setup()
        vp = np.array([0.25, -1.0])
        out = f.jvp_trajectory(ts, Y, p, VY, vp=vp)
        assert out.shape == (2, len(ts))
        for j, t in enumerate(ts):
            expected = f.jvp(t, Y[:, j].copy(), p, VY[:, j].copy(), vp=vp)
            np.testing.assert_array_equal(out[:, j], expected)

    def test_matches_numpy_jacobian_oracle(self):
        f, ts, Y, VY, p = self._setup()
        vp = np.array([0.25, -1.0])
        Jy = f.jacobian("y")
        Jp = f.jacobian("p")
        out = f.jvp_trajectory(ts, Y, p, VY, vp=vp)
        for j, t in enumerate(ts):
            dvar_dy = Jy(t, Y[:, j].copy(), p).toarray()
            dvar_dp = Jp(t, Y[:, j].copy(), p).toarray()
            oracle = dvar_dy @ VY[:, j] + dvar_dp @ vp
            np.testing.assert_allclose(out[:, j], oracle, rtol=1e-12, atol=1e-12)

    def test_preserves_zero_derivative_width(self):
        g = ExprGraph()
        cv = g.array(np.array([1.0, 2.0, 3.0]))
        f = g.compile(g.add(cv, g.input_parameter("a")), name="zeroy", n_states=2)
        assert f.output_len == 3
        n_t = 5
        ts = np.linspace(0.0, 1.0, n_t)
        Y = np.zeros((2, n_t))
        VY = np.ones((2, n_t))
        out = f.jvp_trajectory(ts, Y, np.array([5.0]), VY)
        assert out.shape == (3, n_t)
        np.testing.assert_array_equal(out, np.zeros((3, n_t)))

    def test_shape_validation(self):
        f, ts, Y, VY, p = self._setup()
        with pytest.raises(ValueError, match=r"y_traj.shape\[0\]"):
            f.jvp_trajectory(ts, Y[:1, :], p, VY)
        with pytest.raises(ValueError, match=r"len\(ts\) \(19\) must equal y_traj"):
            f.jvp_trajectory(ts[:-1], Y, p, VY)
        with pytest.raises(ValueError, match=r"vy_traj"):
            f.jvp_trajectory(ts, Y, p, VY[:1, :])
        with pytest.raises(ValueError, match=r"vy_traj"):
            f.jvp_trajectory(ts, Y, p, VY[:, :-1])

    def test_rejects_wrong_vp_length(self):
        f, ts, Y, VY, p = self._setup()
        with pytest.raises(ValueError, match=r"expected vp of length 2.*got 1"):
            f.jvp_trajectory(ts, Y, p, VY, vp=np.zeros(1))

    def test_accepts_c_and_f_order(self):
        f, ts, Y, VY, p = self._setup()
        vp = np.array([0.25, -1.0])
        a = f.jvp_trajectory(
            ts, np.ascontiguousarray(Y), p, np.ascontiguousarray(VY), vp=vp
        )
        b = f.jvp_trajectory(ts, np.asfortranarray(Y), p, np.asfortranarray(VY), vp=vp)
        np.testing.assert_array_equal(a, b)

    def test_reversed_view_matches_per_column_jvp(self):
        f, ts, Y, VY, p = self._setup()
        Yr = np.asfortranarray(Y)[:, ::-1]
        VYr = np.asfortranarray(VY)[:, ::-1]
        out = f.jvp_trajectory(ts, Yr, p, VYr)
        for j, t in enumerate(ts):
            expected = f.jvp(t, Yr[:, j].copy(), p, VYr[:, j].copy())
            np.testing.assert_array_equal(out[:, j], expected)

    def test_rejects_y_dot_expressions(self):
        g = ExprGraph()
        f = g.compile(g.state_vector_dot(0, 1), name="resid")
        with pytest.raises(ValueError, match=r"jvp_trajectory.*y_dot"):
            f.jvp_trajectory(
                np.zeros(1), np.zeros((1, 1)), np.zeros(0), np.zeros((1, 1))
            )


class TestEvalTrajectoryHermite:
    def _setup(self):
        _, f = _make_fn()
        n_knots = 12
        ts = np.linspace(0.0, 1.0, n_knots)
        Y = np.vstack([np.sin(ts), np.cos(ts)])
        YP = np.vstack([np.cos(ts), -np.sin(ts)])
        p = np.array([3.0, 4.0])
        return f, ts, Y, YP, p

    def test_reduces_to_eval_on_knots(self):
        f, ts, Y, YP, p = self._setup()
        out = f.eval_trajectory_hermite(ts, ts, Y, YP, p)
        ref = f.eval_trajectory(ts, Y, p)
        assert out.shape == (2, len(ts))
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("fault", ["reversed", "repeated", "nan", "inf"])
    def test_rejects_knots_not_finite_and_strictly_increasing(self, fault):
        f, ts, Y, YP, p = self._setup()
        knots = ts[::-1].copy() if fault == "reversed" else ts.copy()
        if fault == "repeated":
            knots[4] = knots[3]
        elif fault == "nan":
            knots[3] = np.nan
        elif fault == "inf":
            knots[-1] = np.inf
        with pytest.raises(
            ValueError, match=r"knot times must be finite and strictly increasing"
        ):
            f.eval_trajectory_hermite(ts, knots, Y, YP, p)

    def test_linear_state_exact(self):
        _, f = _make_fn()
        ts = np.linspace(0.0, 1.0, 5)
        Y = np.vstack([2.0 * ts + 0.5, -3.0 * ts + 1.0])
        YP = np.vstack([np.full_like(ts, 2.0), np.full_like(ts, -3.0)])
        p = np.array([3.0, 4.0])
        tq = np.linspace(0.0, 1.0, 37)
        out = f.eval_trajectory_hermite(tq, ts, Y, YP, p)
        Yq = np.ascontiguousarray(np.vstack([2.0 * tq + 0.5, -3.0 * tq + 1.0]))
        ref = f.eval_trajectory(tq, Yq, p)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    # 65 and 200 queries span more than one 64-lane tile, the last one partial.
    @pytest.mark.parametrize("n_query", [41, 65, 200])
    def test_matches_scipy_cubic_hermite_oracle(self, n_query):
        from scipy.interpolate import CubicHermiteSpline

        f, ts, Y, YP, p = self._setup()
        tq = np.linspace(0.0, 1.0, n_query)
        spline = CubicHermiteSpline(ts, Y.T, YP.T)  # (n_query, n_states)
        Yq = np.ascontiguousarray(spline(tq).T)
        ref = f.eval_trajectory(tq, Yq, p)
        out = f.eval_trajectory_hermite(tq, ts, Y, YP, p)
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)

    def test_shape_validation(self):
        f, ts, Y, YP, p = self._setup()
        with pytest.raises(ValueError, match=r"yps shape must equal ys shape"):
            f.eval_trajectory_hermite(ts, ts, Y, YP[:1, :], p)
        with pytest.raises(ValueError, match=r"len\(ts\) \(11\) must equal ys"):
            f.eval_trajectory_hermite(ts, ts[:-1], Y, YP, p)


class TestConcurrency:
    def test_shared_function_and_jacobian_under_threads(self):
        _, f = _make_fn()
        J = f.jacobian()
        rng = np.random.default_rng(42)
        cases = [
            (float(t), rng.uniform(0.1, 2.0, 2), rng.uniform(0.5, 5.0, 2))
            for t in range(64)
        ]
        serial = [(f(t, y, p), J(t, y, p).toarray()) for t, y, p in cases]

        def work(case):
            t, y, p = case
            return f(t, y, p), J(t, y, p).toarray()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            threaded = list(pool.map(work, cases * 4))

        for i, (fv, jv) in enumerate(threaded):
            ref_f, ref_j = serial[i % len(cases)]
            np.testing.assert_array_equal(
                fv, ref_f, err_msg="threaded eval must be bitwise-identical"
            )
            np.testing.assert_array_equal(
                jv, ref_j, err_msg="threaded jacobian must be bitwise-identical"
            )


class TestExtensionModuleLocation:
    def test_extension_ships_inside_pybamm_package(self):
        from pybamm.rust import _core

        assert Path(_core.__file__).parent == Path(pybamm.__file__).parent / "rust"

    def test_public_names_resolve_through_pybamm_rust(self):
        import pybamm.rust

        for name in (
            "CompiledFunction",
            "CompiledModel",
            "ExprGraph",
        ):
            assert hasattr(pybamm.rust, name), name

    def test_every_core_class_is_re_exported(self):
        import pybamm.rust
        from pybamm.rust import _core

        registered = {
            name
            for name, obj in vars(_core).items()
            if isinstance(obj, type) and not name.startswith("_")
        }
        assert registered <= set(pybamm.rust.__all__), sorted(
            registered - set(pybamm.rust.__all__)
        )

    def test_pyclass_module_is_the_public_path(self):
        import pybamm.rust

        for name in pybamm.rust.__all__:
            obj = getattr(pybamm.rust, name)
            if isinstance(obj, type):
                assert obj.__module__ == "pybamm.rust", name

    def test_no_public_name_carries_the_py_prefix(self):
        # `Py*` is PyO3's namespace; a leaked Rust-side prefix reads as a stutter.
        import pybamm.rust

        leaked = [name for name in pybamm.rust.__all__ if name.startswith("Py")]
        assert not leaked, leaked

    def test_only_one_core_binary_is_installed(self):
        # A stale version-specific .so would shadow the abi3 one at import time.
        import pybamm.rust

        artifacts = sorted(
            p.name
            for p in Path(pybamm.rust.__file__).parent.glob("_core*")
            if p.suffix in (".so", ".pyd")
        )
        assert len(artifacts) == 1, artifacts
