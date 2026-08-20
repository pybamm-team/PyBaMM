"""Unit tests for the prep-artifact binding API (CompiledFunction)."""

import importlib
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
    """rhs concatenated with algebraic, matching the solver's residual.

    Inlined from `benchmarks/rust_observability/runners.py` — `benchmarks/`
    is not importable under this test tree's pytest rootdir.
    """
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
        assert f.n_inputs == 2
        assert f.n_states == 2
        assert f.output_len == 2
        assert f.uses_y_dot is False
        assert f.name == "RHS"
        assert "RHS" in repr(f)

    def test_explicit_n_states_override(self):
        g = ExprGraph()
        y0 = g.state_vector(0, 1)  # touches only state 0 of a 3-state system
        f = g.compile(g.mul(y0, y0), n_states=3)
        assert f.n_states == 3

    def test_n_states_override_below_extent_rejected(self):
        # an override below the scanned extent would admit a too-short y
        # and panic inside the tape
        g = ExprGraph()
        y = g.state_vector(0, 5)
        with pytest.raises(ValueError, match=r"n_states=2.*state extent 5"):
            g.compile(g.mul(y, y), n_states=2)


class TestCompilePrep:
    def test_compile_runs_simplify_pipeline(self):
        # compile prep includes simplification — decorating an
        # expression with *1 and +0 must not lengthen the tape
        g = ExprGraph()
        y0 = g.state_vector(0, 1)
        base = g.mul(g.sin(y0), g.input_parameter("a"))
        decorated = g.add(g.mul(base, g.scalar(1.0)), g.scalar(0.0))
        assert g.compile(decorated).n_instructions == g.compile(base).n_instructions

    def test_signature_scans_pre_simplify_extent(self):
        # y1 - y1 cancels, but the user-declared contract still spans both
        # states: a length-2 y must stay valid at the call site
        g = ExprGraph()
        y0 = g.state_vector(0, 1)
        y1 = g.state_vector(1, 2)
        f = g.compile(g.add(y0, g.sub(y1, y1)))
        assert f.n_states == 2
        f(0.0, np.array([1.0, 5.0]), np.array([]))  # must not raise


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
        # PyBaMM stores solve inputs as 1-element arrays, e.g. {'a': array([3.0])}.
        # Native observation passes solution.all_inputs[i] straight to the binding,
        # so pack must accept length-1 arrays as scalars, not only Python floats.
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        out = f(0.0, y, {"a": np.array([3.0]), "b": np.array([4.0])})
        np.testing.assert_allclose(out, [3.0, np.sin(0.5) * 4.0], rtol=1e-12)

    def test_call_dict_with_nonfloat64_values(self):
        # int64/float32 arrays and numpy scalars must coerce to f64 like CasADi.
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

    def test_dict_and_stacked_bitwise_equal(self):
        _, f = _make_fn()
        y = np.array([0.5, 2.0])
        a = f(0.0, y, np.array([3.0, 4.0]))
        b = f(0.0, y, {"a": 3.0, "b": 4.0})
        assert (a == b).all()

    def test_pack(self):
        _, f = _make_fn()
        np.testing.assert_array_equal(f.pack({"a": 3.0, "b": 4.0}), [3.0, 4.0])

    def test_eval_into(self):
        _, f = _make_fn()
        out = np.zeros(2)
        f.eval_into(0.0, np.array([0.5, 2.0]), np.array([3.0, 4.0]), out)
        np.testing.assert_allclose(out, [3.0, np.sin(0.5) * 4.0])

    def test_shared_function_is_reusable(self):
        # &self eval: two interleaved calls on one object, no state bleed
        _, f = _make_fn()
        y1, y2 = np.array([0.5, 2.0]), np.array([1.0, 1.0])
        p = np.array([1.0, 1.0])
        a1 = f(0.0, y1, p)
        b1 = f(0.0, y2, p)
        a2 = f(0.0, y1, p)
        assert (a1 == a2).all() and not (a1 == b1).all()


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
        # a short y_dot must raise, not panic inside the tape
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
        # a length>1 array is an error, not a silent take-first.
        _, f = _make_fn()
        with pytest.raises(ValueError, match=r"input 'a'.*scalar or length-1"):
            f(0.0, np.zeros(2), {"a": np.array([1.0, 2.0]), "b": 3.0})

    def test_eval_into_rejects_aliased_out(self):
        # out aliasing y would be UB without the borrow guard
        _, f = _make_fn()
        arr = np.array([0.5, 2.0])
        with pytest.raises(BaseException, match=r"AlreadyBorrowed"):
            f.eval_into(0.0, arr, np.array([3.0, 4.0]), arr)

    def test_non_contiguous_y_is_a_clear_type_error(self):
        # A strided view must fail at conversion with a TypeError, never
        # reach the slice unwrap inside the FFI.
        _, f = _make_fn()
        y_strided = np.arange(4.0).reshape(2, 2)[:, 0]
        assert not y_strided.flags["C_CONTIGUOUS"]
        with pytest.raises(TypeError, match=r"contiguous"):
            f(0.0, y_strided, np.zeros(2))


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

    def test_jvp_tangent_tape_is_cached(self):
        import time

        _, f = _make_fn()
        y, p, vy = np.array([0.5, 2.0]), np.array([3.0, 4.0]), np.array([1.0, 0.0])
        t0 = time.perf_counter()
        f.jvp(0.0, y, p, vy)
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(100):
            f.jvp(0.0, y, p, vy)
        per_call = (time.perf_counter() - t0) / 100
        assert per_call < first, "derivation must be one-time prep, not per-call"

    def test_jvp_rejects_y_dot_expressions(self):
        # LoadStateVectorDot slices y_dot unconditionally, so without this guard
        # the call would panic across PyO3.
        g = ExprGraph()
        f = g.compile(g.state_vector_dot(0, 1), name="resid")
        with pytest.raises(ValueError, match=r"resid.*y_dot"):
            f.jvp(0.0, np.zeros(1), np.zeros(0), np.zeros(1))

    def test_jvp_vp_broadcast_param_keeps_full_width(self):
        # f = y[0:3] + p0, so df/dp @ [1] must contribute [1, 1, 1]: the tangent_p
        # tape must not collapse to length 1 and truncate it to [1, 0, 0].
        g = ExprGraph()
        sv = g.state_vector(0, 3)
        f = g.compile(g.add(sv, g.input_parameter("a")), name="vecp")
        assert f.output_len == 3
        y = np.array([10.0, 20.0, 30.0])
        out = f.jvp(0.0, y, np.array([5.0]), np.zeros(3), vp=np.array([1.0]))
        np.testing.assert_array_equal(out, [1.0, 1.0, 1.0])

    def test_jvp_zero_dfdy_returns_full_width_zeros(self):
        # Regression: f = const_vector([1,2,3]) + p0; df/dy ≡ 0.
        # jvp wrt y must return length-3 zeros, not a collapsed length-1 [0.].
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
        with pytest.raises(ValueError, match=r"rhs"):
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
        assert J.nnz >= 3  # (0,0),(0,1),(1,0) at minimum
        assert J.n_colors >= 1
        indptr, _indices = J.sparsity()
        assert len(indptr) == 3  # CSC: n_cols + 1

    def test_rectangular_partial_group(self):
        g = ExprGraph()
        y0 = g.state_vector(0, 1)
        y1 = g.state_vector(1, 2)
        f = g.compile(g.mul(y0, y1), n_states=2)  # 1 output row, 2 states
        assert f.jacobian().shape == (1, 2)

    def test_invalid_wrt(self):
        _, f = _make_fn()
        with pytest.raises(ValueError, match="wrt"):
            f.jacobian(wrt="t")

    def test_rejects_y_dot_expressions(self):
        # see TestJvp.test_jvp_rejects_y_dot_expressions — same panic guard
        g = ExprGraph()
        f = g.compile(g.state_vector_dot(0, 1), name="resid")
        with pytest.raises(ValueError, match=r"resid.*y_dot"):
            f.jacobian()

    def test_csc_buffers_shared_and_readonly(self):
        # Only the data array is allocated per call: the cached int32 index arrays
        # are shared with scipy without a cast-copy, and are read-only.
        _, f = _make_fn()
        J = f.jacobian()
        y, p = np.array([0.5, 2.0]), np.array([3.0, 4.0])
        m1, m2 = J(0.0, y, p), J(0.0, y, p)
        assert m1.indices.dtype == np.int32  # scipy's native index dtype
        assert np.shares_memory(m1.indices, m2.indices)
        assert np.shares_memory(m1.indptr, m2.indptr)
        assert not m1.indices.flags.writeable
        assert not m1.indptr.flags.writeable
        with pytest.raises(ValueError):
            m1.indices[0] = m1.indices[0]
        assert m1.data.flags.writeable  # only the pattern is frozen

    def test_spme_vaas_dense_row_coloring(self):
        # SPMe with voltage-as-a-state has one ~65-nnz voltage row (default
        # grid); without the dense-row split, column coloring needs 65 colors.
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
        assert jac.n_colors <= 8  # 65 before the split; sparse remainder ~5

        # numeric parity: split assembly vs finite differences on a few columns
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
    def _setup(self):
        _, f = _make_fn()
        n_t = 50
        ts = np.linspace(0.0, 1.0, n_t)
        Y = np.vstack([np.linspace(0.1, 1.0, n_t), np.linspace(2.0, 3.0, n_t)])
        p = np.array([3.0, 4.0])
        return f, ts, Y, p

    def test_matches_per_column_loop(self):
        f, ts, Y, p = self._setup()
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
        # pins the zero-copy borrow path against ground truth
        f, ts, Y, p = self._setup()
        Yf = np.asfortranarray(Y)
        out = f.eval_trajectory(ts, Yf, p)
        for j, t in enumerate(ts):
            np.testing.assert_array_equal(out[:, j], f(t, Yf[:, j].copy(), p))

    def test_reversed_view_matches_per_column_loop(self):
        # negative-stride views must gather, not borrow the raw buffer
        f, ts, Y, p = self._setup()
        Yr = np.asfortranarray(Y)[:, ::-1]
        out = f.eval_trajectory(ts, Yr, p)
        for j, t in enumerate(ts):
            np.testing.assert_array_equal(out[:, j], f(t, Yr[:, j].copy(), p))

    def test_shape_validation(self):
        f, ts, Y, p = self._setup()
        with pytest.raises(ValueError, match=r"Y.shape\[0\]"):
            f.eval_trajectory(ts, Y[:1, :], p)
        with pytest.raises(ValueError, match=r"len\(ts\)"):
            f.eval_trajectory(ts[:-1], Y, p)


class TestGroup:
    def _setup(self):
        """Two outputs sharing the subexpression sin(y0)*a."""
        g = ExprGraph()
        a = g.input_parameter("a")
        y0 = g.state_vector(0, 1)
        shared = g.mul(g.sin(y0), a)  # shared NodeId
        out1 = g.add(shared, g.scalar(1.0))
        out2 = g.mul(shared, g.scalar(2.0))
        group = g.compile_group({"plus_one": out1, "doubled": out2})
        f1 = g.compile(out1)
        f2 = g.compile(out2)
        return g, group, f1, f2

    def test_outputs_match_siso_compiles(self):
        _, group, f1, f2 = self._setup()
        y = np.array([0.7])
        p = np.array([2.0])
        r1, r2 = group(0.0, y, p)
        assert (r1 == f1(0.0, y, p)).all()
        assert (r2 == f2(0.0, y, p)).all()

    def test_names_and_lens(self):
        _, group, _, _ = self._setup()
        assert group.names == ("plus_one", "doubled")
        assert group.output_lens == [1, 1]

    def test_signature_surface(self):
        # groups expose the same signature surface as CompiledFunction
        _, group, _, _ = self._setup()
        assert group.input_names == ("a",)
        assert group.n_inputs == 1
        assert group.output_len == 2
        assert group.uses_y_dot is False
        np.testing.assert_array_equal(group.pack({"a": 2.0}), [2.0])

    def test_cse_observable_in_tape_length(self):
        _, group, f1, f2 = self._setup()
        # shared work appears once in the group tape
        assert group.n_instructions < f1.n_instructions + f2.n_instructions

    def test_cse_dedupes_structurally_identical_outputs(self):
        # The same subexpression built twice (distinct NodeIds, identical structure)
        # collapses via cse() at compile, matching the shared-NodeId group's tape.
        g = ExprGraph()
        a = g.input_parameter("a")
        s1 = g.mul(g.sin(g.state_vector(0, 1)), a)
        s2 = g.mul(g.sin(g.state_vector(0, 1)), a)  # structural twin of s1
        twin = g.compile_group(
            {"x": g.add(s1, g.scalar(1.0)), "y": g.mul(s2, g.scalar(2.0))}
        )
        shared = g.compile_group(
            {"x": g.add(s1, g.scalar(1.0)), "y": g.mul(s1, g.scalar(2.0))}
        )
        assert twin.n_instructions == shared.n_instructions

    def test_eval_trajectory(self):
        _, group, f1, f2 = self._setup()
        n_t = 20
        ts = np.linspace(0.0, 1.0, n_t)
        Y = np.linspace(0.1, 1.0, n_t).reshape(1, n_t)
        p = np.array([2.0])
        r1, r2 = group.eval_trajectory(ts, Y, p)
        assert r1.shape == (1, n_t) and r2.shape == (1, n_t)
        for j, t in enumerate(ts):
            np.testing.assert_array_equal(r1[:, j], f1(t, Y[:, j].copy(), p))
            np.testing.assert_array_equal(r2[:, j], f2(t, Y[:, j].copy(), p))

    def test_group_rejects_y_dot_at_call(self):
        g = ExprGraph()
        grp = g.compile_group({"r": g.state_vector_dot(0, 1)}, name="resid")
        with pytest.raises(ValueError, match=r"resid.*y_dot"):
            grp(0.0, np.zeros(1), np.array([]))

    def test_group_n_states_override_below_extent_rejected(self):
        g = ExprGraph()
        e = g.mul(g.state_vector(0, 3), g.state_vector(0, 3))
        with pytest.raises(ValueError, match=r"n_states=1.*state extent 3"):
            g.compile_group({"out": e}, n_states=1)

    def test_compile_group_leaves_graph_unchanged(self):
        # compiling a group must not mutate the shared graph arena
        g = ExprGraph()
        e = g.mul(g.sin(g.state_vector(0, 1)), g.input_parameter("a"))
        before = g.n_nodes
        g.compile_group({"out": e})
        assert g.n_nodes == before

    def test_n_instructions_excludes_conditional_branch_blocks(self):
        # `n_instructions` reports the common tape plus one dispatch, matching
        # what casadi's Switch-lowered outer function reports.
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
        # And the reported count says how much of itself is control flow.
        assert f.n_dispatches == 1

    def test_n_instructions_flat_in_branch_count(self):
        # An extra branch must not grow the reported count.
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
    def _setup(self):
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
            n_inputs=1,
            output_exprs=[out],
            event_exprs=[event],
        )
        return g, model

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
        # The standalone dg/dy_alg view must lend the artifact the model already
        # compiled, not re-derive one: a second tangent transform could disagree
        # with the block the solver drives and nothing else would notice.
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
        # accessors hand back the SAME prepared artifacts on every access —
        # identity, not timing, is the contract
        _, model = self._setup()
        assert model.rhs is model.rhs
        assert model.jacobian is model.jacobian
        assert model.outputs[0] is model.outputs[0]
        assert model.events[0] is model.events[0]

    def test_removed_methods_are_gone(self):
        _, model = self._setup()
        for gone in [
            # gone because it derived the C++ Newton driver's write-through
            # address from a shared borrow; the FFI goes via evaluator_pool now
            "as_ptr",
            "eval_rhs",
            "eval_rhs_into",
            "jac_mul",
            "jac_mul_into",
            "assemble_jacobian",
            "eval_sens_all",
            "eval_sens",
            "eval_output",
            "output_len_at",
            "output_lens",
            "eval_events",
            "eval_events_into",
            "total_event_len",
        ]:
            assert not hasattr(model, gone), gone

    def test_kept_solver_surface(self):
        _, model = self._setup()
        assert model.n_states == 2
        assert model.nnz >= 2
        assert model.n_colors >= 1
        assert isinstance(model.evaluator_pool(1).as_ptr(0), int)
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
        # jvp_trajectory wrt y must equal the per-column jvp wrt y, bitwise:
        # same tangent tape, same inputs.
        f, ts, Y, VY, p = self._setup()
        out = f.jvp_trajectory(ts, Y, p, VY)
        assert out.shape == (2, len(ts))
        for j, t in enumerate(ts):
            expected = f.jvp(t, Y[:, j].copy(), p, VY[:, j].copy())
            np.testing.assert_array_equal(out[:, j], expected)

    def test_with_vp_matches_per_column_jvp(self):
        # With a parameter direction, each column equals jvp(.., vp=vp):
        # df/dy @ vy_j + df/dp @ vp.
        f, ts, Y, VY, p = self._setup()
        vp = np.array([0.25, -1.0])
        out = f.jvp_trajectory(ts, Y, p, VY, vp=vp)
        assert out.shape == (2, len(ts))
        for j, t in enumerate(ts):
            expected = f.jvp(t, Y[:, j].copy(), p, VY[:, j].copy(), vp=vp)
            np.testing.assert_array_equal(out[:, j], expected)

    def test_matches_numpy_jacobian_oracle(self):
        # Independent oracle: build the full dvar_dy / dvar_dp jacobians per column
        # and matmul dvar_dy @ vy + dvar_dp @ vp.
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
        # f = const_vector([1,2,3]) + p0, so df/dy is identically zero: the
        # trajectory jvp wrt y must stay full width, not collapse to one row.
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
        with pytest.raises(ValueError, match=r"Y.shape\[0\]"):
            f.jvp_trajectory(ts, Y[:1, :], p, VY)
        with pytest.raises(ValueError, match=r"len\(ts\)"):
            f.jvp_trajectory(ts[:-1], Y, p, VY)
        with pytest.raises(ValueError, match=r"vy_traj"):
            f.jvp_trajectory(ts, Y, p, VY[:1, :])
        # the same guard's time-dimension arm: vy_traj.shape[1] != n_t
        with pytest.raises(ValueError, match=r"vy_traj"):
            f.jvp_trajectory(ts, Y, p, VY[:, :-1])

    def test_accepts_c_and_f_order(self):
        # zero-copy F path and gather C path must agree, for both y and vy.
        f, ts, Y, VY, p = self._setup()
        vp = np.array([0.25, -1.0])
        a = f.jvp_trajectory(
            ts, np.ascontiguousarray(Y), p, np.ascontiguousarray(VY), vp=vp
        )
        b = f.jvp_trajectory(ts, np.asfortranarray(Y), p, np.asfortranarray(VY), vp=vp)
        np.testing.assert_array_equal(a, b)

    def test_reversed_view_matches_per_column_jvp(self):
        # negative-stride views must gather, not borrow the raw buffer
        f, ts, Y, VY, p = self._setup()
        Yr = np.asfortranarray(Y)[:, ::-1]
        VYr = np.asfortranarray(VY)[:, ::-1]
        out = f.jvp_trajectory(ts, Yr, p, VYr)
        for j, t in enumerate(ts):
            expected = f.jvp(t, Yr[:, j].copy(), p, VYr[:, j].copy())
            np.testing.assert_array_equal(out[:, j], expected)

    def test_rejects_y_dot_expressions(self):
        # same panic guard as TestJvp.test_jvp_rejects_y_dot_expressions:
        # the tangent tape slices an empty y_dot, so y_dot funcs are rejected
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
        # querying exactly at the knots reproduces eval_trajectory
        f, ts, Y, YP, p = self._setup()
        out = f.eval_trajectory_hermite(ts, ts, Y, YP, p)
        ref = f.eval_trajectory(ts, Y, p)
        assert out.shape == (2, len(ts))
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    def test_linear_state_exact(self):
        # Hermite reproduces a linear-in-t state exactly at off-grid points
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

    def test_matches_scipy_cubic_hermite_oracle(self):
        from scipy.interpolate import CubicHermiteSpline

        f, ts, Y, YP, p = self._setup()
        tq = np.linspace(0.0, 1.0, 41)
        spline = CubicHermiteSpline(ts, Y.T, YP.T)  # (n_query, n_states)
        Yq = np.ascontiguousarray(spline(tq).T)
        ref = f.eval_trajectory(tq, Yq, p)
        out = f.eval_trajectory_hermite(tq, ts, Y, YP, p)
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)

    def test_shape_validation(self):
        f, ts, Y, YP, p = self._setup()
        with pytest.raises(ValueError, match=r"yps shape"):
            f.eval_trajectory_hermite(ts, ts, Y, YP[:1, :], p)
        with pytest.raises(ValueError, match=r"len\(ts\)"):
            f.eval_trajectory_hermite(ts, ts[:-1], Y, YP, p)


class TestGroupEvalTrajectoryHermite:
    def test_matches_per_output_hermite(self):
        # group hermite == per-CompiledFunction hermite, per output
        _, group, f1, f2 = TestGroup()._setup()
        n_knots = 10
        ts = np.linspace(0.0, 1.0, n_knots)
        Y = np.sin(ts).reshape(1, n_knots)
        YP = np.cos(ts).reshape(1, n_knots)
        p = np.array([2.0])
        tq = np.linspace(0.0, 1.0, 31)
        r1, r2 = group.eval_trajectory_hermite(tq, ts, Y, YP, p)
        np.testing.assert_allclose(
            r1, f1.eval_trajectory_hermite(tq, ts, Y, YP, p), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            r2, f2.eval_trajectory_hermite(tq, ts, Y, YP, p), rtol=1e-12, atol=1e-12
        )


class TestConcurrency:
    def test_shared_function_and_jacobian_under_threads(self):
        import concurrent.futures

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
            assert (fv == ref_f).all(), "threaded eval must be bitwise-identical"
            assert (jv == ref_j).all(), "threaded jacobian must be bitwise-identical"


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
            "PreparedSolver",
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
        # `Py*` is PyO3's own namespace for its smart pointers and native type
        # bindings; a leaked Rust-side prefix reads as a stutter from Python.
        import pybamm.rust

        leaked = [name for name in pybamm.rust.__all__ if name.startswith("Py")]
        assert not leaked, leaked

    def test_only_one_core_binary_is_installed(self):
        # A stale version-specific .so would shadow the abi3 one at import time.
        # The checked-in _core.pyi stub sits alongside and is not a binary.
        import pybamm.rust

        artifacts = sorted(
            p.name
            for p in Path(pybamm.rust.__file__).parent.glob("_core*")
            if p.suffix in (".so", ".pyd")
        )
        assert len(artifacts) == 1, artifacts

    def test_pybamm_rust_distribution_is_gone(self):
        with pytest.raises(ImportError):
            importlib.import_module("pybamm_rust")
