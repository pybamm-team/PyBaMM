"""Unit tests for StandaloneNewtonSolver C++ binding."""

from __future__ import annotations

import casadi
import numpy as np
import pytest
import pybammsolvers.idaklu as idaklu


def _build_solver(
    res_fn,
    jac_fn,
    *,
    atol=None,
    rtol=1e-10,
    step_tol=1e-10,
    max_iter=100,
    max_backtracks=5,
    eps_newt=0.33,
    use_sparse=False,
):
    n_vars = res_fn.nnz_out(0)
    if atol is None:
        atol = [rtol] * n_vars
    return idaklu.StandaloneNewtonSolver(
        idaklu.generate_function(res_fn.serialize()),
        idaklu.generate_function(jac_fn.serialize()),
        atol,
        rtol,
        step_tol,
        max_iter,
        max_backtracks,
        eps_newt,
        use_sparse,
    )


def _make_scalar_fns():
    """F(y) = y^2 - 4, J = 2y. Roots at y = +/-2."""
    t = casadi.SX.sym("t")
    y = casadi.SX.sym("y")
    p = casadi.SX.sym("p", 0)
    res = y**2 - 4
    jac_expr = casadi.jacobian(res, y)
    return (
        casadi.Function("res", [t, y, p], [res]),
        casadi.Function("newton_jac", [t, y, p], [jac_expr]),
    )


def _make_3x3_linear_fns():
    """Ax - b = 0 where A = [[2,1,0],[1,3,1],[0,1,2]], b = [4,10,6].
    Exact solution: x = [1, 2, 2]."""
    t = casadi.SX.sym("t")
    x = casadi.SX.sym("x", 3)
    p = casadi.SX.sym("p", 0)
    A = casadi.DM([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
    b = casadi.DM([4, 10, 6])
    res = A @ x - b
    jac_expr = casadi.jacobian(res, x)
    return (
        casadi.Function("res", [t, x, p], [res]),
        casadi.Function("newton_jac", [t, x, p], [jac_expr]),
    )


@pytest.fixture(scope="module")
def scalar_solver():
    res_fn, jac_fn = _make_scalar_fns()
    return _build_solver(res_fn, jac_fn)


@pytest.fixture(scope="module")
def linear_3x3_solver():
    res_fn, jac_fn = _make_3x3_linear_fns()
    return _build_solver(res_fn, jac_fn)


class TestStandaloneNewtonSolver:
    pytestmark = pytest.mark.unit

    def test_module_has_standalone_newton_solver(self):
        assert hasattr(idaklu, "StandaloneNewtonSolver")
        assert callable(idaklu.StandaloneNewtonSolver)

    def test_scalar_solve_basic(self, scalar_solver):
        success, y = scalar_solver.solve(
            0.0,
            np.array([1.5], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert success
        np.testing.assert_allclose(y, [2.0], atol=1e-10)

    def test_scalar_solve_negative_root(self, scalar_solver):
        success, y = scalar_solver.solve(
            0.0,
            np.array([-1.5], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert success
        np.testing.assert_allclose(y, [-2.0], atol=1e-10)

    def test_returns_success_flag(self, scalar_solver):
        result = scalar_solver.solve(
            0.0,
            np.array([1.5], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert result[0] is True

    def test_returns_numpy_array(self, scalar_solver):
        _, y = scalar_solver.solve(
            0.0,
            np.array([1.5], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert isinstance(y, np.ndarray)
        assert y.dtype == np.float64
        assert y.shape == (1,)

    def test_multivariable_linear_system(self, linear_3x3_solver):
        success, y = linear_3x3_solver.solve(
            0.0,
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert success
        np.testing.assert_allclose(y, [0.75, 2.5, 1.75], atol=1e-10)

    def test_multivariable_nonlinear_system(self):
        """x^2 + y = 5, x + y^2 = 5. From (1,1) -> (1.7912..., 1.7912...)."""
        t = casadi.SX.sym("t")
        xy = casadi.SX.sym("xy", 2)
        p = casadi.SX.sym("p", 0)
        res = casadi.vertcat(xy[0] ** 2 + xy[1] - 5, xy[0] + xy[1] ** 2 - 5)
        jac_expr = casadi.jacobian(res, xy)
        res_fn = casadi.Function("res", [t, xy, p], [res])
        jac_fn = casadi.Function("newton_jac", [t, xy, p], [jac_expr])
        solver = _build_solver(res_fn, jac_fn)

        success, y = solver.solve(
            0.0,
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert success
        expected = (-1 + np.sqrt(21)) / 2  # symmetric root ~1.7912878
        np.testing.assert_allclose(y, [expected, expected], atol=1e-8)

    def test_with_inputs(self):
        """F(y, p) = y - p = 0 with p=[7.0] -> y=7."""
        t = casadi.SX.sym("t")
        y = casadi.SX.sym("y")
        p = casadi.SX.sym("p")
        res = y - p
        jac_expr = casadi.jacobian(res, y)
        res_fn = casadi.Function("res", [t, y, p], [res])
        jac_fn = casadi.Function("newton_jac", [t, y, p], [jac_expr])
        solver = _build_solver(res_fn, jac_fn)

        success, y_sol = solver.solve(
            0.0,
            np.array([0.0], dtype=np.float64),
            np.array([7.0], dtype=np.float64),
        )
        assert success
        np.testing.assert_allclose(y_sol, [7.0], atol=1e-10)

    def test_with_time_parameter(self):
        """F(t, y) = y - 3*t. At t=2.0, y=6.0."""
        t = casadi.SX.sym("t")
        y = casadi.SX.sym("y")
        p = casadi.SX.sym("p", 0)
        res = y - 3 * t
        jac_expr = casadi.jacobian(res, y)
        res_fn = casadi.Function("res", [t, y, p], [res])
        jac_fn = casadi.Function("newton_jac", [t, y, p], [jac_expr])
        solver = _build_solver(res_fn, jac_fn)

        success, y_sol = solver.solve(
            2.0,
            np.array([0.0], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert success
        np.testing.assert_allclose(y_sol, [6.0], atol=1e-10)

    def test_convergence_failure(self):
        """F(y) = y^2 + 1 has no real root."""
        t = casadi.SX.sym("t")
        y = casadi.SX.sym("y")
        p = casadi.SX.sym("p", 0)
        res = y**2 + 1
        jac_expr = casadi.jacobian(res, y)
        res_fn = casadi.Function("res", [t, y, p], [res])
        jac_fn = casadi.Function("newton_jac", [t, y, p], [jac_expr])
        solver = _build_solver(res_fn, jac_fn, max_iter=50)

        success, _ = solver.solve(
            0.0,
            np.array([1.0], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert not success

    def test_sparse_vs_dense(self):
        res_fn, jac_fn = _make_3x3_linear_fns()
        solver_dense = _build_solver(res_fn, jac_fn, use_sparse=False)
        solver_sparse = _build_solver(res_fn, jac_fn, use_sparse=True)

        y0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        p = np.array([], dtype=np.float64)

        ok_d, y_d = solver_dense.solve(0.0, y0, p)
        ok_s, y_s = solver_sparse.solve(0.0, y0, p)

        assert ok_d and ok_s
        np.testing.assert_allclose(y_d, y_s, atol=1e-12)

    def test_n_vars_inferred(self):
        """Construct solvers for different sizes; n_vars is inferred automatically."""
        for n in [1, 3, 10]:
            t = casadi.SX.sym("t")
            y = casadi.SX.sym("y", n)
            p = casadi.SX.sym("p", 0)
            res = y
            jac_expr = casadi.jacobian(res, y)
            res_fn = casadi.Function("res", [t, y, p], [res])
            jac_fn = casadi.Function("newton_jac", [t, y, p], [jac_expr])
            solver = _build_solver(res_fn, jac_fn)

            y0 = np.ones(n, dtype=np.float64) * 5.0
            success, y_sol = solver.solve(0.0, y0, np.array([], dtype=np.float64))
            assert success
            np.testing.assert_allclose(y_sol, np.zeros(n), atol=1e-8)
            assert y_sol.shape == (n,)

    def test_solver_reuse(self, scalar_solver):
        """Call solve() multiple times on the same solver — tests buffer reuse."""
        empty_p = np.array([], dtype=np.float64)
        for _ in range(10):
            ok, y = scalar_solver.solve(
                0.0,
                np.array([1.5], dtype=np.float64),
                empty_p,
            )
            assert ok
            np.testing.assert_allclose(y, [2.0], atol=1e-10)

    def test_different_initial_guesses(self):
        res_fn, jac_fn = _make_scalar_fns()
        solver = _build_solver(res_fn, jac_fn)
        empty_p = np.array([], dtype=np.float64)

        for y0_val, expected in [
            (0.5, 2.0),
            (1.0, 2.0),
            (10.0, 2.0),
            (-0.5, -2.0),
            (-10.0, -2.0),
        ]:
            ok, y = solver.solve(0.0, np.array([y0_val], dtype=np.float64), empty_p)
            assert ok
            np.testing.assert_allclose(y, [expected], atol=1e-8)

    def test_atol_affects_convergence(self):
        res_fn, jac_fn = _make_scalar_fns()

        solver_loose = _build_solver(res_fn, jac_fn, rtol=1e-3, atol=[1e-3])
        solver_tight = _build_solver(res_fn, jac_fn, rtol=1e-12, atol=[1e-12])

        y0 = np.array([1.5], dtype=np.float64)
        empty_p = np.array([], dtype=np.float64)

        ok_l, y_l = solver_loose.solve(0.0, y0, empty_p)
        ok_t, y_t = solver_tight.solve(0.0, y0, empty_p)

        assert ok_l and ok_t
        err_l = abs(y_l[0] - 2.0)
        err_t = abs(y_t[0] - 2.0)
        assert err_t <= err_l

    # ───────── solve_batch tests ─────────

    def test_solve_batch_basic(self, scalar_solver):
        """Batch solve a time-independent scalar system across multiple t."""
        t_eval = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        y0 = np.array([1.5], dtype=np.float64)
        empty_p = np.array([], dtype=np.float64)

        ok, y_mat = scalar_solver.solve_batch(t_eval, y0, empty_p)
        assert ok
        assert y_mat.shape == (1, 3)
        np.testing.assert_allclose(y_mat, [[2.0, 2.0, 2.0]], atol=1e-10)

    def test_solve_batch_time_dependent(self):
        """F(t, y) = y - 3*t.  Solution: y(t) = 3*t."""
        t = casadi.SX.sym("t")
        y = casadi.SX.sym("y")
        p = casadi.SX.sym("p", 0)
        res = y - 3 * t
        jac_expr = casadi.jacobian(res, y)
        res_fn = casadi.Function("res", [t, y, p], [res])
        jac_fn = casadi.Function("newton_jac", [t, y, p], [jac_expr])
        solver = _build_solver(res_fn, jac_fn)

        t_eval = np.linspace(0, 2, 5)
        y0 = np.array([0.0], dtype=np.float64)
        empty_p = np.array([], dtype=np.float64)

        ok, y_mat = solver.solve_batch(t_eval, y0, empty_p)
        assert ok
        assert y_mat.shape == (1, 5)
        np.testing.assert_allclose(y_mat[0], 3 * t_eval, atol=1e-10)

    def test_solve_batch_failure_stops_early(self):
        """F(y) = y^2 + 1 has no real root -- batch must report failure."""
        t = casadi.SX.sym("t")
        y = casadi.SX.sym("y")
        p = casadi.SX.sym("p", 0)
        res = y**2 + 1
        jac_expr = casadi.jacobian(res, y)
        res_fn = casadi.Function("res", [t, y, p], [res])
        jac_fn = casadi.Function("newton_jac", [t, y, p], [jac_expr])
        solver = _build_solver(res_fn, jac_fn, max_iter=50)

        t_eval = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        y0 = np.array([1.0], dtype=np.float64)
        empty_p = np.array([], dtype=np.float64)

        ok, _ = solver.solve_batch(t_eval, y0, empty_p)
        assert not ok

    def test_solve_batch_single_time(self, scalar_solver):
        """Batch with one time point should match single solve."""
        t_eval = np.array([0.0], dtype=np.float64)
        y0 = np.array([1.5], dtype=np.float64)
        empty_p = np.array([], dtype=np.float64)

        ok_s, y_s = scalar_solver.solve(0.0, y0, empty_p)
        ok_b, y_b = scalar_solver.solve_batch(t_eval, y0, empty_p)

        assert ok_s and ok_b
        np.testing.assert_allclose(y_b.ravel(), y_s, atol=1e-12)

    def test_solve_batch_multivariable(self):
        """Batch solve a 2-variable time-dependent system."""
        t = casadi.SX.sym("t")
        xy = casadi.SX.sym("xy", 2)
        p = casadi.SX.sym("p", 0)
        res = casadi.vertcat(xy[0] - 2 * t, xy[1] - t**2)
        jac_expr = casadi.jacobian(res, xy)
        res_fn = casadi.Function("res", [t, xy, p], [res])
        jac_fn = casadi.Function("newton_jac", [t, xy, p], [jac_expr])
        solver = _build_solver(res_fn, jac_fn)

        t_eval = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y0 = np.array([0.0, 0.0], dtype=np.float64)
        empty_p = np.array([], dtype=np.float64)

        ok, y_mat = solver.solve_batch(t_eval, y0, empty_p)
        assert ok
        assert y_mat.shape == (2, 3)
        np.testing.assert_allclose(y_mat[0], 2 * t_eval, atol=1e-10)
        np.testing.assert_allclose(y_mat[1], t_eval**2, atol=1e-10)

    def test_solve_batch_output_is_fortran_order(self, scalar_solver):
        """Output array should be column-major (Fortran order)."""
        t_eval = np.array([0.0, 1.0], dtype=np.float64)
        y0 = np.array([1.5], dtype=np.float64)
        empty_p = np.array([], dtype=np.float64)

        _, y_mat = scalar_solver.solve_batch(t_eval, y0, empty_p)
        assert y_mat.flags["F_CONTIGUOUS"]
