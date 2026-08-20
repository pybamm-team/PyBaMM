"""CasADi parity for the prep-artifact API.

eval / jacobian(wrt="y"/"p") / jvp / eval_trajectory vs casadi.Function
equivalents, on a toy expression and a discretised SPM rhs.
"""

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")

import pybamm
from pybamm.rust import ExprGraph


def _toy_expr():
    """f(t, y, p): 2 states, 2 inputs, smooth, t-dependent."""
    y0 = pybamm.StateVector(slice(0, 1))
    y1 = pybamm.StateVector(slice(1, 2))
    a = pybamm.InputParameter("a")
    b = pybamm.InputParameter("b")
    expr = pybamm.NumpyConcatenation(
        a * y0 * y1 + pybamm.t, pybamm.sin(y0) * b + pybamm.exp(-y1)
    )
    return expr, 2, ["a", "b"]


def _casadi_artifacts(expr, n_states, input_names):
    """casadi Function quartet: f, df/dy, df/dp_stacked, jvp_y."""
    t = casadi.MX.sym("t")
    y = casadi.MX.sym("y", n_states)
    y_dot = casadi.MX.sym("y_dot", n_states)
    p_syms = {name: casadi.MX.sym(name) for name in input_names}
    casadi_symbols = {"t": t, "y": y, "y_dot": y_dot, "inputs": p_syms}
    cexpr = expr.to_casadi(t, y, y_dot, p_syms, casadi_symbols)
    p_stacked = casadi.vertcat(*p_syms.values())
    v = casadi.MX.sym("v", n_states)
    return (
        casadi.Function("f", [t, y, p_stacked], [cexpr]),
        casadi.Function("jy", [t, y, p_stacked], [casadi.jacobian(cexpr, y)]),
        casadi.Function("jp", [t, y, p_stacked], [casadi.jacobian(cexpr, p_stacked)]),
        casadi.Function("jvp", [t, y, p_stacked, v], [casadi.jtimes(cexpr, y, v)]),
    )


class TestToyParity:
    def setup_method(self):
        expr, self.n, names = _toy_expr()
        self.cf, self.cjy, self.cjp, self.cjvp = _casadi_artifacts(expr, self.n, names)
        g = ExprGraph()
        self.f = g.compile(expr.to_rust(g, {}), name="toy", n_states=self.n)
        # registration order == pre-order appearance == p_stacked order
        assert self.f.input_names == tuple(names)
        self.t = 0.7
        self.y = np.array([0.3, 1.2])
        self.p = np.array([2.5, -0.8])

    def test_eval(self):
        np.testing.assert_allclose(
            self.f(self.t, self.y, self.p),
            np.asarray(self.cf(self.t, self.y, self.p)).ravel(),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_jacobian_wrt_y(self):
        np.testing.assert_allclose(
            self.f.jacobian()(self.t, self.y, self.p).toarray(),
            np.asarray(self.cjy(self.t, self.y, self.p)),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_jacobian_wrt_p(self):
        # the spec-named check: jacobian(wrt="p") vs casadi.jacobian(expr, p_stacked)
        np.testing.assert_allclose(
            self.f.jacobian(wrt="p")(self.t, self.y, self.p).toarray(),
            np.asarray(self.cjp(self.t, self.y, self.p)),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_jvp(self):
        v = np.array([0.6, -1.1])
        np.testing.assert_allclose(
            self.f.jvp(self.t, self.y, self.p, v),
            np.asarray(self.cjvp(self.t, self.y, self.p, v)).ravel(),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_eval_trajectory(self):
        n_t = 40
        ts = np.linspace(0.0, 2.0, n_t)
        Y = np.vstack([np.linspace(0.1, 1.0, n_t), np.linspace(-0.5, 1.5, n_t)])
        out = self.f.eval_trajectory(ts, Y, self.p)
        ref = np.column_stack(
            [
                np.asarray(self.cf(tj, Y[:, j], self.p)).ravel()
                for j, tj in enumerate(ts)
            ]
        )
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)


class TestSPMParity:
    @pytest.fixture(scope="class")
    def spm(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.build()
        return sim.built_model

    @staticmethod
    def _y0(spm):
        # concatenated_initial_conditions is populated by build(); y0_list only
        # exists after a solver set_up runs.
        return np.asarray(
            spm.concatenated_initial_conditions.evaluate(), dtype=np.float64
        ).ravel()

    @staticmethod
    def _casadi_fn(expr, n):
        t = casadi.MX.sym("t")
        y = casadi.MX.sym("y", n)
        y_dot = casadi.MX.sym("y_dot", n)
        symbols = {"t": t, "y": y, "y_dot": y_dot, "inputs": {}}
        cexpr = expr.to_casadi(t, y, y_dot, {}, symbols)
        return (
            casadi.Function("f", [t, y], [cexpr]),
            casadi.Function("j", [t, y], [casadi.jacobian(cexpr, y)]),
        )

    def test_rhs_eval_and_jacobians(self, spm):
        rhs = spm.concatenated_rhs
        y0 = self._y0(spm)
        # rhs.size counts only the differential equations, but the solver evaluates
        # rhs against the full state vector, so size the input to y0.
        n = y0.shape[0]
        g = ExprGraph()
        f = g.compile(rhs.to_rust(g, {}), name="SPM_rhs", n_states=n)
        cf, cj = self._casadi_fn(rhs, n)
        p = np.array([])
        np.testing.assert_allclose(
            f(0.0, y0, p), np.asarray(cf(0.0, y0)).ravel(), rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            f.jacobian()(0.0, y0, p).toarray(),
            np.asarray(cj(0.0, y0)),
            rtol=1e-9,
            atol=1e-12,
        )

    def test_output_group_trajectory(self, spm):
        rust_symbols = {}
        g = ExprGraph()
        names = ["Voltage [V]", "Current [A]"]
        # get_processed_variable_or_event returns the discretised expression the
        # solver compiles; raw variables_and_events hold unlowerable nodes.
        exprs = {name: spm.get_processed_variable_or_event(name) for name in names}
        y0 = self._y0(spm)
        # outputs reference the algebraic voltage state, so size to the full DAE
        # state vector rather than the rhs (differential-only) size.
        n = y0.shape[0]
        group = g.compile_group(
            {name: e.to_rust(g, rust_symbols) for name, e in exprs.items()},
            n_states=n,
        )
        n_t = 25
        ts = np.linspace(0.0, 3600.0, n_t)
        Y = np.tile(y0[:, None], (1, n_t)) * np.linspace(1.0, 1.05, n_t)
        p = np.array([])
        results = group.eval_trajectory(ts, Y, p)
        for name, out in zip(names, results, strict=True):
            cf, _ = self._casadi_fn(exprs[name], n)
            ref = np.column_stack(
                [np.asarray(cf(tj, Y[:, j])).ravel() for j, tj in enumerate(ts)]
            )
            np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-12)
