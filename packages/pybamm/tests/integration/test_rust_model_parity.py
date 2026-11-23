"""CasADi parity of compiled expressions from full battery models."""

import casadi
import numpy as np
import pytest

import pybamm
from pybamm.rust import ExprGraph


class TestSPMParity:
    @pytest.fixture(scope="class")
    def spm(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.build()
        return sim.built_model

    @staticmethod
    def _y0(spm):
        # y0_list only exists after a solver set_up runs.
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
        # The solver evaluates rhs against the full state vector, not rhs.size.
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

    def test_named_outputs_trajectory(self, spm):
        rust_symbols = {}
        g = ExprGraph()
        names = ["Voltage [V]", "Current [A]"]
        # Raw variables_and_events hold nodes that can't be lowered.
        exprs = {name: spm.get_processed_variable_or_event(name) for name in names}
        y0 = self._y0(spm)
        # Outputs reference the algebraic voltage state.
        n = y0.shape[0]
        fn = g.compile_outputs(
            {name: e.to_rust(g, rust_symbols) for name, e in exprs.items()},
            n_states=n,
        )
        n_t = 25
        ts = np.linspace(0.0, 3600.0, n_t)
        Y = np.tile(y0[:, None], (1, n_t)) * np.linspace(1.0, 1.05, n_t)
        p = np.array([])
        results = fn.split(fn.eval_trajectory(ts, Y, p))
        for name, out in zip(fn.output_names, results, strict=True):
            cf, _ = self._casadi_fn(exprs[name], n)
            ref = np.column_stack(
                [np.asarray(cf(tj, Y[:, j])).ravel() for j, tj in enumerate(ts)]
            )
            np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-12)
