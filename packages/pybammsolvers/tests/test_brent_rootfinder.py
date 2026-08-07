"""Unit tests for the "brent" CasADi rootfinder plugin."""

from __future__ import annotations

import shutil
import subprocess
import sys

import casadi
import numpy as np
import pytest
from scipy.optimize import brentq

import pybammsolvers.idaklu  # noqa: F401  registers the plugin on import

LO, HI = 1e-9, 1 - 1e-9


def non_monotone(x, p):
    """Several turning points inside the bracket, where a Newton iteration stalls."""
    return casadi.sin(12 * x) * 0.08 + (1.6 - x) ** 3 - p


def _solver(expr_fn, lo=LO, hi=HI, **opts):
    """A rootfinder over ``expr_fn(x, p) == 0``, and the function itself."""
    x, p = casadi.MX.sym("x"), casadi.MX.sym("p")
    f = casadi.Function("f", [x, p], [expr_fn(x, p)])
    return casadi.rootfinder("rf", "brent", f, {"lo": lo, "hi": hi, **opts}), f


class TestBrentPlugin:
    def test_importing_pybammsolvers_registers_the_plugin(self):
        assert casadi.has_rootfinder("brent")

    def test_solves_a_scalar_equation(self):
        rf, _ = _solver(lambda x, p: x * x - p * x - 6.0, lo=0.0, hi=10.0)
        assert float(rf(0.0, 1.0)) == pytest.approx(3.0, abs=1e-12)

    @pytest.mark.parametrize(
        "expr_fn", [lambda x, p: casadi.exp(x) + p * x - 2.0, non_monotone]
    )
    def test_matches_scipy_over_a_sweep(self, expr_fn):
        rf, f = _solver(expr_fn)
        worst_difference = worst_residual = 0.0
        for p in np.linspace(0.2, 1.8, 40):
            if float(f(LO, p)) * float(f(HI, p)) > 0:
                continue  # no bracket for this p, nothing to compare
            got = float(rf(0.0, p))
            want = brentq(lambda x, p=p: float(f(x, p)), LO, HI, xtol=2e-12)
            assert LO <= got <= HI
            worst_difference = max(worst_difference, abs(got - want))
            worst_residual = max(worst_residual, abs(float(f(got, p))))
        assert worst_difference < 1e-9
        assert worst_residual < 1e-12

    def test_bracket_can_be_a_graph_input(self):
        # x^2 - 6 has roots at +-sqrt(6); which one is found is set by the bracket,
        # passed as a live value rather than an option
        x, lo, hi = casadi.MX.sym("x"), casadi.MX.sym("lo"), casadi.MX.sym("hi")
        f = casadi.Function("f", [x, lo, hi], [x * x - 6.0])
        rf = casadi.rootfinder("rf", "brent", f, {"lo_index": 1, "hi_index": 2})
        assert float(rf(0.0, 0.0, 10.0)) == pytest.approx(np.sqrt(6), abs=1e-12)
        assert float(rf(0.0, -10.0, 0.0)) == pytest.approx(-np.sqrt(6), abs=1e-12)

    def test_derivatives_come_from_the_implicit_function_theorem(self):
        # x^2 - p0 x - p1 = 0 at (1, 6) has root 3; dx/dp = [x, 1] / (2x - p0)
        x, p = casadi.MX.sym("x"), casadi.MX.sym("p", 2)
        f = casadi.Function("f", [x, p], [x * x - p[0] * x - p[1]])
        rf = casadi.rootfinder("rf", "brent", f, {"lo": 0.0, "hi": 10.0})
        jacobian = casadi.Function("J", [p], [casadi.jacobian(rf(0.0, p), p)])
        np.testing.assert_allclose(
            np.asarray(jacobian(casadi.DM([1.0, 6.0]))).ravel(), [0.6, 0.2], rtol=1e-12
        )

    def test_composes_inside_a_graph(self):
        rf, _ = _solver(lambda x, p: x * x - p * x - 6.0, lo=0.0, hi=10.0)
        p = casadi.MX.sym("p")
        composed = casadi.Function("composed", [p], [3 * rf(0.0, p) + 1])
        assert float(composed(1.0)) == pytest.approx(10.0, abs=1e-12)

    def test_survives_a_serialize_round_trip(self):
        # pybamm hands functions to IDAKLU as generate_function(fn.serialize())
        rf, _ = _solver(lambda x, p: x * x - p * x - 6.0, lo=0.0, hi=10.0)
        p = casadi.MX.sym("p")
        composed = casadi.Function("composed", [p], [3 * rf(0.0, p) + 1])
        rebuilt = casadi.Function.deserialize(composed.serialize())
        assert float(rebuilt(1.0)) == float(composed(1.0)) == pytest.approx(10.0)

    def test_no_sign_change_fails_rather_than_guessing(self):
        rf, _ = _solver(lambda x, p: x * x + 1.0, lo=0.0, hi=1.0)
        with pytest.raises(RuntimeError, match="rootfinder process failed"):
            rf(0.0, 0.0)

    def test_reports_iteration_count(self):
        rf, _ = _solver(lambda x, p: casadi.exp(x) + p * x - 2.0)
        rf(0.0, 1.0)
        assert rf.stats()["iter_count"] > 0
        assert rf.stats()["return_status"] == "success"

    def test_rejects_a_non_scalar_system(self):
        x, p = casadi.MX.sym("x", 2), casadi.MX.sym("p")
        f = casadi.Function("f", [x, p], [x - p])
        with pytest.raises(RuntimeError, match="Brent solves a scalar residual"):
            casadi.rootfinder("rf", "brent", f, {"lo": 0.0, "hi": 1.0})


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX cc invocation")
@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
class TestBrentCodegen:
    """PyBaMM AOT-compiles its CasADi functions, so a Brent node has to survive
    Function.generate() -> C -> compile -> casadi.external unchanged."""

    @staticmethod
    def _compile(function, tmp_path, name):
        """Generate, compile and load ``function``; returns the C and the external."""
        function.generate(f"{name}.c", {"with_header": False})
        source = tmp_path / f"{name}.c"
        library = tmp_path / f"{name}.so"
        subprocess.run(  # noqa: S603
            ["cc", "-fPIC", "-shared", "-O2", "-o", str(library), str(source)],
            check=True,
            capture_output=True,
        )
        return source.read_text(), casadi.external(name, str(library))

    @pytest.fixture
    def _in_tmp_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_generates_compilable_c(self, _in_tmp_path, recwarn):
        rf, _ = _solver(non_monotone)
        p = casadi.MX.sym("p")
        composed = casadi.Function("composed", [p], [rf(0.0, p)])
        source, _ = self._compile(composed, _in_tmp_path, "composed")
        assert "#error" not in source
        assert not [w for w in recwarn if "code generated" in str(w.message)]

    def test_generated_c_matches_the_interpreted_plugin(self, _in_tmp_path):
        rf, f = _solver(non_monotone)
        p = casadi.MX.sym("p")
        composed = casadi.Function("composed", [p], [rf(0.0, p)])
        _, external = self._compile(composed, _in_tmp_path, "composed")

        compared = 0
        for target in np.linspace(0.2, 1.8, 61):
            if float(f(LO, target)) * float(f(HI, target)) > 0:
                continue  # no bracket for this target
            np.testing.assert_allclose(
                float(external(target)), float(composed(target)), rtol=0, atol=1e-14
            )
            compared += 1
        assert compared >= 40

    def test_a_bracket_read_from_an_input_survives_codegen(self, _in_tmp_path):
        x, lo, hi = casadi.MX.sym("x"), casadi.MX.sym("lo"), casadi.MX.sym("hi")
        f = casadi.Function("f", [x, lo, hi], [x * x - 6.0])
        rf = casadi.rootfinder("rf", "brent", f, {"lo_index": 1, "hi_index": 2})
        a, b = casadi.MX.sym("a"), casadi.MX.sym("b")
        composed = casadi.Function("bracketed", [a, b], [rf(0.0, a, b)])
        _, external = self._compile(composed, _in_tmp_path, "bracketed")
        for bracket in ((0.0, 10.0), (-10.0, 0.0)):
            np.testing.assert_allclose(
                float(external(*bracket)), float(composed(*bracket)), rtol=0, atol=1e-14
            )

    def test_two_brent_nodes_share_one_iteration(self, _in_tmp_path):
        first, _ = _solver(non_monotone)
        second, _ = _solver(lambda x, p: casadi.exp(x) + p * x - 2.0)
        p = casadi.MX.sym("p")
        composed = casadi.Function("both", [p], [first(0.0, p) + second(0.0, p)])
        source, external = self._compile(composed, _in_tmp_path, "both")
        # one iteration behind an include guard, one residual wrapper per node
        assert source.count("#ifndef CASADI_BRENT_IMPL") == 2
        assert source.count("static int casadi_brent_res_") == 2
        np.testing.assert_allclose(
            float(external(0.9)), float(composed(0.9)), rtol=0, atol=1e-14
        )

    def test_the_derivative_survives_codegen(self, _in_tmp_path):
        rf, f = _solver(non_monotone)
        p = casadi.MX.sym("p")
        composed = casadi.Function("composed", [p], [rf(0.0, p)])
        jacobian = casadi.Function("djac", [p], [casadi.jacobian(composed(p), p)])
        _, external = self._compile(jacobian, _in_tmp_path, "djac")
        for target in (0.4, 0.9, 1.4):
            np.testing.assert_allclose(
                float(external(target)), float(jacobian(target)), rtol=0, atol=1e-14
            )
