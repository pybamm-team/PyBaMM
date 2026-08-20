"""Unit tests for the "brent" CasADi rootfinder plugin."""

from __future__ import annotations

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
    """A rootfinder over ``expr_fn(x, p) == 0``, and a callable for the residual.

    The bracket is bound to constants here, so the returned solver takes ``p``.
    """
    x, lo_s, hi_s, p = (casadi.MX.sym(n) for n in ("x", "lo", "hi", "p"))
    g = casadi.Function("g", [x, lo_s, hi_s, p], [expr_fn(x, p)])
    rf = casadi.rootfinder("rf", "brent", g, opts)
    bound = casadi.Function("bound", [p], [rf(0.0, lo, hi, p)])
    return bound, casadi.Function("f", [x, p], [expr_fn(x, p)])


class TestBrentPlugin:
    def test_importing_pybammsolvers_registers_the_plugin(self):
        assert casadi.has_rootfinder("brent")

    def test_solves_a_scalar_equation(self):
        rf, _ = _solver(lambda x, p: x * x - p * x - 6.0, lo=0.0, hi=10.0)
        assert float(rf(1.0)) == pytest.approx(3.0, abs=1e-12)

    @pytest.mark.parametrize(
        "expr_fn", [lambda x, p: casadi.exp(x) + p * x - 2.0, non_monotone]
    )
    def test_matches_scipy_over_a_sweep(self, expr_fn):
        rf, f = _solver(expr_fn)
        worst_difference = worst_residual = 0.0
        for p in np.linspace(0.2, 1.8, 40):
            if float(f(LO, p)) * float(f(HI, p)) > 0:
                continue  # no bracket for this p, nothing to compare
            got = float(rf(p))
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
        rf = casadi.rootfinder("rf", "brent", f, {})
        assert float(rf(0.0, 0.0, 10.0)) == pytest.approx(np.sqrt(6), abs=1e-12)
        assert float(rf(0.0, -10.0, 0.0)) == pytest.approx(-np.sqrt(6), abs=1e-12)

    def test_derivatives_come_from_the_implicit_function_theorem(self):
        # x^2 - p0 x - p1 = 0 at (1, 6) has root 3; dx/dp = [x, 1] / (2x - p0)
        x, lo, hi = casadi.MX.sym("x"), casadi.MX.sym("lo"), casadi.MX.sym("hi")
        p = casadi.MX.sym("p", 2)
        g = casadi.Function("g", [x, lo, hi, p], [x * x - p[0] * x - p[1]])
        rf = casadi.rootfinder("rf", "brent", g, {})
        root = rf(0.0, 0.0, 10.0, p)
        jacobian = casadi.Function("J", [p], [casadi.jacobian(root, p)])
        np.testing.assert_allclose(
            np.asarray(jacobian(casadi.DM([1.0, 6.0]))).ravel(), [0.6, 0.2], rtol=1e-12
        )

    def test_composes_inside_a_graph(self):
        rf, _ = _solver(lambda x, p: x * x - p * x - 6.0, lo=0.0, hi=10.0)
        p = casadi.MX.sym("p")
        composed = casadi.Function("composed", [p], [3 * rf(p) + 1])
        assert float(composed(1.0)) == pytest.approx(10.0, abs=1e-12)

    def test_survives_a_serialize_round_trip(self):
        # pybamm hands functions to IDAKLU as generate_function(fn.serialize())
        rf, _ = _solver(lambda x, p: x * x - p * x - 6.0, lo=0.0, hi=10.0)
        p = casadi.MX.sym("p")
        composed = casadi.Function("composed", [p], [3 * rf(p) + 1])
        rebuilt = casadi.Function.deserialize(composed.serialize())
        assert float(rebuilt(1.0)) == float(composed(1.0)) == pytest.approx(10.0)

    def test_no_sign_change_fails_rather_than_guessing(self):
        rf, _ = _solver(lambda x, p: x * x + 1.0, lo=0.0, hi=1.0)
        with pytest.raises(RuntimeError, match="rootfinder process failed"):
            rf(0.0)

    def test_reports_iteration_count(self):
        x, lo, hi, p = (casadi.MX.sym(n) for n in ("x", "lo", "hi", "p"))
        g = casadi.Function("g", [x, lo, hi, p], [casadi.exp(x) + p * x - 2.0])
        rf = casadi.rootfinder("rf", "brent", g, {})
        rf(0.0, LO, HI, 1.0)
        assert rf.stats()["iter_count"] > 0
        assert rf.stats()["return_status"] == "success"

    def test_rejects_a_non_scalar_system(self):
        x = casadi.MX.sym("x", 2)
        lo, hi = casadi.MX.sym("lo"), casadi.MX.sym("hi")
        g = casadi.Function("g", [x, lo, hi], [x - 1.0])
        with pytest.raises(RuntimeError, match="Brent solves a scalar residual"):
            casadi.rootfinder("rf", "brent", g, {})

    def test_rejects_an_oracle_without_bracket_inputs(self):
        x, p = casadi.MX.sym("x"), casadi.MX.sym("p")
        g = casadi.Function("g", [x, p], [x - p])
        with pytest.raises(RuntimeError, match="g\\(x, lo, hi"):
            casadi.rootfinder("rf", "brent", g, {})


class TestBrentCodegen:
    """PyBaMM AOT-compiles its CasADi functions, so a Brent node has to survive
    Function.generate() -> C -> compile -> casadi.external unchanged."""

    @staticmethod
    def _compile(function, tmp_path, name):
        """Generate, compile and load ``function``; returns the C and the external.

        CasADi's own Importer drives the compiler, so this works wherever CasADi
        does rather than only where a POSIX ``cc`` is on the path.
        """
        function.generate(f"{name}.c", {"with_header": False})
        source = tmp_path / f"{name}.c"
        external = casadi.external(name, casadi.Importer(str(source), "shell"))
        return source.read_text(), external

    @pytest.fixture
    def _in_tmp_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_generates_compilable_c(self, _in_tmp_path, recwarn):
        rf, _ = _solver(non_monotone)
        p = casadi.MX.sym("p")
        composed = casadi.Function("composed", [p], [rf(p)])
        source, _ = self._compile(composed, _in_tmp_path, "composed")
        assert "#error" not in source
        assert not [w for w in recwarn if "code generated" in str(w.message)]

    def test_generated_c_matches_the_interpreted_plugin(self, _in_tmp_path):
        rf, f = _solver(non_monotone)
        p = casadi.MX.sym("p")
        composed = casadi.Function("composed", [p], [rf(p)])
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
        rf = casadi.rootfinder("rf", "brent", f, {})
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
        composed = casadi.Function("both", [p], [first(p) + second(p)])
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
        composed = casadi.Function("composed", [p], [rf(p)])
        jacobian = casadi.Function("djac", [p], [casadi.jacobian(composed(p), p)])
        _, external = self._compile(jacobian, _in_tmp_path, "djac")
        for target in (0.4, 0.9, 1.4):
            np.testing.assert_allclose(
                float(external(target)), float(jacobian(target)), rtol=0, atol=1e-14
            )


class TestBrentCache:
    """A Brent nested inside another must not re-solve on every enclosing iteration.

    Covered on the interpreted and the generated path.
    """

    @staticmethod
    def _nested():
        """``x`` such that ``g(x) = 0`` where ``g`` reads an inner solve of its own."""
        target = casadi.MX.sym("target")
        inner_rf, _ = _solver(non_monotone)
        inner = inner_rf(target)
        outer_x = casadi.MX.sym("x")
        outer = casadi.rootfinder(
            "outer",
            "brent",
            casadi.Function(
                "outer_g",
                [outer_x, casadi.MX.sym("lo"), casadi.MX.sym("hi"), target],
                [outer_x - inner],
            ),
            {"abstol": 1e-13, "max_iter": 200},
        )
        return casadi.Function("nested", [target], [outer(LO, LO, HI, target)])

    def test_repeating_the_inputs_reuses_the_last_solve(self):
        x, lo, hi, p = (casadi.MX.sym(n) for n in ("x", "lo", "hi", "p"))
        rf = casadi.rootfinder(
            "rf",
            "brent",
            casadi.Function("g", [x, lo, hi, p], [non_monotone(x, p)]),
        )
        first = float(rf(0.0, LO, HI, 1.0))
        assert rf.stats()["return_status"] == "success"
        iterations = rf.stats()["iter_count"]

        # Same inputs: the root comes back from the cache, so the iteration count
        # does not move and the status says so.
        assert float(rf(0.0, LO, HI, 1.0)) == first
        assert rf.stats()["return_status"] == "success (cached)"
        assert rf.stats()["iter_count"] == iterations

        # A different target has to be solved afresh.
        rf(0.0, LO, HI, 1.4)
        assert rf.stats()["return_status"] == "success"

    def test_generated_c_carries_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        nested = self._nested()
        nested.generate("nested.c", {"with_header": False})
        source = (tmp_path / "nested.c").read_text()
        # The generated iteration keeps its own copy of the cache, so an
        # ahead-of-time compiled function does not regress to re-solving.
        assert "brent_cache" in source
        assert "CASADI_BRENT_TLS" in source

        external = casadi.external(
            "nested", casadi.Importer(str(tmp_path / "nested.c"), "shell")
        )
        for target in (0.7, 1.0, 1.4):
            np.testing.assert_allclose(
                float(external(target)), float(nested(target)), rtol=0, atol=1e-14
            )


class TestIterationLimit:
    """Running out of iterations is a failure, not a root.

    Exhausting ``max_iter`` leaves an arbitrary point inside the bracket.
    """

    @staticmethod
    def _steep():
        """``exp(x) - 100``, whose root at ``log(100)`` takes many iterations to reach."""
        x, lo, hi, p = (casadi.MX.sym(n) for n in ("x", "lo", "hi", "p"))
        return casadi.Function("g", [x, lo, hi, p], [casadi.exp(x) - 100])

    def test_exhausting_the_iterations_is_not_a_success(self):
        rootfinder = casadi.rootfinder(
            "rf", "brent", self._steep(), {"max_iter": 1, "error_on_fail": False}
        )
        rootfinder(0.0, 0.0, 10.0, 0.0)
        assert rootfinder.stats()["return_status"] == (
            "iteration limit reached without converging"
        )

    def test_the_same_problem_converges_when_given_the_iterations(self):
        rootfinder = casadi.rootfinder("rf", "brent", self._steep(), {"max_iter": 200})
        root = float(rootfinder(0.0, 0.0, 10.0, 0.0))
        assert rootfinder.stats()["return_status"] == "success"
        np.testing.assert_allclose(root, np.log(100), rtol=1e-12)

    def test_an_exhausted_solve_is_not_cached(self):
        rootfinder = casadi.rootfinder(
            "rf", "brent", self._steep(), {"max_iter": 1, "error_on_fail": False}
        )
        rootfinder(0.0, 0.0, 10.0, 0.0)
        rootfinder(0.0, 0.0, 10.0, 0.0)
        # A cached hit would report "success (cached)" and hand back the non-root.
        assert rootfinder.stats()["return_status"] == (
            "iteration limit reached without converging"
        )

    def test_the_generated_c_also_refuses_to_converge(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rootfinder = casadi.rootfinder(
            "rf", "brent", self._steep(), {"max_iter": 1, "error_on_fail": False}
        )
        # `p` stays symbolic: an all-constant call is folded away at build time and
        # never reaches the generated C.
        target = casadi.MX.sym("p")
        wrapped = casadi.Function(
            "wrapped", [target], [rootfinder(0.0, 0.0, 10.0, target)]
        )
        wrapped.generate("wrapped.c", {"with_header": False})
        external = casadi.external(
            "wrapped", casadi.Importer(str(tmp_path / "wrapped.c"), "shell")
        )
        with pytest.raises(RuntimeError):
            external(0.0)
