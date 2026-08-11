#
# Tests for the Brent expression tree node
#

import sys

import casadi
import numpy as np
import pytest
from scipy.optimize import brentq

import pybamm
from pybamm.expression_tree.operations.serialise import (
    convert_symbol_from_json,
    convert_symbol_to_json,
)


def _evaluate(symbol, **inputs):
    """Evaluate a Brent expression over casadi symbols for each named input."""
    symbols = {name: casadi.MX.sym(name) for name in inputs}
    expression = symbol.to_casadi(inputs=symbols)
    function = casadi.Function("f", list(symbols.values()), [expression])
    return float(function(*[inputs[name] for name in symbols]))


class TestBrent:
    def test_solves_a_scalar_equation(self):
        x = pybamm.Variable("x")
        node = pybamm.Brent(pybamm.exp(x) + x - 2.0, x, (-5.0, 5.0))
        got = float(casadi.evalf(node.to_casadi(inputs={})))
        want = brentq(lambda v: np.exp(v) + v - 2.0, -5.0, 5.0, xtol=2e-12)
        assert got == pytest.approx(want, abs=1e-12)

    def test_target_may_be_an_input_parameter(self):
        x = pybamm.Variable("x")
        node = pybamm.Brent(x * x - pybamm.InputParameter("target"), x, (0.0, 10.0))
        assert _evaluate(node, target=9.0) == pytest.approx(3.0, abs=1e-12)
        assert _evaluate(node, target=4.0) == pytest.approx(2.0, abs=1e-12)

    def test_bracket_may_be_input_parameters(self):
        # x^2 = 6 has roots at +-sqrt(6); the bracket selects one, at solve time
        x = pybamm.Variable("x")
        node = pybamm.Brent(
            x * x - 6.0, x, (pybamm.InputParameter("lo"), pybamm.InputParameter("hi"))
        )
        assert _evaluate(node, lo=0.0, hi=10.0) == pytest.approx(np.sqrt(6), abs=1e-12)
        assert _evaluate(node, lo=-10.0, hi=0.0) == pytest.approx(
            -np.sqrt(6), abs=1e-12
        )

    def test_the_expression_may_contain_input_parameters(self):
        # a x^2 = 9 has positive root 3 / sqrt(a)
        x = pybamm.Variable("x")
        node = pybamm.Brent(pybamm.InputParameter("a") * x * x - 9.0, x, (0.0, 10.0))
        for a in (1.0, 4.0, 9.0):
            assert _evaluate(node, a=a) == pytest.approx(3.0 / np.sqrt(a), abs=1e-12)

    def test_every_argument_may_be_an_input_parameter_at_once(self):
        x = pybamm.Variable("x")
        node = pybamm.Brent(
            pybamm.InputParameter("a") * x * x - pybamm.InputParameter("target"),
            x,
            (pybamm.InputParameter("lo"), pybamm.InputParameter("hi")),
        )
        got = _evaluate(node, a=4.0, target=9.0, lo=0.0, hi=10.0)
        assert got == pytest.approx(1.5, abs=1e-12)
        got = _evaluate(node, a=4.0, target=9.0, lo=-10.0, hi=0.0)
        assert got == pytest.approx(-1.5, abs=1e-12)

    def test_solves_over_the_state_vector(self):
        state = pybamm.StateVector(slice(0, 1))
        x = pybamm.Variable("x")
        node = pybamm.Brent(x * state - 6.0, x, (0.0, 10.0))
        y = casadi.MX.sym("y", 1)
        expression = casadi.Function("f", [y], [node.to_casadi(y=y, inputs={})])
        assert float(expression(2.0)) == pytest.approx(3.0, abs=1e-12)
        assert float(expression(3.0)) == pytest.approx(2.0, abs=1e-12)

    def test_derivative_is_exact(self):
        # x = sqrt(target), so dx/d(target) = 1 / (2 sqrt(target))
        x = pybamm.Variable("x")
        node = pybamm.Brent(x * x - pybamm.InputParameter("target"), x, (0.0, 10.0))
        symbol = casadi.MX.sym("target")
        root = node.to_casadi(inputs={"target": symbol})
        derivative = casadi.Function("J", [symbol], [casadi.jacobian(root, symbol)])
        assert float(derivative(9.0)) == pytest.approx(1 / 6, rel=1e-12)

    def test_composes_into_a_larger_expression(self):
        x = pybamm.Variable("x")
        node = pybamm.Brent(x * x - 9.0, x, (0.0, 10.0))
        got = float(casadi.evalf((3 * node + pybamm.Scalar(1)).to_casadi(inputs={})))
        assert got == pytest.approx(10.0, abs=1e-12)

    def test_nests(self):
        # the inner solve gives sqrt(16) = 4, so the outer gives sqrt(4) = 2
        inner_x = pybamm.Variable("inner")
        inner = pybamm.Brent(inner_x * inner_x - 16.0, inner_x, (0.0, 10.0))
        outer_x = pybamm.Variable("outer")
        outer = pybamm.Brent(outer_x * outer_x - inner, outer_x, (0.0, 10.0))
        assert float(casadi.evalf(outer.to_casadi(inputs={}))) == pytest.approx(2.0)

    def test_evaluating_does_not_re_enter_python(self):
        # the whole solve runs in the CasADi graph, so a Brent node must cost no more
        # python frames per evaluation than the same expression without one
        state = pybamm.StateVector(slice(3, 4))
        x = pybamm.Variable("x")
        node = pybamm.Brent(pybamm.exp(x) + x * state - 2.0, x, (-5.0, 5.0))
        y = casadi.MX.sym("y", 500)
        with_brent = casadi.Function("a", [y], [3 * node.to_casadi(y=y, inputs={}) + 1])
        without = casadi.Function("b", [y], [3 * casadi.exp(y[3]) + 1])
        values = np.zeros(500)
        values[3] = 1.5

        def count_frames(function):
            calls = 0

            def profile(frame, event, arg):
                nonlocal calls
                if event == "call":
                    calls += 1

            sys.setprofile(profile)
            try:
                for _ in range(50):
                    function(values)
            finally:
                sys.setprofile(None)
            return calls

        assert count_frames(with_brent) == count_frames(without)

    def test_the_oracle_only_reads_what_the_residual_needs(self):
        # a residual that ignores time must not drag time into the solve
        state = pybamm.StateVector(slice(0, 1))
        x = pybamm.Variable("x")
        node = pybamm.Brent(x * state - 6.0, x, (0.0, 10.0))
        t, y = casadi.MX.sym("t"), casadi.MX.sym("y", 1)
        names = [s.name() for s in casadi.symvar(node.to_casadi(t=t, y=y, inputs={}))]
        assert names == ["y"]

    def test_no_sign_change_fails_rather_than_guessing(self):
        x = pybamm.Variable("x")
        node = pybamm.Brent(x * x + 1 - 0.0, x, (0.0, 1.0))
        with pytest.raises(RuntimeError, match="rootfinder process failed"):
            casadi.evalf(node.to_casadi(inputs={}))

    def test_children_and_copy(self):
        x = pybamm.Variable("x")
        node = pybamm.Brent(x * x - 9.0, x, (0.0, 10.0))
        assert len(node.children) == 5
        copy = node.create_copy()
        assert copy.name == node.name
        assert float(casadi.evalf(copy.to_casadi(inputs={}))) == pytest.approx(3.0)

    def test_expansion_finds_a_bracket_the_bounds_missed(self):
        # the bracket holds no sign change, so the plain method has nothing to do; with
        # expansion the walk finds the root at 3 anyway
        x = pybamm.Variable("x")
        residual = x * x - 9.0
        with pytest.raises(RuntimeError, match="rootfinder process failed"):
            casadi.evalf(pybamm.Brent(residual, x, (0.0, 1.0)).to_casadi(inputs={}))
        node = pybamm.Brent(residual, x, (0.0, 1.0), max_expansions=50)
        assert float(casadi.evalf(node.to_casadi(inputs={}))) == pytest.approx(3.0)

    def test_expansion_never_raises(self):
        # x^2 + 1 has no real root at all, so nothing can be bracketed; expansion must
        # still return a number rather than failing
        x = pybamm.Variable("x")
        node = pybamm.Brent(x * x + 1.0, x, (0.0, 1.0), max_expansions=20)
        assert np.isfinite(float(casadi.evalf(node.to_casadi(inputs={}))))

    def test_the_guess_only_sets_the_scale(self):
        # far guess, wrong side, tiny bounds: the answer is the same either way
        x = pybamm.Variable("x")
        for guess in (-500.0, 0.5, 1e4):
            node = pybamm.Brent(
                x * x - 9.0, x, (0.0, 0.1), guess=guess, max_expansions=80
            )
            got = float(casadi.evalf(node.to_casadi(inputs={})))
            assert got == pytest.approx(3.0, abs=1e-9)

    def test_the_hyperparameters_are_part_of_the_identity(self):
        # the id keys the caches in ParameterValues.process_symbol and _to_casadi, so
        # two nodes differing only in a hyperparameter must not collide there
        x = pybamm.Variable("x")
        residual = x * x - 9.0
        strict = pybamm.Brent(residual, x, (0.0, 1.0))
        expanding = pybamm.Brent(residual, x, (0.0, 1.0), max_expansions=50)
        assert strict.id != expanding.id
        assert pybamm.Brent(residual, x, (0, 1), abstol=1e-12).id != strict.id
        assert pybamm.Brent(residual, x, (0, 1), max_iter=7).id != strict.id

        parameter_values = pybamm.ParameterValues({})
        with pytest.raises(RuntimeError, match="rootfinder process failed"):
            casadi.evalf(parameter_values.process_symbol(strict).to_casadi(inputs={}))
        got = casadi.evalf(
            parameter_values.process_symbol(expanding).to_casadi(inputs={})
        )
        assert float(got) == pytest.approx(3.0)

    def test_errors(self):
        x = pybamm.Variable("x")
        with pytest.raises(ValueError, match="max_expansions must not be negative"):
            pybamm.Brent(x * x - 9.0, x, (0, 1), max_expansions=-1)
        with pytest.raises(TypeError, match=r"unknown must be a pybamm\.Symbol"):
            pybamm.Brent(x * x - 9.0, 1.0, (0, 1))
        with pytest.raises(TypeError, match=r"residual must be a pybamm\.Symbol"):
            pybamm.Brent(1.0, x, (0, 1))
        with pytest.raises(ValueError, match="does not appear in"):
            pybamm.Brent(pybamm.Scalar(2) * pybamm.t - 9.0, x, (0, 1))
        with pytest.raises(ValueError, match="bounds must be a"):
            pybamm.Brent(x * x - 9.0, x, (0, 1, 2))

        node = pybamm.Brent(x * x - 9.0, x, (0.0, 10.0))
        with pytest.raises(NotImplementedError, match="no symbolic derivative"):
            node.diff(pybamm.t)
        with pytest.raises(NotImplementedError, match="no symbolic jacobian"):
            node._jac(pybamm.t)

    def test_round_trips_through_json(self):
        x = pybamm.Variable("x")
        node = pybamm.Brent(x * x - 9.0, x, (0.0, 10.0), abstol=1e-12, max_iter=42)
        rebuilt = convert_symbol_from_json(convert_symbol_to_json(node))
        assert rebuilt.abstol == 1e-12
        assert rebuilt.max_iter == 42
        assert rebuilt.unknown == node.unknown
        assert float(casadi.evalf(rebuilt.to_casadi(inputs={}))) == pytest.approx(3.0)

    def test_an_unresolved_unknown_is_an_error(self):
        with pytest.raises(TypeError, match="Cannot convert symbol of type"):
            pybamm.Variable("x").to_casadi(inputs={})
