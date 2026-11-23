"""Source-level solution-fold checks with analytically specified trajectories."""

from functools import reduce
from itertools import product
from math import fsum
from operator import add

import numpy as np
import pytest

import pybamm
from pybamm.solvers.observation import OutputAssembly

SEED = 7.0
GAPPED = ((0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0))
ADJACENT = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))
EXPRESSIONS = ("Integral", "Signal")


def _segment(t0, t1, computed, offset=0.0):
    model = pybamm.BaseModel()
    y = pybamm.Variable("Signal")
    model.rhs = {y: pybamm.Scalar(3)}
    model.initial_conditions = {y: pybamm.Scalar(2 + 3 * t0 + offset)}
    model.variables = {
        "Integral": pybamm.ExplicitTimeIntegral(y, pybamm.Scalar(SEED)),
        "Signal": y,
    }
    pybamm.Discretisation().process_model(model)

    knots = np.linspace(t0, t1, 5)
    samples = 2 + 3 * knots + offset
    full = pybamm.Solution(knots, samples[None, :], model, {})
    if not computed:
        return full

    solution = pybamm.Solution(
        knots, np.empty((0, knots.size)), model, {}, variables_returned=True
    )
    leaves = {}
    integrals = {}
    for name in EXPRESSIONS:
        var = model.get_processed_variable_or_event(name)
        leaf, _, integral = full._convert_to_casadi(var, {}, (1, knots.size))
        leaves[name] = leaf
        if integral is not None:
            integrals[name] = integral
    payload = np.column_stack((samples, samples))
    OutputAssembly(EXPRESSIONS, leaves, time_integrals=integrals).attach(
        solution, payload
    )
    return solution


def _integral_over(t0, t1):
    return (t1 - t0) * (2 + 1.5 * (t0 + t1))


def _expected(intervals, pattern):
    # The documented binary fold bridges gaps only while its accumulated
    # solution and the next input are both full-state.
    first_computed = pattern.find("C")
    n = len(intervals) if first_computed < 0 else first_computed
    spans = [(intervals[0][0], intervals[n - 1][1])] if n else []
    spans.extend(intervals[n:])
    return SEED + fsum(_integral_over(*span) for span in spans)


@pytest.mark.parametrize(
    "intervals,pattern",
    [
        *(
            pytest.param(GAPPED[:3], "".join(s), id="gap-" + "".join(s))
            for s in product("FC", repeat=3)
        ),
        *(
            pytest.param(ADJACENT, "".join(s), id="adjacent-" + "".join(s))
            for s in product("FC", repeat=3)
        ),
        pytest.param(GAPPED, "FFCF", id="later-full"),
        pytest.param(GAPPED, "FFCC", id="later-computed"),
        pytest.param(GAPPED, "FFFC", id="three-full-prefix"),
        pytest.param(GAPPED, "CFFF", id="computed-first"),
    ],
)
def test_bulk_integral_matches_documented_binary_fold(intervals, pattern):
    segments = [
        _segment(*span, style == "C")
        for span, style in zip(intervals, pattern, strict=True)
    ]
    binary = reduce(add, segments)
    bulk = pybamm.Solution.from_sub_solutions(segments)
    np.testing.assert_array_equal(bulk.t, binary.t)
    np.testing.assert_allclose(bulk["Signal"].entries, 2 + 3 * bulk.t)
    np.testing.assert_array_equal(bulk["Signal"].entries, binary["Signal"].entries)
    expected = _expected(intervals, pattern)
    np.testing.assert_allclose(binary["Integral"].entries, [expected], atol=1e-10)
    np.testing.assert_allclose(bulk["Integral"].entries, [expected], atol=1e-10)


def test_bulk_fold_uses_the_binary_prefix_at_a_shared_knot_with_a_jump():
    segments = [
        _segment(0, 1, False),
        _segment(1, 2, False, offset=8.0),
        _segment(2, 3, True, offset=8.0),
    ]
    binary = reduce(add, segments)
    bulk = pybamm.Solution.from_sub_solutions(segments)
    np.testing.assert_array_equal(bulk.t, binary.t)
    np.testing.assert_array_equal(bulk["Signal"].entries, binary["Signal"].entries)

    # Binary fold drops the right-side value at the first shared knot:
    # the first trapezoid after t=1 loses (8 / 2) * (1 / 4) = 1.
    expected = 26.5 + 16.0 - 1.0
    np.testing.assert_allclose(binary["Integral"].entries, [expected], atol=1e-10)
    np.testing.assert_allclose(bulk["Integral"].entries, [expected], atol=1e-10)
