"""Round-trip regression test for the #5548 cases (each previously dropped a
non-children constructor argument on round-trip).
"""

from __future__ import annotations

import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.operations.serialise import (
    convert_symbol_from_json as _from,
)
from pybamm.expression_tree.operations.serialise import (
    convert_symbol_to_json as _to,
)


def _V(d=None):
    return pybamm.Variable("u", domains={"primary": [d]} if d else {})


_CASES = [
    pybamm.BoundaryGradient(pybamm.Time(), "left"),
    pybamm.BackwardIndefiniteIntegral(
        _V("negative electrode"),
        pybamm.SpatialVariable("x", domain=["negative electrode"]),
    ),
    pybamm.ExplicitTimeIntegral(_V("negative electrode"), pybamm.Scalar(0.0)),
    pybamm.Index(pybamm.StateVector(slice(0, 1)), 0, check_size=False),
    pybamm.BoundaryIntegral(_V("negative electrode"), region="negative tab"),
    pybamm.DeltaFunction(_V("negative particle"), "left", "negative electrode"),
    pybamm.EvaluateAt(_V("negative electrode"), 0.5),
    pybamm.Magnitude(_V("negative electrode"), "x"),
    pybamm.SizeAverage(_V("negative particle size"), pybamm.Scalar(1.0)),
    pybamm.TensorField([pybamm.Scalar(1.0), pybamm.Scalar(2.0)]),
    pybamm.TertiaryBroadcast(
        pybamm.Variable(
            "v",
            domains={
                "primary": ["negative particle"],
                "secondary": ["negative electrode"],
            },
        ),
        "current collector",
    ),
    pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1.0), "negative electrode"),
    pybamm.SpatialVariableEdge("x", domain=["negative electrode"]),
    pybamm.BoundaryValue(_V("negative electrode"), "left"),
    pybamm.DefiniteIntegralVector(_V("negative electrode")),
    pybamm.VectorField(_V("negative electrode"), _V("negative electrode")),
    pybamm.DiscreteTimeSum(
        pybamm.DiscreteTimeData(np.array([0.0, 1.0]), np.array([1.0, 2.0]), "dtd")
    ),
    pybamm.Concatenation(
        pybamm.FullBroadcast(
            pybamm.Scalar(1.0),
            broadcast_domains={"primary": ["negative electrode"]},
        ),
        pybamm.FullBroadcast(
            pybamm.Scalar(2.0),
            broadcast_domains={"primary": ["separator"]},
        ),
    ),
    pybamm.ConcatenationVariable(_V("negative electrode"), _V("separator")),
    pybamm.SparseStack(_V("negative electrode"), _V("separator")),
]


@pytest.mark.parametrize("tree", _CASES, ids=lambda t: type(t).__name__)
def test_issue_5548_repro_round_trips(tree):
    restored = _from(_to(tree))
    assert type(restored) is type(tree)
    assert restored.id == tree.id
