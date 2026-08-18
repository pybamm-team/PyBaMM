"""Forces every concrete pybamm.Symbol subclass to have a Hypothesis strategy.

A new subclass added to pybamm must either:
  - be registered in ``tests/strategies/symbols.py::_STRATEGIES``, or
  - be added to ``_NOT_ROUND_TRIPPABLE`` (permanent exemption) or
    ``_KNOWN_FAILING`` (serialiser bug awaiting a fix), with a one-line comment.

Forgetting all of them is a test failure, which is the whole point.

A second guard exercises every branch factory against exemplar children
covering each constraint axis (every domain the recursive pool can produce,
auxiliary domains, edge-evaluating nodes), so strategy-vs-constructor
mismatches are caught deterministically rather than by random composition.
"""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import find, settings
from hypothesis.errors import NoSuchExample, Unsatisfiable

import pybamm
from pybamm.expression_tree.operations.serialise import (
    convert_symbol_from_json,
    convert_symbol_to_json,
)
from tests.strategies.symbols import (
    _EXEMPT,
    _KNOWN_FAILING,
    _LEAF_CLASSES,
    _NOT_ROUND_TRIPPABLE,
    _STRATEGIES,
)


def _all_concrete_subclasses(root: type) -> set[type]:
    """Recursively collect concrete pybamm subclasses of ``root``.

    A class is concrete if it has no ``__abstractmethods__``. Synthetic Symbol
    subclasses defined inside test modules are ignored (only classes shipped in
    the ``pybamm`` package are subject to the strategy-coverage contract).
    """
    seen: set[type] = set()
    stack = list(root.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return {
        c
        for c in seen
        if not getattr(c, "__abstractmethods__", None)
        and c.__module__.startswith("pybamm")
    }


def test_every_concrete_symbol_subclass_has_a_strategy():
    concrete = _all_concrete_subclasses(pybamm.Symbol)
    covered = set(_STRATEGIES.keys()) | _EXEMPT
    missing = concrete - covered
    assert not missing, (
        "These pybamm.Symbol subclasses have no strategy and are not exempt. "
        "Add to _STRATEGIES in tests/strategies/symbols.py, or to _EXEMPT "
        "with a justifying comment: " + ", ".join(sorted(c.__name__ for c in missing))
    )


def test_strategies_and_exempt_are_disjoint():
    overlap = set(_STRATEGIES.keys()) & _EXEMPT
    assert not overlap, (
        "Classes in both _STRATEGIES and _EXEMPT — pick one: "
        + ", ".join(sorted(c.__name__ for c in overlap))
    )


def test_exemption_sets_are_disjoint():
    """A class is either permanently exempt or a known bug — never both."""
    overlap = _NOT_ROUND_TRIPPABLE & _KNOWN_FAILING
    assert not overlap, (
        "Classes in both _NOT_ROUND_TRIPPABLE and _KNOWN_FAILING — pick one: "
        + ", ".join(sorted(c.__name__ for c in overlap))
    )


# One exemplar child per constraint axis a constructor may check: each primary
# domain the recursive pool can produce, an auxiliary (secondary) domain, and
# branch-rooted / edge-evaluating children.
_EXEMPLAR_CHILDREN: dict[str, object] = {
    "empty-domain": lambda: pybamm.Scalar(1.5),
    "negative-electrode": lambda: pybamm.Variable(
        "v", domains={"primary": ["negative electrode"]}
    ),
    "positive-electrode": lambda: pybamm.Variable(
        "v", domains={"primary": ["positive electrode"]}
    ),
    "separator": lambda: pybamm.Variable("v", domains={"primary": ["separator"]}),
    "negative-particle": lambda: pybamm.Variable(
        "v", domains={"primary": ["negative particle"]}
    ),
    "current-collector": lambda: pybamm.Variable(
        "v", domains={"primary": ["current collector"]}
    ),
    "particle-with-secondary": lambda: pybamm.Variable(
        "v",
        domains={
            "primary": ["negative particle"],
            "secondary": ["negative electrode"],
        },
    ),
    "cc-branch-node": lambda: pybamm.Gradient(
        pybamm.Variable("v", domains={"primary": ["current collector"]})
    ),
    "electrode-on-edges": lambda: pybamm.Gradient(
        pybamm.Variable("v", domains={"primary": ["negative electrode"]})
    ),
}

_BRANCH_CLASSES = sorted(
    (cls for cls in _STRATEGIES if cls not in _LEAF_CLASSES),
    key=lambda cls: cls.__name__,
)

_FIND_SETTINGS = settings(max_examples=20, deadline=None, database=None)


@pytest.mark.parametrize("child_key", sorted(_EXEMPLAR_CHILDREN))
@pytest.mark.parametrize("cls", _BRANCH_CLASSES, ids=lambda cls: cls.__name__)
def test_branch_factories_handle_every_child_shape(cls, child_key):
    """Every branch factory must, for every child shape, either build a
    constructor-valid node that round-trips or reject the child cleanly
    (constructive guard / filter) — never leak a constructor error at
    generation time."""
    child = _EXEMPLAR_CHILDREN[child_key]()
    strategy = _STRATEGIES[cls](st.just(child))
    try:
        node = find(strategy, lambda _: True, settings=_FIND_SETTINGS)
    except (NoSuchExample, Unsatisfiable):
        return  # the factory rejects this child shape by construction
    restored = convert_symbol_from_json(convert_symbol_to_json(node))
    assert restored.id == node.id, (
        f"{cls.__name__} round-trip changed identity for child "
        f"{child_key!r}.\n  original: {node!r}\n  restored: {restored!r}"
    )
