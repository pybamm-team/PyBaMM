"""Forces every concrete pybamm.Symbol subclass to have a Hypothesis strategy.

A new subclass added to pybamm must either:
  - be registered in ``tests/strategies/symbols.py::_STRATEGIES``, or
  - be added to ``_NOT_ROUND_TRIPPABLE`` (permanent exemption) or
    ``_KNOWN_FAILING`` (serialiser bug awaiting a fix), with a one-line comment.

Forgetting all of them is a test failure, which is the whole point.
"""

from __future__ import annotations

import pybamm
from tests.strategies.symbols import (
    _EXEMPT,
    _KNOWN_FAILING,
    _NOT_ROUND_TRIPPABLE,
    _STRATEGIES,
)


def _all_concrete_subclasses(root: type) -> set[type]:
    """Recursively collect concrete subclasses of ``root``.

    A class is concrete if it has no ``__abstractmethods__``.
    """
    seen: set[type] = set()
    stack = list(root.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return {c for c in seen if not getattr(c, "__abstractmethods__", None)}


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
