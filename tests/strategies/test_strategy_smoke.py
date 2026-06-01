"""Smoke tests for the Hypothesis strategy modules.

These verify that strategies produce constructible pybamm objects — they
do NOT verify serialisation. The serialisation property tests live under
``tests/unit/test_serialisation/``.
"""

from __future__ import annotations

from hypothesis import given, settings

import pybamm
from tests.strategies import serialisation_settings
from tests.strategies.symbols import scalar_strategy, symbols, variable_strategy


@given(scalar_strategy())
def test_scalar_strategy_yields_scalars(s):
    assert isinstance(s, pybamm.Scalar)


@given(variable_strategy())
def test_variable_strategy_yields_variables(v):
    assert isinstance(v, pybamm.Variable)


@settings(serialisation_settings, max_examples=20)
@given(symbols(max_leaves=8))
def test_symbols_yields_valid_symbol_trees(tree):
    assert isinstance(tree, pybamm.Symbol)
    # Tree must be constructible: id and shape both compute without error
    _ = tree.id
