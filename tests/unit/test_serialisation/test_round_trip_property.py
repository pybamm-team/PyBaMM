"""Hypothesis property tests for the JSON serialisation surfaces.

Companion to the parametrized regression tests in ``test_round_trip.py`` —
those guard against known bug shapes; these guard against unknown ones by
fuzzing across the construction grammar.
"""

from __future__ import annotations

import copy
import functools
import json

import pytest
from hypothesis import assume, given

import pybamm
from pybamm.expression_tree.operations.serialise import (
    _experiment_step_factories,
    convert_symbol_from_json,
    convert_symbol_to_json,
)
from tests.strategies import serialisation_settings
from tests.strategies.serialise_values import (
    _EXCLUDED_OPTION_VALUES,
    _FUZZED_OPTION_KEYS,
    experiment_kwargs,
    model_options_kwargs,
    parameter_values_overrides,
    solver_kwargs,
    step_kwargs,
)
from tests.strategies.symbols import symbols
from tests.unit.test_serialisation._helpers import (
    _STEP_ATTRS,
    _attr,
    _experiments_equal,
    _solver_init_args_equal,
    _values_equal,
)


@serialisation_settings
@given(symbols(max_leaves=6))
def test_symbol_json_round_trip(tree: pybamm.Symbol):
    """Every Symbol tree must satisfy ``from_json(to_json(tree)).id == tree.id``."""
    restored = convert_symbol_from_json(convert_symbol_to_json(tree))
    assert restored.id == tree.id, (
        f"Symbol round-trip changed identity.\n"
        f"  original: {tree!r}\n"
        f"  restored: {restored!r}"
    )


@serialisation_settings
@given(solver_kwargs())
def test_solver_round_trip(kwargs):
    """Fuzz BaseSolver-subclass kwargs and round-trip through to_config/from_config."""
    solver_cls = kwargs.pop("_cls")
    original = solver_cls(**kwargs)
    config = original.to_config()
    restored = pybamm.BaseSolver.from_config(config)
    assert _solver_init_args_equal(original, restored)


def _resolve_step_factory(step_type: str):
    """Return the pybamm.step factory for a serialised step ``"type"`` string.

    Uses the same ``_experiment_step_factories`` map the serialiser does, so the
    factory and the fuzzed type strings stay in lock-step (and hyphenated types
    like ``"c-rate"`` resolve correctly).
    """
    return _experiment_step_factories()[step_type]


@serialisation_settings
@given(step_kwargs())
def test_experiment_step_round_trip(kwargs):
    """Per-step kwargs must round-trip through Experiment serialisation.

    Steps are round-tripped by wrapping in a single-step Experiment and
    using Experiment.to_config() / Experiment.from_config(), which is the
    only serialisation path available for steps (they have no standalone
    to_config / from_config methods).

    skip_ok is fuzzed as a bool (NOT int 0/1) so #5495's bool->int coercion
    bug stays caught.
    """
    step_type = kwargs.pop("_type")
    factory = _resolve_step_factory(step_type)
    original = factory(**kwargs)

    experiment = pybamm.Experiment([original])
    config = json.loads(json.dumps(experiment.to_config()))
    restored_experiment = pybamm.Experiment.from_config(config)
    restored = restored_experiment.steps[0]

    # Drive the comparison from the shared _STEP_ATTRS list (single source of
    # truth). uses_default_duration is a derived flag, not round-tripped here.
    for attr in (a for a in _STEP_ATTRS if a != "uses_default_duration"):
        orig_val = _attr(original, attr)
        rest_val = _attr(restored, attr)
        assert _values_equal(orig_val, rest_val), (
            f"step attribute {attr!r} did not round-trip: "
            f"original={orig_val!r}, restored={rest_val!r}"
        )


@serialisation_settings
@given(experiment_kwargs())
def test_experiment_round_trip(kwargs):
    """Whole-Experiment round-trip including top-level period/temperature/termination."""
    operating_conditions = kwargs.pop("_steps")
    original = pybamm.Experiment(operating_conditions, **kwargs)
    config = original.to_config()
    restored = pybamm.Experiment.from_config(config)
    assert _experiments_equal(original, restored)


@functools.lru_cache(maxsize=1)
def _chen2020_base() -> pybamm.ParameterValues:
    """Load the Chen2020 set once; callers deep-copy before mutating it."""
    return pybamm.ParameterValues("Chen2020")


@serialisation_settings
@given(parameter_values_overrides())
def test_parameter_values_round_trip(overrides):
    """ParameterValues with fuzzed overrides round-trips through to_json/from_json."""
    base = copy.deepcopy(_chen2020_base())
    base.update(overrides, check_already_exists=False)
    data = json.loads(json.dumps(base.to_json()))
    restored = pybamm.ParameterValues.from_json(data)
    for key, value in overrides.items():
        assert _values_equal(restored[key], value), (
            f"ParameterValues override {key!r} did not round-trip"
        )


@serialisation_settings
@given(model_options_kwargs())
def test_model_options_round_trip(kwargs):
    """Model option dicts must survive to_config/from_config across SPM/SPMe/DFN."""
    model_cls = kwargs.pop("_cls")
    try:
        original = model_cls(options=kwargs)
    except (pybamm.OptionError, NotImplementedError, ValueError):
        # BatteryModelOptions validates illegal combos; discard rejected combos.
        assume(False)
    config = json.loads(json.dumps(original.to_config()))
    restored = pybamm.BaseModel.from_config(config)
    assert type(restored) is type(original)
    assert dict(restored.options) == dict(original.options)


def test_model_option_strategy_tracks_canonical():
    """``model_options_kwargs`` derives its values from ``possible_options``;
    this guards the parts that are still hand-maintained (the fuzzed key set and
    the value exclusions) so neither can silently drift from production.
    """
    canonical = pybamm.BatteryModelOptions({}).possible_options
    for key in _FUZZED_OPTION_KEYS:
        assert key in canonical, f"fuzzed option key {key!r} no longer exists"
    for key, excluded in _EXCLUDED_OPTION_VALUES.items():
        assert key in _FUZZED_OPTION_KEYS, f"exclusion for un-fuzzed key {key!r}"
        stale = excluded - set(canonical[key])
        assert not stale, f"stale exclusion(s) for {key!r}: {stale}"


class TestKnownSymbolSerialiserBugs:
    """Specific shapes where convert_symbol_to_json/from_json drop a non-children
    constructor arg. Each is xfail(strict=True): the test FAILS the day the fix
    lands, prompting removal of the entry. All referenced in tracking issue #5548.
    """

    @pytest.mark.parametrize(
        "tree",
        [
            pytest.param(
                pybamm.Variable("v", scale=pybamm.Scalar(2.0)),
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="tracked in #5548 — Variable.scale dropped by convert_symbol_to_json",
                ),
                id="Variable-with-non-default-scale",
            ),
            pytest.param(
                pybamm.Variable("v", reference=pybamm.Scalar(1.0)),
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="tracked in #5548 — Variable.reference dropped by convert_symbol_to_json",
                ),
                id="Variable-with-non-default-reference",
            ),
            pytest.param(
                pybamm.FullBroadcast(
                    pybamm.Scalar(1.0),
                    broadcast_domains={"primary": ["negative electrode"]},
                    name="custom",
                ),
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="tracked in #5548 — FullBroadcast custom name dropped",
                ),
                id="FullBroadcast-with-custom-name",
            ),
        ],
    )
    def test_known_failure(self, tree):
        restored = convert_symbol_from_json(convert_symbol_to_json(tree))
        assert restored.id == tree.id
