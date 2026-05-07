"""Generic JSON round-trip property tests for the public ``Serialise`` API.

Each fixture asserts that every constructor-arg-derived attribute survives
``to_config()`` / ``from_config()``. Adding a new init arg means adding a
fixture exercising it; if the dumper or loader drops it, the test fails.

Validated to catch the regression shape behind #5495 (bool->int coercion),
#5496 (dropped per-step ``temperature``) and #5497 (dropped nested
``root_method``).
"""

from __future__ import annotations

import inspect
import json
import warnings
from datetime import datetime

import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.operations.serialise import (
    Serialise,
    convert_symbol_from_json,
    convert_symbol_to_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MISSING = object()

# Per-step attributes that must round-trip. ``_experiments_equal`` skips
# attrs missing on both sides, so ``value`` is fine for the Rest case
# (which has no ``value``).
_STEP_ATTRS = (
    "duration",
    "value",
    "temperature",
    "termination",
    "period",
    "tags",
    "description",
    "direction",
    "start_time",
    "skip_ok",
)
_EXPERIMENT_TOP_LEVEL_ATTRS = ("period", "temperature", "termination_string")


def _attr(obj, name):
    """Read constructor-arg-named attribute, trying both ``name`` and ``_name``."""
    for candidate in (name, f"_{name}"):
        if hasattr(obj, candidate):
            return getattr(obj, candidate)
    return _MISSING


def _values_equal(a, b):
    """Strict equality with bool-vs-int discrimination and float tolerance.

    Recurses into dicts, lists/tuples, and ``BaseSolver`` instances so a
    nested ``root_method`` is compared structurally.
    """
    if isinstance(a, pybamm.BaseSolver) or isinstance(b, pybamm.BaseSolver):
        return _solver_init_args_equal(a, b)
    # bool must match bool exactly (catches #5495-shaped regressions where
    # True silently round-trips as 1).
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return a == pytest.approx(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_values_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, type(a))
            and len(a) == len(b)
            and all(_values_equal(x, y) for x, y in zip(a, b, strict=False))
        )
    return a == b


def _solver_init_args_equal(a, b):
    """Two solvers are equal if same class and every ``__init__`` arg matches."""
    if type(a) is not type(b):
        return False
    sig = inspect.signature(type(a).__init__)
    for name in sig.parameters:
        if name == "self":
            continue
        va = _attr(a, name)
        vb = _attr(b, name)
        if va is _MISSING and vb is _MISSING:
            continue
        if not _values_equal(va, vb):
            return False
    return True


def _experiments_equal(a, b):
    """Equality across both top-level experiment attrs and per-step attrs."""
    for attr in _EXPERIMENT_TOP_LEVEL_ATTRS:
        if not _values_equal(getattr(a, attr, None), getattr(b, attr, None)):
            return False
    if len(a.steps) != len(b.steps):
        return False
    for orig, new in zip(a.steps, b.steps, strict=True):
        if type(orig) is not type(new):
            return False
        for attr in _STEP_ATTRS:
            if not hasattr(orig, attr) and not hasattr(new, attr):
                continue
            if not _values_equal(getattr(orig, attr, None), getattr(new, attr, None)):
                return False
    return True


def _custom_model_with_rhs():
    m = pybamm.BaseModel("custom_rhs")
    c = pybamm.Variable("c", domain="negative particle")
    m.rhs = {c: -c}
    m.initial_conditions = {c: pybamm.Scalar(1)}
    return m


def _custom_model_with_events():
    m = pybamm.BaseModel("custom_events")
    c = pybamm.Variable("c")
    m.rhs = {c: -c}
    m.initial_conditions = {c: pybamm.Scalar(1)}
    m.events = [pybamm.Event("c hits zero", c, pybamm.EventType.TERMINATION)]
    return m


def _build_symbol_fixtures():
    a = pybamm.Variable("a")
    b = pybamm.Parameter("b")
    return [
        pybamm.Scalar(3.14),
        pybamm.Scalar(np.inf),
        pybamm.Scalar(-np.inf),
        pybamm.Scalar(np.nan),
        pybamm.Parameter("Cell capacity [A.h]"),
        pybamm.InputParameter("Current"),
        pybamm.Variable("c"),
        pybamm.Time(),
        pybamm.SpatialVariable("x", domain=["negative electrode"]),
        a + b,
        a * b - pybamm.Scalar(2),
        -a,
        pybamm.PrimaryBroadcast(pybamm.Scalar(1), "negative electrode"),
        pybamm.FullBroadcast(
            pybamm.Scalar(1),
            "negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        ),
        pybamm.Interpolant(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]), a),
        pybamm.FunctionParameter("f", {"a": a}),
    ]


# ---------------------------------------------------------------------------
# Fixtures (test data)
# ---------------------------------------------------------------------------
SOLVER_FIXTURES = [
    pybamm.ScipySolver(rtol=1e-5, atol=1e-7),
    pybamm.CasadiSolver(rtol=1e-4, mode="fast", root_method="casadi"),
    # IDAKLU exercises bool options (#5495), nested root_method (#5497),
    # and tolerance fields together.
    pybamm.IDAKLUSolver(
        root_method="casadi",
        root_tol=1e-9,
        options={"print_stats": True, "compile": False},
    ),
    pybamm.IDAKLUSolver(root_method="nonlinear_solver"),
    pybamm.IDAKLUSolver(root_method="lm"),
    pybamm.CompositeSolver(
        [pybamm.ScipySolver(rtol=1e-5), pybamm.CasadiSolver(mode="fast")]
    ),
]


EXPERIMENT_FIXTURES = [
    # Every concrete step type with a non-default value, exercising the
    # ``serialise_experiment`` step dispatch.
    pybamm.Experiment([pybamm.step.current(1.0, duration=100)]),
    pybamm.Experiment([pybamm.step.voltage(4.2, duration=200)]),
    pybamm.Experiment([pybamm.step.power(10.0, duration=50)]),
    pybamm.Experiment([pybamm.step.c_rate(0.5, duration=300)]),
    pybamm.Experiment([pybamm.step.rest(duration=60)]),
    # Resistance step type (regression for missing entry in
    # ``serialise_experiment`` step_func_map).
    pybamm.Experiment([pybamm.step.resistance(0.1, duration=100)]),
    # Per-step temperature override (regression for #5496).
    pybamm.Experiment(
        [
            pybamm.step.current(1.0, duration=100, temperature=298.15),
            pybamm.step.rest(duration=60, temperature=313.15),
            pybamm.step.voltage(4.2, duration=200),
        ]
    ),
    # Termination conditions on a step.
    pybamm.Experiment([pybamm.step.current(1.0, duration=3600, termination="3.0 V")]),
    # Multi-cycle experiment.
    pybamm.Experiment(
        [(pybamm.step.current(1.0, duration=100), pybamm.step.rest(duration=60))] * 3
    ),
    # Top-level experiment kwargs (period / temperature / termination) — guards
    # against the experiment-level fields being dropped during ``to_config``.
    pybamm.Experiment(
        [pybamm.step.current(1.0, duration=100)],
        period="30 seconds",
        temperature="25 oC",
        termination=["80% capacity", "2.5 V"],
    ),
    # Top-level termination as a single string — guards against ``list("2.5 V")``
    # exploding the input into individual chars during the round-trip.
    pybamm.Experiment(
        [pybamm.step.current(1.0, duration=100)],
        termination="2.5 V",
    ),
    # Per-step ``period`` override and other user-settable BaseStep fields:
    # ``tags``, ``description``, ``start_time``, ``skip_ok``.
    pybamm.Experiment(
        [
            pybamm.step.current(
                1.0,
                duration=100,
                period="10 seconds",
                tags=["formation", "first"],
                description="Constant current charge",
                start_time=datetime(2024, 1, 1, 12, 0, 0),
                skip_ok=False,
            ),
            pybamm.step.rest(duration=60, period="5 seconds"),
        ]
    ),
]


PARAMETER_VALUES_FIXTURES = [
    # Plain scalars (most common case).
    pybamm.ParameterValues(
        {"Temperature [K]": 298.15, "Negative electrode thickness [m]": 1e-4}
    ),
    # Bool-bearing entry — guards against #5495-shaped coercion in the
    # ParameterValues serialisation path.
    pybamm.ParameterValues(
        {"Temperature [K]": 298.15, "Some flag": True, "Other flag": False}
    ),
]


SPATIAL_METHOD_FIXTURES = [
    pybamm.FiniteVolume(),
    pybamm.ZeroDimensionalSpatialMethod(),
]


MESH_GENERATOR_FIXTURES = [
    pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
    pybamm.MeshGenerator(pybamm.SubMesh0D),
]


SUBMESH_FIXTURES = [
    pybamm.SubMesh0D,
    pybamm.Uniform1DSubMesh,
]


BUILTIN_MODEL_FIXTURES = [
    pybamm.lithium_ion.SPM(),
    pybamm.lithium_ion.SPM(options={"thermal": "lumped"}),
    pybamm.lithium_ion.DFN(),
]


CUSTOM_MODEL_FIXTURES = [
    _custom_model_with_rhs(),
    _custom_model_with_events(),
]


SYMBOL_FIXTURES = _build_symbol_fixtures()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSolverRoundTrip:
    @pytest.mark.parametrize(
        "solver", SOLVER_FIXTURES, ids=lambda s: f"{type(s).__name__}"
    )
    def test_preserves_init_args(self, solver):
        """Every constructor-arg-derived attribute must survive JSON round-trip.

        This is the structural fix for the #5495 / #5497 bug class.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = solver.to_config()

        restored = pybamm.BaseSolver.from_config(json.loads(json.dumps(config)))

        assert _solver_init_args_equal(solver, restored), (
            f"{type(solver).__name__} failed round-trip: at least one "
            f"__init__-derived attribute was lost or mutated. Compare "
            f"{vars(solver)!r} against {vars(restored)!r}."
        )


class TestExperimentRoundTrip:
    @pytest.mark.parametrize(
        "experiment",
        EXPERIMENT_FIXTURES,
        ids=lambda e: "+".join(type(s).__name__ for s in e.steps[:3]),
    )
    def test_preserves_step_state(self, experiment):
        """Every constructor-reachable attribute (top-level and per-step) survives."""
        config = json.loads(json.dumps(experiment.to_config()))
        restored = pybamm.Experiment.from_config(config)

        assert _experiments_equal(experiment, restored), (
            "Experiment failed round-trip: at least one attribute was lost. "
            f"Original: period={experiment.period!r}, temperature={experiment.temperature!r}, "
            f"termination={experiment.termination_string!r}, steps={experiment.steps!r}. "
            f"Restored: period={restored.period!r}, temperature={restored.temperature!r}, "
            f"termination={restored.termination_string!r}, steps={restored.steps!r}."
        )


class TestParameterValuesRoundTrip:
    @pytest.mark.parametrize(
        "parameter_values",
        PARAMETER_VALUES_FIXTURES,
        ids=lambda pv: f"keys={len(pv)}",
    )
    def test_preserves_entries(self, parameter_values):
        """Every key/value pair must survive round-trip with strict types."""
        data = json.loads(json.dumps(parameter_values.to_json()))
        restored = pybamm.ParameterValues.from_json(data)

        assert dict(parameter_values).keys() == dict(restored).keys(), (
            "ParameterValues round-trip lost or added keys."
        )
        for key in parameter_values:
            assert _values_equal(parameter_values[key], restored[key]), (
                f"ParameterValues[{key!r}] failed round-trip: "
                f"{parameter_values[key]!r} vs {restored[key]!r} "
                f"(types {type(parameter_values[key]).__name__} vs "
                f"{type(restored[key]).__name__})"
            )


class TestSpatialMethodRoundTrip:
    @pytest.mark.parametrize(
        "spatial_method",
        SPATIAL_METHOD_FIXTURES,
        ids=lambda sm: type(sm).__name__,
    )
    def test_preserves_class_and_options(self, spatial_method):
        """Class identity and options must survive round-trip."""
        config = json.loads(json.dumps(spatial_method.to_config()))
        restored = pybamm.SpatialMethod.from_config(config)
        assert type(restored) is type(spatial_method)
        assert _values_equal(restored.options, spatial_method.options)


class TestMeshGeneratorRoundTrip:
    @pytest.mark.parametrize(
        "mesh_generator",
        MESH_GENERATOR_FIXTURES,
        ids=lambda mg: mg.submesh_type.__name__,
    )
    def test_preserves_submesh_class(self, mesh_generator):
        """Submesh class identity must survive round-trip."""
        config = json.loads(json.dumps(mesh_generator.to_config()))
        restored = pybamm.MeshGenerator.from_config(config)
        assert type(restored) is type(mesh_generator)
        assert restored.submesh_type is mesh_generator.submesh_type


class TestSubMeshClassRoundTrip:
    @pytest.mark.parametrize(
        "submesh_class",
        SUBMESH_FIXTURES,
        ids=lambda c: c.__name__,
    )
    def test_class_identifier(self, submesh_class):
        """``SubMesh.to_config`` returns a class identifier that must round-trip."""
        config = json.loads(json.dumps(submesh_class.to_config()))
        restored = submesh_class.from_config(config)
        assert restored is submesh_class


class TestBuiltinModelRoundTrip:
    @pytest.mark.parametrize(
        "model",
        BUILTIN_MODEL_FIXTURES,
        ids=lambda m: f"{type(m).__name__}-{m.options.get('thermal', 'default')}",
    )
    def test_preserves_options(self, model):
        """Built-in model class identity and options dict must survive round-trip."""
        config = json.loads(json.dumps(model.to_config()))
        restored = pybamm.BaseModel.from_config(config)
        assert type(restored) is type(model)
        assert dict(restored.options) == dict(model.options)


class TestCustomModelRoundTrip:
    @pytest.mark.parametrize(
        "model",
        CUSTOM_MODEL_FIXTURES,
        ids=lambda m: m.name,
    )
    def test_preserves_rhs_and_events(self, model):
        """Custom model rhs / initial_conditions / events must survive round-trip.

        Uses ``Serialise._json_encoder`` because the dict returned by
        ``BaseModel.to_json`` carries values (``EventType``, numpy scalars) that
        only encode through the project's custom encoder.
        """
        data = json.loads(json.dumps(model.to_json(), default=Serialise._json_encoder))
        restored = pybamm.BaseModel.from_json(data)

        assert restored.name == model.name

        def _id_set(d):
            return {k.id for k in d}

        assert _id_set(restored.rhs) == _id_set(model.rhs)
        assert _id_set(restored.initial_conditions) == _id_set(model.initial_conditions)
        assert len(restored.events) == len(model.events)
        for orig_ev, new_ev in zip(model.events, restored.events, strict=True):
            assert orig_ev.name == new_ev.name
            assert orig_ev.event_type == new_ev.event_type


class TestSymbolRoundTrip:
    @pytest.mark.parametrize(
        "symbol",
        SYMBOL_FIXTURES,
        ids=lambda s: type(s).__name__,
    )
    def test_preserves_structure(self, symbol):
        """Every Symbol type must round-trip with the same class and structural id.

        NaN is treated specially (NaN != NaN), but a Scalar(NaN) still has a
        well-defined ``.id`` after round-trip if reconstruction preserves it.
        """
        json_dict = convert_symbol_to_json(symbol)
        restored = convert_symbol_from_json(json.loads(json.dumps(json_dict)))

        assert type(restored) is type(symbol), (
            f"{type(symbol).__name__} round-tripped as {type(restored).__name__}"
        )
        if isinstance(symbol, pybamm.Scalar) and (
            np.isnan(symbol.value) or np.isinf(symbol.value)
        ):
            if np.isnan(symbol.value):
                assert np.isnan(restored.value)
            else:
                assert restored.value == symbol.value
            return
        assert restored.id == symbol.id, (
            f"{type(symbol).__name__} structural id changed across round-trip: "
            f"{symbol!r} -> {restored!r}"
        )
