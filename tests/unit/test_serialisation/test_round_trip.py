"""Generic JSON round-trip property tests for the public ``Serialise`` API.

Each fixture asserts that every constructor-arg-derived attribute survives
``to_config()`` / ``from_config()``. Adding a new init arg means adding a
fixture exercising it; if the dumper or loader drops it, the test fails.

Validated to catch the regression shape behind #5495 (bool->int coercion),
#5496 (dropped per-step ``temperature``) and #5497 (dropped nested
``root_method``). ``BaseModel`` and symbol-level (``convert_symbol_*``)
round-trips are out of scope and warrant dedicated coverage.
"""

from __future__ import annotations

import inspect
import json
import warnings

import numpy as np
import pytest

import pybamm

_MISSING = object()


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
            and all(_values_equal(x, y) for x, y in zip(a, b))
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


@pytest.mark.parametrize(
    "solver", SOLVER_FIXTURES, ids=lambda s: f"{type(s).__name__}"
)
def test_solver_round_trip_preserves_init_args(solver):
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


# Per-step attributes that must round-trip. ``value`` is read on subclasses
# that have it (Current/Voltage/Power/CRate); Rest is treated separately.
_STEP_ATTRS = ("duration", "value", "temperature", "termination")


def _experiment_steps_equal(a, b):
    if len(a.steps) != len(b.steps):
        return False
    for orig, new in zip(a.steps, b.steps):
        if type(orig) is not type(new):
            return False
        for attr in _STEP_ATTRS:
            if not hasattr(orig, attr) and not hasattr(new, attr):
                continue
            if not _values_equal(getattr(orig, attr, None), getattr(new, attr, None)):
                return False
    return True


EXPERIMENT_FIXTURES = [
    # Every concrete step type with a non-default value, exercising the
    # ``serialise_experiment`` step dispatch.
    pybamm.Experiment([pybamm.step.current(1.0, duration=100)]),
    pybamm.Experiment([pybamm.step.voltage(4.2, duration=200)]),
    pybamm.Experiment([pybamm.step.power(10.0, duration=50)]),
    pybamm.Experiment([pybamm.step.c_rate(0.5, duration=300)]),
    pybamm.Experiment([pybamm.step.rest(duration=60)]),
    # Per-step temperature override (regression for #5496).
    pybamm.Experiment(
        [
            pybamm.step.current(1.0, duration=100, temperature=298.15),
            pybamm.step.rest(duration=60, temperature=313.15),
            pybamm.step.voltage(4.2, duration=200),
        ]
    ),
    # Termination conditions on a step.
    pybamm.Experiment(
        [pybamm.step.current(1.0, duration=3600, termination="3.0 V")]
    ),
    # Multi-cycle experiment.
    pybamm.Experiment(
        [(pybamm.step.current(1.0, duration=100), pybamm.step.rest(duration=60))]
        * 3
    ),
]


@pytest.mark.parametrize(
    "experiment",
    EXPERIMENT_FIXTURES,
    ids=lambda e: "+".join(type(s).__name__ for s in e.steps[:3]),
)
def test_experiment_round_trip_preserves_step_state(experiment):
    """Every step attribute reachable from the constructor must survive."""
    config = json.loads(json.dumps(experiment.to_config()))
    restored = pybamm.Experiment.from_config(config)

    assert _experiment_steps_equal(experiment, restored), (
        "Experiment failed round-trip: at least one step attribute was lost. "
        f"Original steps: {experiment.steps!r}. "
        f"Restored steps: {restored.steps!r}."
    )


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


@pytest.mark.parametrize(
    "parameter_values",
    PARAMETER_VALUES_FIXTURES,
    ids=lambda pv: f"keys={len(pv)}",
)
def test_parameter_values_round_trip_preserves_entries(parameter_values):
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


SPATIAL_METHOD_FIXTURES = [
    pybamm.FiniteVolume(),
    pybamm.ZeroDimensionalSpatialMethod(),
]


@pytest.mark.parametrize(
    "spatial_method",
    SPATIAL_METHOD_FIXTURES,
    ids=lambda sm: type(sm).__name__,
)
def test_spatial_method_round_trip(spatial_method):
    """Class identity and options must survive round-trip."""
    config = json.loads(json.dumps(spatial_method.to_config()))
    restored = pybamm.SpatialMethod.from_config(config)
    assert type(restored) is type(spatial_method)
    assert _values_equal(restored.options, spatial_method.options)


MESH_GENERATOR_FIXTURES = [
    pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
    pybamm.MeshGenerator(pybamm.SubMesh0D),
]


@pytest.mark.parametrize(
    "mesh_generator",
    MESH_GENERATOR_FIXTURES,
    ids=lambda mg: mg.submesh_type.__name__,
)
def test_mesh_generator_round_trip(mesh_generator):
    """Submesh class identity must survive round-trip."""
    config = json.loads(json.dumps(mesh_generator.to_config()))
    restored = pybamm.MeshGenerator.from_config(config)
    assert type(restored) is type(mesh_generator)
    assert restored.submesh_type is mesh_generator.submesh_type


SUBMESH_FIXTURES = [
    pybamm.SubMesh0D,
    pybamm.Uniform1DSubMesh,
]


@pytest.mark.parametrize(
    "submesh_class",
    SUBMESH_FIXTURES,
    ids=lambda c: c.__name__,
)
def test_submesh_class_round_trip(submesh_class):
    """``SubMesh.to_config`` returns a class identifier that must round-trip."""
    config = json.loads(json.dumps(submesh_class.to_config()))
    restored = submesh_class.from_config(config)
    assert restored is submesh_class
