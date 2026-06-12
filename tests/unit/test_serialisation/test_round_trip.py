"""Generic JSON round-trip property tests for the public ``Serialise`` API.

Each fixture asserts that every constructor-arg-derived attribute survives
``to_config()`` / ``from_config()``. Adding a new init arg means adding a
fixture exercising it; if the dumper or loader drops it, the test fails.

Validated to catch the regression shape behind #5495 (bool->int coercion),
#5496 (dropped per-step ``temperature``) and #5497 (dropped nested
``root_method``).
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
import warnings
from datetime import datetime

import numpy as np
import pytest
from hypothesis import given, reject, settings
from hypothesis import strategies as st

import pybamm
from pybamm.expression_tree.operations.serialise import (
    Serialise,
    convert_symbol_from_json,
    convert_symbol_to_json,
)
from pybamm.models.full_battery_models.base_battery_model import BatteryModelOptions
from tests.unit.test_serialisation._helpers import (
    _experiments_equal,
    _solver_init_args_equal,
    _values_equal,
)


def _custom_model_with_rhs():
    m = pybamm.BaseModel("custom_rhs")
    c = pybamm.Variable("c", domain="negative particle")
    # Multi-term RHS so a loader-side expression mutation actually changes
    # the structural id (a single ``-c`` could mask sign flips).
    m.rhs = {c: -c * c + pybamm.Scalar(0.5)}
    m.initial_conditions = {c: pybamm.Scalar(0.8)}
    return m


def _custom_model_with_events():
    m = pybamm.BaseModel("custom_events")
    c = pybamm.Variable("c")
    m.rhs = {c: -c}
    m.initial_conditions = {c: pybamm.Scalar(1)}
    m.events = [
        pybamm.Event(
            "c hits half", c - pybamm.Scalar(0.5), pybamm.EventType.TERMINATION
        )
    ]
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
    # Every concrete step type, covering ``serialise_experiment`` dispatch.
    pybamm.Experiment([pybamm.step.current(1.0, duration=100)]),
    pybamm.Experiment([pybamm.step.voltage(4.2, duration=200)]),
    pybamm.Experiment([pybamm.step.power(10.0, duration=50)]),
    pybamm.Experiment([pybamm.step.c_rate(0.5, duration=300)]),
    pybamm.Experiment([pybamm.step.rest(duration=60)]),
    pybamm.Experiment([pybamm.step.resistance(0.1, duration=100)]),
    # Per-step temperature override (regression for #5496).
    pybamm.Experiment(
        [
            pybamm.step.current(1.0, duration=100, temperature=298.15),
            pybamm.step.rest(duration=60, temperature=313.15),
            pybamm.step.voltage(4.2, duration=200),
        ]
    ),
    pybamm.Experiment([pybamm.step.current(1.0, duration=3600, termination="3.0 V")]),
    # Omitted ``duration`` must round-trip so ``uses_default_duration`` stays True.
    pybamm.Experiment([pybamm.step.current(1.0, termination="3.0 V")]),
    pybamm.Experiment(
        [(pybamm.step.current(1.0, duration=100), pybamm.step.rest(duration=60))] * 3
    ),
    # Top-level experiment kwargs.
    pybamm.Experiment(
        [pybamm.step.current(1.0, duration=100)],
        period="30 seconds",
        temperature="25 oC",
        termination=["80% capacity", "2.5 V"],
    ),
    # Single-string termination guards against ``list("2.5 V")`` splitting chars.
    pybamm.Experiment(
        [pybamm.step.current(1.0, duration=100)],
        termination="2.5 V",
    ),
    # Per-step ``period``/``tags``/``description``/``start_time``/``skip_ok``.
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
    pybamm.ParameterValues(
        {"Temperature [K]": 298.15, "Negative electrode thickness [m]": 1e-4}
    ),
    # Bool-bearing entry guards against #5495-shaped coercion.
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


class TestExperimentSerialiseCanonical:
    """``serialise_experiment`` must not duplicate experiment-level defaults
    onto every step. The top-level ``period`` / ``temperature`` (broadcast by
    ``Experiment.process_steps``), processed empty ``tags``, and computed
    ``direction`` should all stay out of per-step dicts so the JSON keeps
    ``default vs explicit`` distinguishable.
    """

    def test_experiment_level_defaults_not_per_step(self):
        experiment = pybamm.Experiment(
            [pybamm.step.current(1.0, duration=100)],
            period="30 seconds",
            temperature="25 oC",
        )
        config = experiment.to_config()
        for step_dict in config["cycles"][0]:
            assert "period" not in step_dict
            assert "temperature" not in step_dict
            assert "tags" not in step_dict
            assert "direction" not in step_dict
        # Top-level defaults must still be present so the round-trip works.
        assert "period" in config
        assert "temperature" in config

    def test_per_step_period_overriding_default_is_kept(self):
        experiment = pybamm.Experiment(
            [
                (
                    pybamm.step.current(1.0, duration=100, period="10 seconds"),
                    pybamm.step.current(0.5, duration=200),
                )
            ],
            period="30 seconds",
        )
        config = experiment.to_config()
        # First step overrode the experiment default → key kept.
        assert "period" in config["cycles"][0][0]
        # Second step inherited → key dropped.
        assert "period" not in config["cycles"][0][1]


class TestExperimentSerialiseUnsupported:
    """Custom callables can't survive a JSON round-trip — fail fast.

    Both ``CustomTermination`` and ``CustomStepExplicit``/``CustomStepImplicit``
    hold user-supplied Python callables (event functions, current-value
    functions). Serialising them would silently lose the callable, so the
    serialiser must refuse with a clear error.
    """

    def test_custom_termination_raises(self):
        def stoich_cutoff(variables):
            return variables["Negative electrode stoichiometry"] - 0.1

        custom_term = pybamm.step.CustomTermination(
            name="Negative stoichiometry cut-off",
            event_function=stoich_cutoff,
        )
        experiment = pybamm.Experiment(
            [pybamm.step.current(1.0, duration=3600, termination=custom_term)]
        )
        with pytest.raises(NotImplementedError, match="CustomTermination"):
            experiment.to_config()

    def test_custom_step_explicit_raises(self):
        def current_function(variables):
            return 1.0

        experiment = pybamm.Experiment(
            [pybamm.step.CustomStepExplicit(current_function, duration=100)]
        )
        with pytest.raises(NotImplementedError, match="CustomStepExplicit"):
            experiment.to_config()


class TestExperimentFromConfigValidation:
    """Reject malformed inputs cleanly — ``from_config`` is public.

    A user-built dict may contain stringly-typed values (e.g. from YAML or a
    hand-rolled JSON file). Past loaders silently coerced them via
    ``bool(...)``, which flips ``"False"`` to ``True``.
    """

    def test_skip_ok_string_is_rejected(self):
        config = {
            "cycles": [
                [
                    {
                        "type": "current",
                        "value": 1.0,
                        "duration": 100,
                        "skip_ok": "False",
                    }
                ]
            ]
        }
        with pytest.raises(TypeError, match="skip_ok must be a bool"):
            pybamm.Experiment.from_config(config)

    @pytest.mark.parametrize("skip_ok", [True, False])
    def test_json_bool_skip_ok_round_trips(self, skip_ok):
        config = {
            "cycles": [
                [
                    {
                        "type": "current",
                        "value": 1.0,
                        "duration": 100,
                        "skip_ok": skip_ok,
                    }
                ]
            ]
        }
        restored = pybamm.Experiment.from_config(json.loads(json.dumps(config)))
        assert restored.steps[0].skip_ok is skip_ok


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

        Compares structural ids of *both* keys and values, so a loader that
        flips a sign or replaces an initial condition would fail this test.

        Uses ``Serialise._json_encoder`` because the dict returned by
        ``BaseModel.to_json`` carries values (``EventType``, numpy scalars) that
        only encode through the project's custom encoder.
        """
        data = json.loads(json.dumps(model.to_json(), default=Serialise._json_encoder))
        restored = pybamm.BaseModel.from_json(data)

        assert restored.name == model.name

        def _id_dict(d):
            return {k.id: v.id for k, v in d.items()}

        assert _id_dict(restored.rhs) == _id_dict(model.rhs)
        assert _id_dict(restored.initial_conditions) == _id_dict(
            model.initial_conditions
        )
        assert len(restored.events) == len(model.events)
        events_by_name = {ev.name: ev for ev in restored.events}
        for orig_ev in model.events:
            assert orig_ev.name in events_by_name
            new_ev = events_by_name[orig_ev.name]
            assert orig_ev.event_type == new_ev.event_type
            assert orig_ev.expression.id == new_ev.expression.id


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


# Generative option round-trip fuzzing. JSON has no tuple type, so a tuple option
# (e.g. "particle phases": ("2", "1")) deserialises as a list, which validation
# rejects. Fuzzes the build-free to_json/from_json path; the deterministic
# fixtures below cover to_config/from_config (which builds the model).

# Option registry, sourced from pybamm so the strategy never hardcodes values.
_POSSIBLE_OPTIONS = BatteryModelOptions({}).possible_options

_STRING_VALUES = {
    key: [v for v in values if isinstance(v, str)]
    for key, values in _POSSIBLE_OPTIONS.items()
}

# Drawable keys. "dimensionality" (the only int option) is excluded: a non-zero
# value needs companion geometry options the random strategy can't assemble; its
# int round-trip is covered by test_int_option_survives_json_round_trip_as_int.
_STRING_OPTION_KEYS = sorted(
    key for key, values in _STRING_VALUES.items() if key != "dimensionality" and values
)

# Options accepting a 2-tuple of strings; mirrors the list in BatteryModelOptions
# (test_tuple_capable_keys_match_validation_source guards drift).
_TUPLE_CAPABLE_KEYS = frozenset(
    {
        "diffusivity",
        "exchange-current density",
        "intercalation kinetics",
        "interface utilisation",
        "lithium plating",
        "loss of active material",
        "number of MSMR reactions",
        "open-circuit potential",
        "particle",
        "particle mechanics",
        "particle phases",
        "particle size",
        "SEI",
        "SEI on cracks",
        "stress-induced diffusion",
    }
)

# SPM matches the original regression; DFN has a broader option surface.
_OPTION_MODEL_CLASSES = [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN]


def _assert_param_matches_options(model):
    """Assert a loaded lithium-ion model's ``param`` matches its restored options.

    Guards #5598; lead-acid ``param`` is options-independent, so skip it.
    """
    if isinstance(model, pybamm.lithium_ion.BaseModel):
        assert dict(model.param.options) == dict(model.options), (
            f"param built from {dict(model.param.options)!r} != restored options "
            f"{dict(model.options)!r}; load_custom_model must rebuild param."
        )


def test_rebuild_param_syncs_param_with_reassigned_options():
    """The ``_rebuild_param`` hook resyncs ``param`` to ``options``.

    Deserialisation assigns ``options`` after construction; the setter leaves
    ``param`` untouched, so the model-owned hook must rebuild it.
    """
    # A complete, valid two-phase options set (as a load path would restore).
    full_opts = dict(pybamm.lithium_ion.SPM({"particle phases": ("2", "1")}).options)

    model = pybamm.lithium_ion.SPM(build=False)  # default: single particle phase
    model.options = full_opts
    # The setter alone leaves param built from the default (single-phase) options.
    assert dict(model.param.options) != dict(model.options)
    model._rebuild_param()
    assert dict(model.param.options) == dict(model.options)


@st.composite
def valid_option_dicts(draw):
    """Draw a small registry-valid option dict, biased to exercise tuples."""
    keys = draw(
        st.lists(
            st.sampled_from(_STRING_OPTION_KEYS),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    options = {}
    for key in keys:
        choices = _STRING_VALUES[key]
        if key in _TUPLE_CAPABLE_KEYS and draw(st.booleans()):
            options[key] = (
                draw(st.sampled_from(choices)),
                draw(st.sampled_from(choices)),
            )
        else:
            options[key] = draw(st.sampled_from(choices))
    return options


def test_tuple_capable_keys_match_validation_source():
    """``_TUPLE_CAPABLE_KEYS`` must match the tuple-capable list in
    BatteryModelOptions -- AST-extracted so it can't silently drift.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(BatteryModelOptions)))
    candidates = [
        {
            elt.value
            for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
    ]
    # The tuple-capable list is the one literal containing these marker keys.
    matches = [c for c in candidates if {"particle phases", "SEI on cracks"} <= c]
    assert len(matches) == 1, (
        "Could not uniquely locate the tuple-capable option list in "
        "BatteryModelOptions source; update the extraction in this test."
    )
    assert matches[0] == set(_TUPLE_CAPABLE_KEYS), (
        "Tuple-capable options in pybamm have changed. Update "
        "_TUPLE_CAPABLE_KEYS so option fuzzing keeps generating tuple values "
        f"for every such key. Source: {sorted(matches[0])}, "
        f"test constant: {sorted(_TUPLE_CAPABLE_KEYS)}."
    )
    assert _TUPLE_CAPABLE_KEYS <= set(_POSSIBLE_OPTIONS), (
        "_TUPLE_CAPABLE_KEYS contains keys absent from the option registry."
    )


class TestModelOptionsRoundTrip:
    """Any valid option dict must survive a JSON round-trip with exact types."""

    @pytest.mark.parametrize(
        "model_cls", _OPTION_MODEL_CLASSES, ids=lambda c: c.__name__
    )
    @settings(max_examples=100, deadline=None)
    @given(options=valid_option_dicts())
    def test_options_survive_json_round_trip(self, model_cls, options):
        try:
            # Construction is the precondition, not the SUT; pass a copy
            # (pybamm mutates the dict).
            model = model_cls(dict(options), build=False)
        except (pybamm.OptionError, NotImplementedError):
            # Invalid/unsupported option set -> discard. Any other exception is a
            # real bug that must surface (a broad except here once hid an MSMR
            # int('none') crash).
            reject()

        expected = dict(model.options)

        via_json = pybamm.BaseModel.from_json(
            json.loads(json.dumps(model.to_json(), default=Serialise._json_encoder))
        )
        assert _values_equal(expected, dict(via_json.options)), (
            f"{model_cls.__name__} options lost or mutated via "
            f"to_json/from_json for input {options!r}: "
            f"{expected!r} -> {dict(via_json.options)!r}"
        )
        # Options surviving isn't enough -- the derived param must match too.
        _assert_param_matches_options(via_json)


def test_int_option_survives_json_round_trip_as_int():
    """``dimensionality`` (the only int option) must round-trip as an int, not be
    coerced to/from bool (the #5495 shape, ``True == 1``). Bool-like options are
    strings "true"/"false" (pybamm rejects Python bools), already fuzzed; a
    non-zero value needs companion geometry options to construct.
    """
    options = {
        "dimensionality": 1,
        "current collector": "potential pair",
        "cell geometry": "pouch",
    }
    model = pybamm.lithium_ion.DFN(dict(options), build=False)

    via_json = pybamm.BaseModel.from_json(
        json.loads(json.dumps(model.to_json(), default=Serialise._json_encoder))
    )
    restored = via_json.options["dimensionality"]
    assert restored == 1
    assert isinstance(restored, int) and not isinstance(restored, bool)


# Buildable tuple options that must round-trip through both load paths --
# deterministic, always-run coverage of the bug shape.
_TUPLE_OPTION_FIXTURES = [
    (pybamm.lithium_ion.SPM, "particle phases", ("2", "1")),
    (pybamm.lithium_ion.SPM, "SEI", ("none", "constant")),
    (pybamm.lithium_ion.DFN, "particle size", ("single", "distribution")),
    (pybamm.lithium_ion.DFN, "particle", ("Fickian diffusion", "uniform profile")),
]


class TestTupleOptionRoundTripRegression:
    @pytest.mark.parametrize(
        "model_cls,key,value",
        _TUPLE_OPTION_FIXTURES,
        ids=["SPM-particle phases", "SPM-SEI", "DFN-particle size", "DFN-particle"],
    )
    def test_tuple_option_survives_both_paths(self, model_cls, key, value):
        model = model_cls({key: value}, build=False)

        via_json = pybamm.BaseModel.from_json(
            json.loads(json.dumps(model.to_json(), default=Serialise._json_encoder))
        )
        assert via_json.options[key] == value
        assert isinstance(via_json.options[key], tuple)
        # Composite "particle phases" is the case (param stuck single-phase).
        _assert_param_matches_options(via_json)

        via_config = pybamm.BaseModel.from_config(
            json.loads(json.dumps(model.to_config()))
        )
        assert via_config.options[key] == value
        assert isinstance(via_config.options[key], tuple)


# Discretised-model round-trip (save_model/load_model): previously only checked
# "load and solve". These verify structural identity and solution equivalence
# (mesh transitively: same equations + solution => same mesh).

_DISCRETISED_MODEL_CLASSES = [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN]


def _discretised_model(model_cls, options=None):
    """Build, parameterise and discretise a model; return it and its mesh
    (the mesh is needed by ``serialise_model`` for named variables).
    """
    model = model_cls(options)
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.process_model(model)
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    return model, mesh


def _round_trip_discretised(model, mesh):
    """Serialise a discretised model and load it back, via dict (no temp file)."""
    serialised = Serialise().serialise_model(model, mesh)
    return Serialise().load_model(json.loads(json.dumps(serialised)))


@pytest.mark.parametrize(
    "model_cls", _DISCRETISED_MODEL_CLASSES, ids=lambda c: c.__name__
)
class TestDiscretisedModelRoundTrip:
    def test_derived_state_matches_original(self, model_cls):
        """Concatenated equations, mass matrix, bounds and options must match."""
        model, mesh = _discretised_model(model_cls)
        loaded = _round_trip_discretised(model, mesh)

        for attr in (
            "concatenated_rhs",
            "concatenated_algebraic",
            "concatenated_initial_conditions",
            "mass_matrix",
        ):
            original = getattr(model, attr)
            restored = getattr(loaded, attr)
            assert original.id == restored.id, (
                f"{model_cls.__name__}.{attr} changed across the discretised "
                f"round-trip: {original.id} != {restored.id}"
            )

        assert all(
            np.array_equal(a, b)
            for a, b in zip(model.bounds, loaded.bounds, strict=True)
        ), f"{model_cls.__name__} bounds changed across the discretised round-trip"
        assert dict(model.options) == dict(loaded.options)

        # Events are serialised too -- verify them structurally rather than
        # relying on the solution check.
        assert len(model.events) == len(loaded.events)
        for original_event, restored_event in zip(
            model.events, loaded.events, strict=True
        ):
            assert original_event.name == restored_event.name
            assert original_event.event_type == restored_event.event_type
            assert original_event.expression.id == restored_event.expression.id, (
                f"{model_cls.__name__} event {original_event.name!r} changed "
                "across the discretised round-trip"
            )

    def test_solution_matches_original(self, model_cls):
        """The reloaded model must solve to the same answer as the original."""
        model, mesh = _discretised_model(model_cls)
        # Serialise before solving so any solve-time caching cannot leak in.
        loaded = _round_trip_discretised(model, mesh)

        t_eval = [0, 3600]
        original_solution = model.default_solver.solve(model, t_eval)
        loaded_solution = loaded.default_solver.solve(loaded, t_eval)

        assert np.allclose(
            original_solution.y, loaded_solution.y, rtol=1e-6, atol=1e-8
        ), (
            f"{model_cls.__name__} state trajectory diverged after a "
            "discretised round-trip"
        )
        assert np.allclose(
            original_solution["Voltage [V]"].entries,
            loaded_solution["Voltage [V]"].entries,
            rtol=1e-6,
            atol=1e-8,
        ), f"{model_cls.__name__} Voltage [V] diverged after a discretised round-trip"


def test_nondefault_tuple_option_survives_discretised_round_trip():
    """A tuple-valued option must survive the discretised save/load path too.

    The parametrised tests above only use default options; this guards
    ``load_model``'s list->tuple conversion. ``particle`` is used because it
    discretises with the default parameter set (composite ``particle phases``
    would need a phase-prefixed one).
    """
    options = {"particle": ("Fickian diffusion", "uniform profile")}
    model, mesh = _discretised_model(pybamm.lithium_ion.DFN, options)
    loaded = _round_trip_discretised(model, mesh)

    assert loaded.options["particle"] == ("Fickian diffusion", "uniform profile")
    assert isinstance(loaded.options["particle"], tuple)
    _assert_param_matches_options(loaded)
