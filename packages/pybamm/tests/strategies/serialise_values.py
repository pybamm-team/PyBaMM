"""Hypothesis strategies for the top-level pybamm.Serialise.* surface.

Each strategy generates valid kwargs for a constructor exercised by the
existing parametrized round-trip tests. The property tests under
``tests/unit/test_serialisation/test_round_trip_property.py`` consume
these strategies.

Generated dicts include a ``_cls`` key naming the concrete class to
construct — the property test pops it before passing kwargs to the
constructor.
"""

from __future__ import annotations

import hypothesis.strategies as st

import pybamm
from pybamm.expression_tree.operations.serialise import _experiment_step_factories

_POSITIVE_TOL = st.floats(
    min_value=1e-12, max_value=1e-2, allow_nan=False, allow_infinity=False
)

# Concrete BaseSolver subclasses the solver fuzzer (and the coverage meta-test)
# draw from -- single source of truth so coverage can't drift from the fuzzer.
_SOLVER_CLASSES = (
    pybamm.IDAKLUSolver,
    pybamm.CasadiSolver,
    pybamm.ScipySolver,
    pybamm.AlgebraicSolver,
    pybamm.CasadiAlgebraicSolver,
    pybamm.NonlinearSolver,
    pybamm.CompositeSolver,
)


def _root_method() -> st.SearchStrategy:
    """root_method can be a string, None, or a nested BaseSolver (#5497)."""
    return st.one_of(
        st.none(),
        st.sampled_from(["casadi", "lm", "hybr"]),
        st.builds(
            pybamm.AlgebraicSolver,
            method=st.sampled_from(["lm", "hybr"]),
            tol=_POSITIVE_TOL,
        ),
    )


def solver_kwargs() -> st.SearchStrategy[dict]:
    """Pick a concrete solver class and generate valid kwargs for it.

    Each class branch generates only the kwargs its constructor accepts.
    CompositeSolver wraps two randomly chosen sub-solvers from the simple
    (non-composite) subset of _SOLVER_CLASSES.
    """

    _SIMPLE_SOLVER_CLASSES = [
        c for c in _SOLVER_CLASSES if c is not pybamm.CompositeSolver
    ]

    @st.composite
    def _strategy(draw):
        cls = draw(st.sampled_from(list(_SOLVER_CLASSES)))
        kwargs: dict = {"_cls": cls}
        if cls is pybamm.IDAKLUSolver:
            kwargs["rtol"] = draw(_POSITIVE_TOL)
            kwargs["atol"] = draw(_POSITIVE_TOL)
            kwargs["root_method"] = draw(_root_method())
            kwargs["options"] = draw(
                st.fixed_dictionaries(
                    {
                        "print_stats": st.booleans(),
                        "compile": st.booleans(),
                    }
                )
            )
        elif cls in (pybamm.CasadiSolver, pybamm.ScipySolver):
            kwargs["rtol"] = draw(_POSITIVE_TOL)
            kwargs["atol"] = draw(_POSITIVE_TOL)
        elif cls is pybamm.AlgebraicSolver:
            kwargs["method"] = draw(st.sampled_from(["lm", "hybr"]))
            kwargs["tol"] = draw(_POSITIVE_TOL)
        elif cls is pybamm.CasadiAlgebraicSolver:
            kwargs["tol"] = draw(_POSITIVE_TOL)
            kwargs["step_tol"] = draw(_POSITIVE_TOL)
        elif cls is pybamm.NonlinearSolver:
            kwargs["rtol"] = draw(_POSITIVE_TOL)
            kwargs["atol"] = draw(_POSITIVE_TOL)
        elif cls is pybamm.CompositeSolver:
            # Draw two simple solver classes (possibly the same) for sub_solvers.
            sub_cls_a, sub_cls_b = draw(
                st.lists(
                    st.sampled_from(_SIMPLE_SOLVER_CLASSES),
                    min_size=2,
                    max_size=2,
                )
            )
            kwargs["sub_solvers"] = [sub_cls_a(), sub_cls_b()]
        return kwargs

    return _strategy()


def step_kwargs() -> st.SearchStrategy[dict]:
    """Generate kwargs for the concrete experiment-step constructors.

    The step types are taken from the serialiser's own
    ``_experiment_step_factories`` map, so this strategy automatically covers
    every step type the serialiser supports (including ``c-rate``) and can
    never drift from it.  The ``_type`` key holds the serialised type string;
    the property test resolves it back to a factory via the same map.

    ``rest`` takes no ``value`` (and is reconstructed from a 0 A current); all
    other types take ``value`` as the first positional argument.
    """

    @st.composite
    def _strategy(draw):
        step_type = draw(st.sampled_from(sorted(_experiment_step_factories())))
        kwargs: dict = {"_type": step_type}
        if step_type == "rest":
            # rest() accepts duration but not value
            kwargs["duration"] = draw(st.floats(min_value=1.0, max_value=3600.0))
        else:
            if step_type == "c-rate":
                # C-rate 0 causes divide-by-zero in duration heuristic; keep |magnitude| ≥ 0.05
                magnitude = draw(
                    st.floats(
                        min_value=0.05,
                        max_value=10.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
                kwargs["value"] = magnitude * draw(st.sampled_from([1.0, -1.0]))
            else:
                kwargs["value"] = draw(
                    st.floats(
                        min_value=-10.0,
                        max_value=10.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            kwargs["duration"] = draw(
                st.one_of(st.none(), st.floats(min_value=1.0, max_value=3600.0))
            )
        kwargs["temperature"] = draw(
            st.one_of(st.none(), st.floats(min_value=250.0, max_value=320.0))
        )
        kwargs["period"] = draw(
            st.one_of(st.none(), st.floats(min_value=0.1, max_value=60.0))
        )
        kwargs["tags"] = draw(
            st.lists(st.text(min_size=1, max_size=8), min_size=0, max_size=3)
        )
        kwargs["description"] = draw(
            st.one_of(st.none(), st.text(min_size=0, max_size=20))
        )
        kwargs["skip_ok"] = draw(st.booleans())  # MUST be bool, not int (#5495)
        return kwargs

    return _strategy()


def parameter_values_overrides() -> st.SearchStrategy[dict]:
    """Generate ParameterValues override dicts.

    Keys come from a small whitelist of parameters that exist in the
    Chen2020 base parameter set. Values are bounded positive floats.
    """
    float_values = st.floats(
        min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
    )
    return st.dictionaries(
        keys=st.sampled_from(
            [
                "Nominal cell capacity [A.h]",
                "Initial temperature [K]",
                "Reference temperature [K]",
                "Lower voltage cut-off [V]",
                "Upper voltage cut-off [V]",
            ]
        ),
        values=float_values,
        min_size=1,
        max_size=4,
    )


# Option keys fuzzed by model_options_kwargs, chosen to be (near-)orthogonal across
# SPM/SPMe/DFN; keys not fuzzed need coordinated multi-option changes (MSMR, dimensionality, convection)
_FUZZED_OPTION_KEYS = (
    "calculate discharge energy",
    "lithium plating",
    "loss of active material",
    "particle",
    "particle mechanics",
    "SEI",
    "surface form",
    "thermal",
)

# Values requiring multi-option combinations (MSMR triple, x-lumped→pouch geometry)
_EXCLUDED_OPTION_VALUES: dict[str, frozenset[str]] = {
    "particle": frozenset({"MSMR"}),
    "thermal": frozenset({"x-lumped"}),
}

# Derived once from production data: {key: [accepted values]}. New option values
# added to ``possible_options`` are picked up automatically.
_OPTION_SPACE: dict[str, list] = {
    key: [
        value
        for value in pybamm.BatteryModelOptions({}).possible_options[key]
        if value not in _EXCLUDED_OPTION_VALUES.get(key, frozenset())
    ]
    for key in _FUZZED_OPTION_KEYS
}


def model_options_kwargs() -> st.SearchStrategy[dict]:
    """Generate kwargs for pybamm.lithium_ion.{SPM,SPMe,DFN}(options=...).

    Samples a subset of ``BatteryModelOptions`` keys (``_FUZZED_OPTION_KEYS``)
    with values taken straight from the canonical ``possible_options``
    (``_OPTION_SPACE``). Any illegal combination is rejected by the option
    validator at construction time; the test uses ``assume(False)`` to discard
    those.
    """

    @st.composite
    def _strategy(draw):
        cls = draw(
            st.sampled_from(
                [
                    pybamm.lithium_ion.SPM,
                    pybamm.lithium_ion.SPMe,
                    pybamm.lithium_ion.DFN,
                ]
            )
        )
        chosen_keys = draw(
            st.lists(
                st.sampled_from(_FUZZED_OPTION_KEYS),
                min_size=1,
                max_size=4,
                unique=True,
            )
        )
        kwargs: dict = {"_cls": cls}
        for key in chosen_keys:
            kwargs[key] = draw(st.sampled_from(_OPTION_SPACE[key]))
        return kwargs

    return _strategy()


def experiment_kwargs() -> st.SearchStrategy[dict]:
    """Generate kwargs for pybamm.Experiment plus a step list.

    The step list is generated as a list of short instruction strings,
    matching the human-readable form Experiment accepts (e.g.,
    ["Discharge at 1A for 1 hour", "Rest for 30 minutes"]).
    Strategy is intentionally simple — structural step attributes are
    covered by the per-step fuzzer (``step_kwargs``).
    """

    @st.composite
    def _strategy(draw):
        steps = draw(
            st.lists(
                st.sampled_from(
                    [
                        "Discharge at 1A for 1 hour",
                        "Charge at 0.5A until 4.2V",
                        "Hold at 4.2V for 30 minutes",
                        "Rest for 1 hour",
                    ]
                ),
                min_size=1,
                max_size=3,
            )
        )
        kwargs: dict = {"_steps": steps}
        kwargs["period"] = draw(
            st.one_of(st.none(), st.sampled_from(["1 second", "1 minute", "5 seconds"]))
        )
        kwargs["temperature"] = draw(
            st.one_of(st.none(), st.floats(min_value=250.0, max_value=320.0))
        )
        kwargs["termination"] = draw(
            st.one_of(
                st.none(),
                st.sampled_from(["80% capacity", "70% capacity", "3.0V"]),
            )
        )
        return kwargs

    return _strategy()
