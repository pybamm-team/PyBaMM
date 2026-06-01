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
    """Pick a concrete solver class and generate valid kwargs for it."""

    @st.composite
    def _strategy(draw):
        cls = draw(
            st.sampled_from(
                [
                    pybamm.IDAKLUSolver,
                    pybamm.CasadiSolver,
                    pybamm.ScipySolver,
                ]
            )
        )
        kwargs: dict = {
            "_cls": cls,
            "rtol": draw(_POSITIVE_TOL),
            "atol": draw(_POSITIVE_TOL),
        }
        if cls is pybamm.IDAKLUSolver:
            kwargs["root_method"] = draw(_root_method())
            kwargs["options"] = draw(
                st.fixed_dictionaries(
                    {
                        "print_stats": st.booleans(),
                        "compile": st.booleans(),
                    }
                )
            )
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
                # A C-rate of 0 is meaningless (0 current is a rest step) and
                # makes CRate's default-duration heuristic (1 / |C-rate|) divide
                # by zero, so keep the magnitude bounded away from zero. Sign is
                # free: positive discharges, negative charges.
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


# Option keys fuzzed by ``model_options_kwargs``. Chosen to be (near-)orthogonal
# across SPM/SPMe/DFN so a random draw is almost always a valid model. Their
# *values* are derived from the canonical ``possible_options`` (see
# ``_OPTION_SPACE``) so they can never drift; we only sample this fixed key set.
# Keys deliberately not fuzzed need a coordinated multi-option change:
#   - "open-circuit potential"/"particle"/"intercalation kinetics": MSMR must be
#     set on all three together (so "particle" is fuzzed but with "MSMR" pruned).
#   - "dimensionality": non-zero values require "current collector" changes.
#   - "convection": must be "none" for lithium-ion models.
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

# Values that only construct as part of a multi-option combination, verified by
# building SPM/SPMe/DFN with each value in isolation:
#   - particle "MSMR" needs the MSMR triple (see above).
#   - thermal "x-lumped" needs "cell geometry"="pouch" ("x-full" sets that
#     default itself, "x-lumped" does not).
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
