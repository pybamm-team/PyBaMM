"""Every concrete serialisable subclass of every known base must be exercised by a
round-trip strategy or be explicitly exempt -- the Symbol contract, generalised.

A new BaseSolver/SubMesh subclass added to pybamm must register a strategy (or
fuzzed kwarg producer) for its base, or be added to that base's exemption set with
a one-line reason. Forgetting is a CI failure.
"""

from __future__ import annotations

import pybamm
from tests.strategies.serialise_values import _SOLVER_CLASSES
from tests.unit.test_serialisation.test_symbol_strategy_coverage import (
    _all_concrete_subclasses,
)


def _covered_solver_classes() -> set[type]:
    return set(_SOLVER_CLASSES)


# Bases reconstructed but not independently fuzzed (carried inside a model/mesh
# round-trip) are exempted with a reason rather than given a standalone strategy.
_SOLVER_EXEMPT: set[type] = {
    pybamm.DummySolver,  # no numerical config to round-trip
    pybamm.JaxSolver,  # requires optional jax/jaxlib dep; not installed in CI
}

# A submesh is round-trippable only if it (or an ancestor below the bare SubMesh
# base) defines its OWN _from_json. Most SubMesh subclasses inherit to_json but
# define no _from_json -- they encode but cannot decode, a pre-existing limitation
# PR2 does not expand. They are NOT silently "covered"; each must be listed here
# with a reason. POPULATE from the Step-2 failure output, one justified line each.
_SUBMESH_EXEMPT: set[type] = {
    pybamm.Chebyshev1DSubMesh,  # encode-only: inherits to_json, no _from_json
    pybamm.Exponential1DSubMesh,  # encode-only: inherits to_json, no _from_json
    pybamm.ScikitChebyshev2DSubMesh,  # encode-only: inherits to_json, no _from_json
    pybamm.ScikitExponential2DSubMesh,  # encode-only: inherits to_json, no _from_json
    pybamm.ScikitSubMesh2D,  # encode-only: inherits to_json, no _from_json
    pybamm.SpectralVolume1DSubMesh,  # encode-only: inherits to_json, no _from_json
    pybamm.SubMesh1D,  # encode-only: inherits to_json, no _from_json
    pybamm.SubMesh2D,  # encode-only: inherits to_json, no _from_json
    pybamm.SymbolicUniform1DSubMesh,  # encode-only: inherits to_json, no _from_json
    pybamm.Uniform2DSubMesh,  # encode-only: inherits to_json, no _from_json
    pybamm.UserSupplied1DSubMesh,  # encode-only: inherits to_json, no _from_json
    pybamm.UserSupplied2DSubMesh,  # encode-only: inherits to_json, no _from_json
}


def _defines_own_from_json(cls: type) -> bool:
    # An OVERRIDING _from_json, not merely inherited: walk the MRO for _from_json in
    # a class __dict__.
    return any("_from_json" in k.__dict__ for k in cls.__mro__)


def test_every_concrete_solver_subclass_is_covered_or_exempt():
    concrete = _all_concrete_subclasses(pybamm.BaseSolver)
    missing = concrete - _covered_solver_classes() - _SOLVER_EXEMPT
    assert not missing, (
        "BaseSolver subclasses with no round-trip coverage and no exemption: "
        + ", ".join(sorted(c.__name__ for c in missing))
    )


def test_every_concrete_submesh_subclass_round_trips_or_exempt():
    concrete = _all_concrete_subclasses(pybamm.SubMesh)
    uncovered = {
        c
        for c in concrete - _SUBMESH_EXEMPT
        if not (_defines_own_from_json(c) and "to_json" in dir(c))
    }
    assert not uncovered, (
        "SubMesh subclasses that are encode-only (no own _from_json) and not in "
        "_SUBMESH_EXEMPT: " + ", ".join(sorted(c.__name__ for c in uncovered))
    )
