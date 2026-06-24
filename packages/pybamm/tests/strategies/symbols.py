"""Hypothesis strategies for pybamm.Symbol subclasses.

The strategies are hand-curated. A meta-test
(``tests/unit/test_serialisation/test_symbol_strategy_coverage.py``)
walks ``Symbol.__subclasses__()`` and fails if a concrete subclass has no
strategy registered here. New ``Symbol`` subclasses must either be added
to ``_STRATEGIES`` or to one of the two exemption sets (with justification):
``_NOT_ROUND_TRIPPABLE`` for permanent exemptions, or ``_KNOWN_FAILING`` for
serialiser bugs awaiting a fix.
"""

from __future__ import annotations

from collections.abc import Callable

import hypothesis.strategies as st
import numpy as np

import pybamm
from pybamm.expression_tree.averages import _BaseAverage as _BaseAverageClass
from pybamm.expression_tree.binary_operators import _Heaviside as _HeavisideClass
from pybamm.expression_tree.operations.serialise import ExpressionFunctionParameter

# Small fixed enumeration of domains; not exhaustive.
_DOMAINS = st.sampled_from(
    [
        [],
        ["negative electrode"],
        ["positive electrode"],
        ["separator"],
    ]
)

_FINITE_FLOATS = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

# Small positive floats for exponents, scales, etc.
_POSITIVE_FLOATS = st.floats(
    min_value=1e-6,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)

# Short identifier strings
_NAMES = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8)

# Constructive sub-strategies used in branch factories instead of .filter.


def _domain_free_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Leaves that always have an empty domain: Scalar and Time."""
    return st.one_of(
        st.builds(pybamm.Scalar, _FINITE_FLOATS),
        st.builds(pybamm.Time),
    )


def _neg_electrode_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Variable with domain=['negative electrode']."""
    return st.builds(
        pybamm.Variable,
        _NAMES,
        domains=st.just({"primary": ["negative electrode"]}),
    )


def _sep_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Variable with domain=['separator']."""
    return st.builds(
        pybamm.Variable,
        _NAMES,
        domains=st.just({"primary": ["separator"]}),
    )


def _electrode_domain_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Variable leaves with one of the three electrode domains."""
    return st.one_of(
        _neg_electrode_leaves(),
        st.builds(
            pybamm.Variable,
            _NAMES,
            domains=st.just({"primary": ["positive electrode"]}),
        ),
        _sep_leaves(),
    )


def _particle_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Variable leaves with a particle domain (negative or positive particle)."""
    return st.builds(
        pybamm.Variable,
        _NAMES,
        domains=st.sampled_from(
            [
                {"primary": ["negative particle"]},
                {"primary": ["positive particle"]},
            ]
        ),
    )


def _neg_particle_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Variable leaves with domain=['negative particle'] (for SecondaryBroadcast)."""
    return st.builds(
        pybamm.Variable,
        _NAMES,
        domains=st.just({"primary": ["negative particle"]}),
    )


def _cc_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Variable leaves with domain=['current collector']."""
    return st.builds(
        pybamm.Variable,
        _NAMES,
        domains=st.just({"primary": ["current collector"]}),
    )


def _any_domain_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Variable leaves with any non-empty domain (electrode, particle, or cc)."""
    return st.one_of(
        _electrode_domain_leaves(),
        _particle_leaves(),
        _cc_leaves(),
    )


def scalar_strategy() -> st.SearchStrategy[pybamm.Scalar]:
    return st.builds(pybamm.Scalar, _FINITE_FLOATS)


def variable_strategy() -> st.SearchStrategy[pybamm.Variable]:
    return st.builds(
        pybamm.Variable,
        _NAMES,
        domains=_DOMAINS.map(lambda d: {"primary": d} if d else {}),
    )


# Branch lookup table, populated after the branch factories are defined below.
_STRATEGIES: dict[
    type[pybamm.Symbol], Callable[[st.SearchStrategy], st.SearchStrategy]
] = {}


def parameter_strategy() -> st.SearchStrategy[pybamm.Parameter]:
    return st.builds(
        pybamm.Parameter,
        _NAMES,
    )


def time_strategy() -> st.SearchStrategy[pybamm.Time]:
    return st.builds(pybamm.Time)


def input_parameter_strategy() -> st.SearchStrategy[pybamm.InputParameter]:
    return st.builds(
        pybamm.InputParameter,
        _NAMES,
    )


def _leaves() -> st.SearchStrategy[pybamm.Symbol]:
    return st.one_of(
        scalar_strategy(),
        variable_strategy(),
        parameter_strategy(),
        time_strategy(),
        input_parameter_strategy(),
    )


def _unary_branch(
    child_strategy: st.SearchStrategy[pybamm.Symbol],
    cls: type[pybamm.UnaryOperator],
) -> st.SearchStrategy[pybamm.UnaryOperator]:
    return child_strategy.map(cls)


def _binary_branch(
    child_strategy: st.SearchStrategy[pybamm.Symbol],
    cls: type[pybamm.BinaryOperator],
) -> st.SearchStrategy[pybamm.BinaryOperator]:
    # Domain-free right leaf is compatible with any left child (Symbol.get_children_domains).
    return st.builds(cls, child_strategy, _domain_free_leaves())


def _boundary_value_branch(
    child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.BoundaryValue]:
    return st.builds(
        pybamm.BoundaryValue,
        child_strategy,
        st.sampled_from(["left", "right"]),
    )


# Non-empty domains only — FullBroadcast and PrimaryBroadcast both reject [].
_NONEMPTY_DOMAINS = st.sampled_from(
    [
        ["negative electrode"],
        ["positive electrode"],
        ["separator"],
    ]
)


def _full_broadcast_branch(
    child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.FullBroadcast]:
    # Filter out 'current collector' domain (FullBroadcast rejects it).
    return st.builds(
        pybamm.FullBroadcast,
        child_strategy.filter(lambda c: c.domain != ["current collector"]),
        _NONEMPTY_DOMAINS,
    )


def _primary_broadcast_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.PrimaryBroadcast]:
    # PrimaryBroadcast has strict domain-compatibility rules; domain-free
    # children sidestep them without restricting the broadcast targets.
    return st.builds(
        pybamm.PrimaryBroadcast,
        _domain_free_leaves(),
        _NONEMPTY_DOMAINS,
    )


def _constant_strategy() -> st.SearchStrategy[pybamm.Constant]:
    """Named scalar constant (e.g. Faraday constant)."""
    return st.builds(pybamm.Constant, _FINITE_FLOATS, _NAMES)


def _coupled_variable_strategy() -> st.SearchStrategy[pybamm.CoupledVariable]:
    """A variable whose equation is set by another model/submodel."""
    return st.builds(
        pybamm.CoupledVariable,
        _NAMES,
        domain=st.sampled_from([[], ["negative electrode"], ["separator"]]),
    )


def _spatial_variable_strategy() -> st.SearchStrategy[pybamm.SpatialVariable]:
    """Spatial variable — requires a non-empty domain and a compatible name.

    We use only generic names ('x', 'y', 'z') that don't trigger the
    name-vs-domain consistency checks inside SpatialVariable.__init__.
    """
    safe_names = st.sampled_from(["x", "y", "z"])
    safe_domains = st.sampled_from(
        [
            ["negative electrode"],
            ["positive electrode"],
            ["separator"],
            ["current collector"],
        ]
    )
    return st.builds(pybamm.SpatialVariable, safe_names, domain=safe_domains)


def _function_parameter_strategy() -> st.SearchStrategy[pybamm.FunctionParameter]:
    """FunctionParameter — a named callable placeholder."""
    return st.builds(
        pybamm.FunctionParameter,
        _NAMES,
        st.fixed_dictionaries({"x": scalar_strategy()}),
    )


def _simple_unary(cls: type) -> Callable:
    """Return a branch-factory for a unary operator that takes only a child."""
    return lambda children: _unary_branch(children, cls)


def _secondary_broadcast_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.SecondaryBroadcast]:
    """SecondaryBroadcast: child in 'negative particle' → broadcasts to electrode."""
    # Simplest safe case: particle → electrode with non-empty child domain.
    return st.builds(
        pybamm.SecondaryBroadcast,
        _neg_particle_leaves(),
        st.just(["negative electrode"]),
    )


def _conditional_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.Conditional]:
    """Conditional(selector, branch1, branch2) — selector must be 0-D scalar."""
    # Plain Scalar for selector avoids Variable domain shape issues.
    return st.builds(
        pybamm.Conditional,
        scalar_strategy(),
        scalar_strategy(),
        scalar_strategy(),
    )


def _neg_electrode_nonvariable_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Non-Variable domain-bearing leaves for 'negative electrode'.

    pybamm.Concatenation raises TypeError when ALL children are Variable instances
    (use ConcatenationVariable for that).  FullBroadcast of a Scalar produces a
    domain-bearing Symbol that is not a Variable, so it passes the constructor guard.
    """
    return st.builds(
        pybamm.FullBroadcast,
        scalar_strategy(),
        st.just(["negative electrode"]),
    )


def _sep_nonvariable_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Non-Variable domain-bearing leaves for 'separator'."""
    return st.builds(
        pybamm.FullBroadcast,
        scalar_strategy(),
        st.just(["separator"]),
    )


def _concatenation_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.Concatenation]:
    """Concatenation of two non-Variable domain-bearing children with disjoint domains.

    pybamm.Concatenation rejects children that are all Variable instances; those
    cases are handled by the ConcatenationVariable strategy.  We use FullBroadcast
    leaves here to satisfy the domain-bearing requirement without triggering the
    Variable-only guard.
    """
    return st.builds(
        pybamm.Concatenation,
        _neg_electrode_nonvariable_leaves(),
        _sep_nonvariable_leaves(),
    )


def _concatenation_variable_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.ConcatenationVariable]:
    """ConcatenationVariable — concatenation of two Variable nodes."""
    neg = st.builds(
        pybamm.Variable, _NAMES, domains=st.just({"primary": ["negative electrode"]})
    )
    sep = st.builds(
        pybamm.Variable, _NAMES, domains=st.just({"primary": ["separator"]})
    )
    return st.builds(pybamm.concatenation, neg, sep)


def _sparse_stack_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.SparseStack]:
    """SparseStack of two domain-bearing children with disjoint domains."""
    return st.builds(pybamm.SparseStack, _neg_electrode_leaves(), _sep_leaves())


def _numpy_concatenation_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.NumpyConcatenation]:
    """NumpyConcatenation with no children (pure concatenation marker)."""
    return st.just(pybamm.NumpyConcatenation())


def _indefinite_integral_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.IndefiniteIntegral]:
    """IndefiniteIntegral: integrates a domain-bearing child w.r.t. its spatial var."""

    def make_indef(child: pybamm.Symbol) -> pybamm.IndefiniteIntegral:
        x = pybamm.SpatialVariable("x", domain=child.domain)
        return pybamm.IndefiniteIntegral(child, x)

    return _electrode_domain_leaves().map(make_indef)


def _x_average_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.XAverage]:
    """XAverage: child must have an electrode primary domain."""
    return _electrode_domain_leaves().map(pybamm.XAverage)


def _z_average_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.ZAverage]:
    """ZAverage: child must have 'current collector' domain."""
    return _cc_leaves().map(pybamm.ZAverage)


def _yz_average_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.YZAverage]:
    """YZAverage: child must have 'current collector' domain."""
    return _cc_leaves().map(pybamm.YZAverage)


def _r_average_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.RAverage]:
    """RAverage: child must have a particle domain."""
    return _particle_leaves().map(pybamm.RAverage)


def _arcsinh2_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy:
    """Arcsinh2(a, b) — two domain-free children."""
    from pybamm.expression_tree.functions import Arcsinh2

    return st.builds(Arcsinh2, _domain_free_leaves(), _domain_free_leaves())


def _reg_power_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy:
    """RegPower(base, exponent, scale=1) — domain-free children."""
    return st.builds(
        pybamm.RegPower,
        _domain_free_leaves(),
        _domain_free_leaves(),
        st.just(pybamm.Scalar(1.0)),  # scale=1, no extra variability needed
    )


def _interpolant_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.Interpolant]:
    """Interpolant leaf (data-bearing, no symbol children needed for strategy)."""
    x_data = st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=8,
        unique=True,
    ).map(lambda pts: np.array(sorted(pts)))
    y_data = st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=8,
    ).map(np.array)

    def make_interp(xy):
        x_arr, y_arr = xy
        # Ensure x and y have the same length
        length = min(len(x_arr), len(y_arr))
        x_trimmed = x_arr[:length]
        y_trimmed = y_arr[:length]
        child = pybamm.Variable("a")
        return pybamm.Interpolant(x_trimmed, y_trimmed, child)

    return st.tuples(x_data, y_data).map(make_interp)


def _discrete_time_data_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.DiscreteTimeData]:
    """DiscreteTimeData leaf with numpy time/data arrays."""
    time_data = st.lists(
        st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        min_size=2,
        max_size=6,
        unique=True,
    ).map(lambda pts: np.array(sorted(pts)))
    value_data = st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=6,
    ).map(np.array)

    def make_dtd(tv):
        t_arr, v_arr = tv
        length = min(len(t_arr), len(v_arr))
        return pybamm.DiscreteTimeData(t_arr[:length], v_arr[:length], "dtd")

    return st.tuples(time_data, value_data).map(make_dtd)


def _expression_function_parameter_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[ExpressionFunctionParameter]:
    """ExpressionFunctionParameter — wraps a child with a function name."""
    return st.builds(
        ExpressionFunctionParameter,
        _NAMES,
        _domain_free_leaves(),
        st.just("np.exp"),
        st.just([]),
    )


# Domains a VectorField component may carry — the non-empty domains covered by
# ``_any_domain_leaves`` (electrode, particle, current collector).
_FIELD_DOMAINS = st.sampled_from(
    [
        ["negative electrode"],
        ["positive electrode"],
        ["separator"],
        ["negative particle"],
        ["positive particle"],
        ["current collector"],
    ]
)


def _vector_field_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.VectorField]:
    """VectorField(lr, tb) — two children with the same non-empty domain.

    VectorField requires both lr_field and tb_field to share the same domain.
    Rather than draw two leaves and filter for a matching domain, we draw a
    single domain first and build both components with it, so every pair is
    valid by construction.
    """

    def make_vf(domain: list[str]) -> st.SearchStrategy[pybamm.VectorField]:
        leaf = st.builds(pybamm.Variable, _NAMES, domains=st.just({"primary": domain}))
        return st.builds(pybamm.VectorField, leaf, leaf)

    return _FIELD_DOMAINS.flatmap(make_vf)


# Sides shared by boundary operators.
_SIDES = st.sampled_from(["left", "right"])


def _backward_indefinite_integral_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.BackwardIndefiniteIntegral]:
    """BackwardIndefiniteIntegral: integrates a domain-bearing child w.r.t. its spatial var."""

    def make_backward(child: pybamm.Symbol) -> pybamm.BackwardIndefiniteIntegral:
        x = pybamm.SpatialVariable("x", domain=child.domain)
        return pybamm.BackwardIndefiniteIntegral(child, x)

    return _electrode_domain_leaves().map(make_backward)


def _boundary_gradient_branch(
    child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.BoundaryGradient]:
    """BoundaryGradient(child, side) — child must have a non-empty domain."""
    return st.builds(pybamm.BoundaryGradient, _any_domain_leaves(), _SIDES)


def _boundary_mesh_size_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.BoundaryMeshSize]:
    """BoundaryMeshSize(child, side) — child must have a non-empty domain."""
    return st.builds(pybamm.BoundaryMeshSize, _any_domain_leaves(), _SIDES)


def _boundary_integral_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.BoundaryIntegral]:
    """BoundaryIntegral(child, region) — exercises both default and non-default regions."""
    return st.builds(
        pybamm.BoundaryIntegral,
        _any_domain_leaves(),
        st.sampled_from(["entire", "negative tab", "positive tab"]),
    )


def _explicit_time_integral_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.ExplicitTimeIntegral]:
    """ExplicitTimeIntegral(child, initial_condition) — initial_condition a Scalar."""
    return st.builds(
        pybamm.ExplicitTimeIntegral,
        _any_domain_leaves(),
        scalar_strategy(),
    )


def _index_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.Index]:
    """Index(StateVector, index) — check_size disabled so any index builds cleanly."""
    return st.builds(
        pybamm.Index,
        st.just(pybamm.StateVector(slice(0, 1))),
        st.just(0),
        check_size=st.just(False),
    )


def _delta_function_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.DeltaFunction]:
    """DeltaFunction(child, side, domain) — child domain must differ from the target domain."""
    return st.builds(
        pybamm.DeltaFunction,
        _neg_electrode_leaves(),
        _SIDES,
        st.just("separator"),
    )


def _evaluate_at_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.EvaluateAt]:
    """EvaluateAt(child, position) — numeric position on a domain-bearing child."""
    return st.builds(pybamm.EvaluateAt, _any_domain_leaves(), _FINITE_FLOATS)


def _one_dimensional_integral_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.OneDimensionalIntegral]:
    """OneDimensionalIntegral(child, integration_domain, direction)."""

    def make_odi(child: pybamm.Symbol) -> pybamm.OneDimensionalIntegral:
        return pybamm.OneDimensionalIntegral(child, child.domain, "x")

    return _electrode_domain_leaves().map(make_odi)


def _upwind_downwind_2d_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.UpwindDownwind2D]:
    """UpwindDownwind2D(child, lr_direction, tb_direction) — domain-bearing child."""
    return st.builds(
        pybamm.UpwindDownwind2D,
        _any_domain_leaves(),
        st.sampled_from(["upwind", "downwind"]),
        st.sampled_from(["upwind", "downwind"]),
    )


def _node_to_edge_2d_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.NodeToEdge2D]:
    """NodeToEdge2D(child, direction) — domain-bearing node child (not on edges)."""
    return st.builds(
        pybamm.NodeToEdge2D,
        _any_domain_leaves(),
        st.sampled_from(["lr", "tb"]),
    )


def _magnitude_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.Magnitude]:
    """Magnitude(child, direction) — domain-bearing child, directional component."""
    return st.builds(
        pybamm.Magnitude,
        _any_domain_leaves(),
        st.sampled_from(["x", "y", "z"]),
    )


def _discrete_time_sum_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.DiscreteTimeSum]:
    """DiscreteTimeSum: built around a DiscreteTimeData child."""
    return st.builds(pybamm.DiscreteTimeSum, _discrete_time_data_branch(None))


def _size_average_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.SizeAverage]:
    """SizeAverage(child, f_a_dist) — child on a particle-size domain."""
    child = st.builds(
        pybamm.Variable,
        _NAMES,
        domains=st.sampled_from(
            [
                {"primary": ["negative particle size"]},
                {"primary": ["positive particle size"]},
            ]
        ),
    )
    return st.builds(pybamm.SizeAverage, child, scalar_strategy())


def _primary_broadcast_to_edges_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.PrimaryBroadcastToEdges]:
    """PrimaryBroadcastToEdges — mirror PrimaryBroadcast with a domain-free child."""
    return st.builds(
        pybamm.PrimaryBroadcastToEdges,
        _domain_free_leaves(),
        _NONEMPTY_DOMAINS,
    )


def _secondary_broadcast_to_edges_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.SecondaryBroadcastToEdges]:
    """SecondaryBroadcastToEdges — particle child broadcast to an electrode."""
    return st.builds(
        pybamm.SecondaryBroadcastToEdges,
        _neg_particle_leaves(),
        st.just(["negative electrode"]),
    )


def _tertiary_secondary_leaves() -> st.SearchStrategy[pybamm.Symbol]:
    """Variable leaves with primary + secondary domains (valid TertiaryBroadcast child)."""
    return st.builds(
        pybamm.Variable,
        _NAMES,
        domains=st.just(
            {"primary": ["negative particle"], "secondary": ["negative electrode"]}
        ),
    )


def _tertiary_broadcast_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.TertiaryBroadcast]:
    """TertiaryBroadcast — child with primary+secondary domains broadcast to a tertiary."""
    return st.builds(
        pybamm.TertiaryBroadcast,
        _tertiary_secondary_leaves(),
        st.just(["current collector"]),
    )


def _tertiary_broadcast_to_edges_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.TertiaryBroadcastToEdges]:
    """TertiaryBroadcastToEdges — as TertiaryBroadcast but to edges."""
    return st.builds(
        pybamm.TertiaryBroadcastToEdges,
        _tertiary_secondary_leaves(),
        st.just(["current collector"]),
    )


def _full_broadcast_to_edges_branch(
    child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.FullBroadcastToEdges]:
    """FullBroadcastToEdges — mirror FullBroadcast with a non-empty target domain."""
    return st.builds(
        pybamm.FullBroadcastToEdges,
        child_strategy.filter(lambda c: c.domain != ["current collector"]),
        _NONEMPTY_DOMAINS,
    )


def _heaviside_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[_HeavisideClass]:
    """_Heaviside(name, left, right) — semi-private base from two domain-free leaves."""
    return st.builds(
        _HeavisideClass,
        _NAMES,
        _domain_free_leaves(),
        _domain_free_leaves(),
    )


def _tensor_field_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.TensorField]:
    """TensorField of rank 1 (flat list) or rank 2 (list of equal-length rows)."""
    rank_1 = st.lists(_domain_free_leaves(), min_size=1, max_size=3)
    rank_2 = st.integers(min_value=1, max_value=3).flatmap(
        lambda n_cols: st.lists(
            st.lists(_domain_free_leaves(), min_size=n_cols, max_size=n_cols),
            min_size=1,
            max_size=3,
        )
    )
    return st.builds(pybamm.TensorField, st.one_of(rank_1, rank_2))


def _state_vector_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.StateVector]:
    """StateVector with a sampled y-slice."""
    return st.builds(pybamm.StateVector, st.sampled_from([slice(0, 1), slice(0, 2)]))


def _state_vector_dot_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.StateVectorDot]:
    """StateVectorDot with a sampled y-slice."""
    return st.builds(pybamm.StateVectorDot, st.sampled_from([slice(0, 1), slice(0, 2)]))


def _variable_dot_strategy() -> st.SearchStrategy[pybamm.VariableDot]:
    """VariableDot — like Variable (a named, optionally domain-bearing leaf)."""
    return st.builds(
        pybamm.VariableDot,
        _NAMES,
        domain=_DOMAINS,
    )


def _spatial_variable_edge_strategy() -> st.SearchStrategy[pybamm.SpatialVariableEdge]:
    """SpatialVariableEdge — like SpatialVariable but the Edge subclass."""
    safe_names = st.sampled_from(["x", "y", "z"])
    safe_domains = st.sampled_from(
        [
            ["negative electrode"],
            ["positive electrode"],
            ["separator"],
            ["current collector"],
        ]
    )
    return st.builds(pybamm.SpatialVariableEdge, safe_names, domain=safe_domains)


# Small fixed 2x2 / 2x1 numpy arrays for Array/Matrix/Vector leaves.
_SMALL_MATRIX = st.just(np.array([[1.0, 2.0], [3.0, 4.0]]))
_SMALL_VECTOR = st.just(np.array([[1.0], [2.0]]))


def _array_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.Array]:
    """Array leaf from a fixed small numpy array."""
    return st.builds(pybamm.Array, _SMALL_MATRIX)


def _matrix_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.Matrix]:
    """Matrix leaf from a fixed small numpy array."""
    return st.builds(pybamm.Matrix, _SMALL_MATRIX)


def _vector_branch(
    _child_strategy: st.SearchStrategy[pybamm.Symbol],
) -> st.SearchStrategy[pybamm.Vector]:
    """Vector leaf from a fixed small column numpy array."""
    return st.builds(pybamm.Vector, _SMALL_VECTOR)


# Register branches in the lookup table. Each value is a callable that
# takes the child strategy and returns a strategy for that node.
_STRATEGIES.update(
    {
        pybamm.Scalar: lambda _children: scalar_strategy(),
        pybamm.Variable: lambda _children: variable_strategy(),
        pybamm.Parameter: lambda _children: parameter_strategy(),
        pybamm.Time: lambda _children: time_strategy(),
        pybamm.InputParameter: lambda _children: input_parameter_strategy(),
        pybamm.Negate: lambda children: _unary_branch(children, pybamm.Negate),
        pybamm.AbsoluteValue: lambda children: _unary_branch(
            children, pybamm.AbsoluteValue
        ),
        pybamm.Addition: lambda children: _binary_branch(children, pybamm.Addition),
        pybamm.Subtraction: lambda children: _binary_branch(
            children, pybamm.Subtraction
        ),
        pybamm.Multiplication: lambda children: _binary_branch(
            children, pybamm.Multiplication
        ),
        pybamm.Division: lambda children: _binary_branch(children, pybamm.Division),
        pybamm.Power: lambda children: _binary_branch(children, pybamm.Power),
        pybamm.BoundaryValue: _boundary_value_branch,
        pybamm.FullBroadcast: _full_broadcast_branch,
        pybamm.PrimaryBroadcast: _primary_broadcast_branch,
        # additional leaf strategies
        pybamm.Constant: lambda _children: _constant_strategy(),
        pybamm.CoupledVariable: lambda _children: _coupled_variable_strategy(),
        pybamm.SpatialVariable: lambda _children: _spatial_variable_strategy(),
        pybamm.FunctionParameter: lambda _children: _function_parameter_strategy(),
        # simple unary operators
        pybamm.Transpose: _simple_unary(pybamm.Transpose),
        pybamm.Sign: _simple_unary(pybamm.Sign),
        pybamm.Floor: _simple_unary(pybamm.Floor),
        pybamm.Ceiling: _simple_unary(pybamm.Ceiling),
        pybamm.NotConstant: _simple_unary(pybamm.NotConstant),
        # binary operators
        pybamm.MatrixMultiplication: lambda children: _binary_branch(
            children, pybamm.MatrixMultiplication
        ),
        pybamm.KroneckerProduct: lambda children: _binary_branch(
            children, pybamm.KroneckerProduct
        ),
        pybamm.TensorProduct: lambda children: _binary_branch(
            children, pybamm.TensorProduct
        ),
        pybamm.Inner: lambda children: _binary_branch(children, pybamm.Inner),
        pybamm.Equality: lambda children: _binary_branch(children, pybamm.Equality),
        pybamm.EqualHeaviside: lambda children: _binary_branch(
            children, pybamm.EqualHeaviside
        ),
        pybamm.NotEqualHeaviside: lambda children: _binary_branch(
            children, pybamm.NotEqualHeaviside
        ),
        pybamm.Modulo: lambda children: _binary_branch(children, pybamm.Modulo),
        pybamm.Minimum: lambda children: _binary_branch(children, pybamm.Minimum),
        pybamm.Maximum: lambda children: _binary_branch(children, pybamm.Maximum),
        pybamm.Hypot: lambda children: _binary_branch(children, pybamm.Hypot),
        # SpecificFunction subclasses (all unary)
        pybamm.Arcsinh: _simple_unary(pybamm.Arcsinh),
        pybamm.Arctan: _simple_unary(pybamm.Arctan),
        pybamm.Cos: _simple_unary(pybamm.Cos),
        pybamm.Cosh: _simple_unary(pybamm.Cosh),
        pybamm.Erf: _simple_unary(pybamm.Erf),
        pybamm.Exp: _simple_unary(pybamm.Exp),
        pybamm.Log: _simple_unary(pybamm.Log),
        pybamm.Max: _simple_unary(pybamm.Max),
        pybamm.Min: _simple_unary(pybamm.Min),
        pybamm.Sin: _simple_unary(pybamm.Sin),
        pybamm.Sinh: _simple_unary(pybamm.Sinh),
        pybamm.Sqrt: _simple_unary(pybamm.Sqrt),
        pybamm.Tanh: _simple_unary(pybamm.Tanh),
        # Function subclasses with special constructors
        pybamm.Arcsinh2: _arcsinh2_branch,
        pybamm.RegPower: _reg_power_branch,
        # data-bearing leaves
        pybamm.Interpolant: _interpolant_branch,
        ExpressionFunctionParameter: _expression_function_parameter_branch,
        # n-ary / complex branch strategies
        pybamm.Conditional: _conditional_branch,
        pybamm.Concatenation: _concatenation_branch,
        pybamm.ConcatenationVariable: _concatenation_variable_branch,
        pybamm.SparseStack: _sparse_stack_branch,
        pybamm.NumpyConcatenation: _numpy_concatenation_branch,
        pybamm.SecondaryBroadcast: _secondary_broadcast_branch,
        # integral / average strategies
        pybamm.IndefiniteIntegral: _indefinite_integral_branch,
        pybamm.XAverage: _x_average_branch,
        pybamm.ZAverage: _z_average_branch,
        pybamm.YZAverage: _yz_average_branch,
        pybamm.RAverage: _r_average_branch,
        # vector/tensor fields
        pybamm.VectorField: _vector_field_branch,
        # spatial unary operators: child-only, round-trip via the generic spatial
        # hook; all require a non-empty-domain child (DomainError otherwise)
        pybamm.Gradient: lambda _children: _any_domain_leaves().map(pybamm.Gradient),
        pybamm.Laplacian: lambda _children: _any_domain_leaves().map(pybamm.Laplacian),
        pybamm.GradientSquared: lambda _children: _any_domain_leaves().map(
            pybamm.GradientSquared
        ),
        pybamm.Mass: lambda _children: _any_domain_leaves().map(pybamm.Mass),
        pybamm.BoundaryMass: lambda _children: _any_domain_leaves().map(
            pybamm.BoundaryMass
        ),
        pybamm.DefiniteIntegralVector: lambda _children: st.builds(
            pybamm.DefiniteIntegralVector,
            _any_domain_leaves(),
            vector_type=st.sampled_from(["row", "column"]),
        ),
        # Upwind / Downwind: (self, child) only — round-trip via the generic spatial hook.
        # Both require non-empty-domain children.
        pybamm.Upwind: lambda _children: _any_domain_leaves().map(pybamm.Upwind),
        pybamm.Downwind: lambda _children: _any_domain_leaves().map(pybamm.Downwind),
        # Divergence needs an edge-evaluating child, so wrap a domain-bearing
        # leaf in a Gradient. Round-trips via the generic spatial hook.
        pybamm.Divergence: lambda _children: _any_domain_leaves().map(
            lambda leaf: pybamm.Divergence(pybamm.Gradient(leaf))
        ),
        # classes with non-children constructor args; serialise losslessly (#5548)
        pybamm.BackwardIndefiniteIntegral: _backward_indefinite_integral_branch,
        pybamm.BoundaryGradient: _boundary_gradient_branch,
        pybamm.BoundaryMeshSize: _boundary_mesh_size_branch,
        pybamm.BoundaryIntegral: _boundary_integral_branch,
        pybamm.OneDimensionalIntegral: _one_dimensional_integral_branch,
        pybamm.ExplicitTimeIntegral: _explicit_time_integral_branch,
        pybamm.Index: _index_branch,
        pybamm.DeltaFunction: _delta_function_branch,
        pybamm.EvaluateAt: _evaluate_at_branch,
        pybamm.UpwindDownwind2D: _upwind_downwind_2d_branch,
        pybamm.NodeToEdge2D: _node_to_edge_2d_branch,
        pybamm.Magnitude: _magnitude_branch,
        pybamm.DiscreteTimeData: _discrete_time_data_branch,
        pybamm.DiscreteTimeSum: _discrete_time_sum_branch,
        pybamm.SizeAverage: _size_average_branch,
        pybamm.PrimaryBroadcastToEdges: _primary_broadcast_to_edges_branch,
        pybamm.SecondaryBroadcastToEdges: _secondary_broadcast_to_edges_branch,
        pybamm.TertiaryBroadcast: _tertiary_broadcast_branch,
        pybamm.TertiaryBroadcastToEdges: _tertiary_broadcast_to_edges_branch,
        pybamm.FullBroadcastToEdges: _full_broadcast_to_edges_branch,
        _HeavisideClass: _heaviside_branch,
        pybamm.TensorField: _tensor_field_branch,
        pybamm.StateVector: _state_vector_branch,
        pybamm.StateVectorDot: _state_vector_dot_branch,
        pybamm.VariableDot: lambda _children: _variable_dot_strategy(),
        pybamm.SpatialVariableEdge: lambda _children: _spatial_variable_edge_strategy(),
        pybamm.Array: _array_branch,
        pybamm.Matrix: _matrix_branch,
        pybamm.Vector: _vector_branch,
    }
)


# Concrete Symbol subclasses excluded from round-trip tests, split by exclusion reason.

# Permanently not round-trippable: abstract bases, functions that raise by design, and mesh-required classes.
_NOT_ROUND_TRIPPABLE: frozenset[type[pybamm.Symbol]] = frozenset(
    {
        pybamm.UnaryOperator,  # abstract base; all usable subclasses are covered
        pybamm.BinaryOperator,  # abstract base; all usable subclasses are covered
        pybamm.SpatialOperator,  # abstract base for pre-discretisation spatial ops
        pybamm.BoundaryOperator,  # abstract base; BoundaryValue + BoundaryGradient covered
        pybamm.BaseIndefiniteIntegral,  # abstract base; IndefiniteIntegral + Backward covered
        pybamm.UpwindDownwind,  # abstract base; Upwind and Downwind strategies cover it
        pybamm.VariableBase,  # abstract base; Variable and VariableDot cover it
        pybamm.StateVectorBase,  # abstract base; StateVector + StateVectorDot cover it
        pybamm.Function,  # to_json() raises NotImplementedError — only SpecificFunction subclasses round-trip
        pybamm.SpecificFunction,  # base for named funcs; direct instantiation not useful
        pybamm.Broadcast,  # abstract base; PrimaryBroadcast/Secondary/Full covered
        pybamm.Integral,  # base for domain-constrained integrals; heavyweight constructor
        pybamm.IndependentVariable,  # abstract base; Time + SpatialVariable covered
        _BaseAverageClass,  # abstract base; XAverage/ZAverage/etc. cover it
        pybamm.DomainConcatenation,  # constructor requires full_mesh (pybamm.Mesh); not user-constructible
    }
)

# Concrete classes known not to round-trip (empty: all concrete subclasses round-trip #5548).
_KNOWN_FAILING: frozenset[type[pybamm.Symbol]] = frozenset()

# Union consumed by the coverage meta-test: every concrete Symbol subclass must
# be in _STRATEGIES or excluded here.
_EXEMPT: frozenset[type[pybamm.Symbol]] = _NOT_ROUND_TRIPPABLE | _KNOWN_FAILING


# Leaf classes — excluded from the branch list in ``symbols()`` because
# they are handled by ``_leaves()`` and need no child strategy.
_LEAF_CLASSES: frozenset[type[pybamm.Symbol]] = frozenset(
    {
        pybamm.Scalar,
        pybamm.Variable,
        pybamm.Parameter,
        pybamm.Time,
        pybamm.InputParameter,
    }
)


def symbols(max_leaves: int = 6) -> st.SearchStrategy[pybamm.Symbol]:
    """Recursive Symbol-tree strategy.

    Uses ``st.recursive`` to build trees up to ``max_leaves`` deep. Each
    branch is one of the registered entries in ``_STRATEGIES``.
    """
    branch_keys = [k for k in _STRATEGIES if k not in _LEAF_CLASSES]

    def extend(children: st.SearchStrategy[pybamm.Symbol]):
        return st.one_of(*[_STRATEGIES[cls](children) for cls in branch_keys])

    return st.recursive(_leaves(), extend, max_leaves=max_leaves)
