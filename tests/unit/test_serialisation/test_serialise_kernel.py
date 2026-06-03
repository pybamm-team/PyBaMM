from __future__ import annotations

import json

import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.operations import serialise_kernel as sk


def _rt(tree):
    # Round-trip through real JSON text (not in-memory) so a hook that leaves a
    # non-JSON-native value (raw slice/ndarray/Symbol) is caught by json.dumps.
    return sk.decode(json.loads(json.dumps(sk.encode(tree))))


def test_class_path_round_trips_a_pybamm_class():
    path = sk._class_path(pybamm.Scalar)
    assert sk._resolve_class(path) is pybamm.Scalar


def test_resolve_unknown_class_raises():
    with pytest.raises(sk.SerialisationError):
        sk._resolve_class("pybamm.NoSuchClassXYZ")


def test_float_inf_nan_round_trip():
    for val in (float("inf"), float("-inf"), 3.5):
        assert sk._decode_leaf(sk._encode_float(val)) == val
    nan = sk._decode_leaf(sk._encode_float(float("nan")))
    assert nan != nan  # NaN


def test_ndarray_round_trip():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    node = sk.encode(arr)
    assert np.array_equal(sk.decode(node), arr)


def test_slice_round_trip():
    node = sk.encode(slice(1, 5, 2))
    assert sk.decode(node) == slice(1, 5, 2)


def test_tuple_round_trips_as_tuple_not_list():
    node = sk.encode((1, 2, 3))
    out = sk.decode(node)
    assert out == (1, 2, 3) and isinstance(out, tuple)


def test_encode_native_leaves_pass_through():
    assert sk.encode(3) == 3
    assert sk.encode("x") == "x"
    assert sk.encode(True) is True
    assert sk.encode(None) is None


def test_containers_recurse():
    assert sk.encode([1, "a", None]) == [1, "a", None]
    assert sk.decode(sk.encode({"k": [1, 2]})) == {"k": [1, 2]}


def test_class_reference_round_trip():
    node = sk.encode(pybamm.Scalar)  # a class, not an instance
    assert sk.decode(node) is pybamm.Scalar


def test_encode_unknown_object_raises():
    class Foreign:
        pass

    with pytest.raises(sk.SerialisationError, match="Cannot serialise"):
        sk.encode(Foreign())


# Safe-or-loud raise sentinels: low-level resolve/decode branches must fail loudly.
def test_resolve_non_class_path_raises():
    # A path resolving to a non-class object (here a builtin function) must raise,
    # not return the object.
    with pytest.raises(sk.SerialisationError, match="did not resolve to a class"):
        sk._resolve_class("builtins.len")


def test_decode_unknown_float_sentinel_raises():
    with pytest.raises(sk.SerialisationError, match="Unknown float sentinel"):
        sk._decode_leaf({sk.TAG: "builtins.float", "value": "Bogus"})


def test_decode_unsupported_type_raises():
    # A value that is neither a JSON scalar, list, nor dict has no decoding.
    with pytest.raises(sk.SerialisationError, match="Cannot decode value of type"):
        sk.decode({1, 2, 3})


# Synthetic helper classes for DefaultCodec tests
class _Clean(pybamm.Symbol):
    def __init__(self, child, mode="a"):
        super().__init__("clean", children=[child])
        self.mode = mode


class _Renamed(pybamm.Symbol):
    def __init__(self, child, mode):
        super().__init__("renamed", children=[child])
        self._cfg = mode  # stored under a name that is neither mode nor _mode


def test_default_codec_captures_clean_param():
    codec = sk.DefaultCodec()
    obj = _Clean(pybamm.Scalar(1.0), mode="z")
    node = codec.to_json(obj, sk.encode)
    assert node["mode"] == "z"
    assert "children" in node


def test_default_codec_captures_param_with_default_value():
    # A non-default value of a defaulted param must still be captured (fixes the
    # BoundaryIntegral region-reset class of bug).
    codec = sk.DefaultCodec()
    node = codec.to_json(_Clean(pybamm.Scalar(1.0), mode="a"), sk.encode)
    assert node["mode"] == "a"


def test_default_codec_raises_on_unresolvable_required_param():
    codec = sk.DefaultCodec()
    with pytest.raises(sk.SerialisationError, match="mode"):
        codec.to_json(_Renamed(pybamm.Scalar(1.0), "x"), sk.encode)


def test_default_codec_from_json_ignores_extra_legacy_keys():
    # Legacy JSON always wrote a "name" key; a clean class whose __init__ has no
    # `name` param must still reconstruct (the key is ignored, not forwarded as
    # an unexpected kwarg). Guards legacy compatibility (#5548).
    codec = sk.DefaultCodec()
    node = codec.to_json(_Clean(pybamm.Scalar(1.0), mode="q"), sk.encode)
    node[sk.TAG] = sk._class_path(_Clean)
    node["name"] = "legacy-name"  # spurious key from the old fallback
    node["entries_string"] = "noise"  # another legacy-only key
    restored = codec.from_json(node, sk.decode, _Clean)
    assert restored.mode == "q"


def test_default_codec_round_trips_singular_domain_param():
    # A class whose __init__ takes `domain` (not `domains`) must still recover
    # its domain: to_json emits the full `domains` dict, from_json maps it onto
    # the singular `domain` param. Guards the CoupledVariable silent-drop bug --
    # CoupledVariable is the one concrete Symbol that reaches DefaultCodec.
    codec = sk.DefaultCodec()
    cv = pybamm.CoupledVariable("c", domain=["negative electrode"])
    node = codec.to_json(cv, sk.encode)
    node[sk.TAG] = sk._class_path(pybamm.CoupledVariable)
    restored = codec.from_json(node, sk.decode, pybamm.CoupledVariable)
    assert restored.domain == ["negative electrode"]


# HookCodec + codec lookup
def test_lookup_returns_hook_codec_for_overriding_class():
    # Index overrides to_json/_from_json -> HookCodec.
    codec = sk._lookup_codec(pybamm.Index)
    assert isinstance(codec, sk.HookCodec)


def test_lookup_returns_hook_codec_for_inherited_base_from_json():
    # Negate defines neither method itself, but inherits UnaryOperator._from_json
    # which is NOT Symbol._from_json, so _overrides_hooks is True -> HookCodec.
    codec = sk._lookup_codec(pybamm.Negate)
    assert isinstance(codec, sk.HookCodec)


def test_lookup_returns_default_for_class_overriding_neither():
    # CoupledVariable subclasses Symbol directly and overrides neither method, so
    # it is the one concrete Symbol routed to DefaultCodec.
    codec = sk._lookup_codec(pybamm.CoupledVariable)
    assert isinstance(codec, sk.DefaultCodec)


def test_lookup_returns_none_for_foreign_type():
    class Foreign:
        pass

    assert sk._lookup_codec(Foreign) is None


def test_hook_guard_passes_when_scalar_param_is_emitted():
    class _Good(pybamm.Symbol):
        def __init__(self, child, mode="a"):
            super().__init__("good", children=[child])
            self.mode = mode

        def to_json(self):
            return {"name": self.name, "domains": self.domains, "mode": self.mode}

        @classmethod
        def _from_json(cls, snippet):
            return cls(snippet["children"][0], snippet["mode"])

    node = sk.HookCodec().to_json(_Good(pybamm.Scalar(1.0), mode="z"), sk.encode)
    assert node["mode"] == "z"


def test_hook_guard_raises_when_scalar_param_is_dropped():
    class _Leaky(pybamm.Symbol):
        def __init__(self, child, mode="a"):
            super().__init__("leaky", children=[child])
            self.mode = mode

        def to_json(self):
            return {"name": self.name, "domains": self.domains}  # drops `mode`

        @classmethod
        def _from_json(cls, snippet):
            return cls(snippet["children"][0])

    with pytest.raises(sk.SerialisationError, match="mode"):
        sk.encode(_Leaky(pybamm.Scalar(1.0), mode="z"))


def test_hook_guard_allows_symbol_valued_param_via_children():
    class _Wrapper(pybamm.Symbol):
        def __init__(self, child, extra):
            self.extra = extra  # a Symbol
            super().__init__("wrap", children=[child, extra])

        def to_json(self):
            return {"name": self.name, "domains": self.domains}

        @classmethod
        def _from_json(cls, snippet):
            return cls(*snippet["children"])

    sk.encode(_Wrapper(pybamm.Scalar(1.0), pybamm.Scalar(2.0)))  # no raise


def test_hook_guard_allows_declared_derived_param():
    class _Cached(pybamm.Symbol):
        _serialise_derived_params = frozenset({"cache"})

        def __init__(self, child, cache=None):
            super().__init__("cached", children=[child])
            self.cache = cache if cache is not None else "derived"

        def to_json(self):
            return {"name": self.name, "domains": self.domains}

        @classmethod
        def _from_json(cls, snippet):
            return cls(snippet["children"][0])

    sk.encode(_Cached(pybamm.Scalar(1.0)))  # no raise


# Module-level (not function-local) so decode can resolve it by import path during
# the round-trip below; a `<locals>` class is not importable.
class _OwnChildren(pybamm.Symbol):
    def __init__(self, child, extra):
        self.extra = extra
        super().__init__("own", children=[child])  # extra not a symbol child

    def to_json(self):
        return {
            "name": self.name,
            "domains": self.domains,
            "children": [self.children[0], self.extra],
        }

    @classmethod
    def _from_json(cls, snippet):
        return cls(snippet["children"][0], snippet["children"][1])


def test_hook_codec_encodes_hook_supplied_children():
    # A hook whose to_json returns its OWN children list (here [child, extra],
    # where extra is NOT in obj.children) must have THAT list encoded and guarded,
    # not obj.children. This is the contract SizeAverage/Variable/EvaluateAt rely on.
    a, b = pybamm.Scalar(1.0), pybamm.Scalar(2.0)
    obj = _OwnChildren(a, b)
    node = sk.HookCodec().to_json(obj, sk.encode)
    # the hook-supplied list (2 items) is encoded, not obj.children (1 item)
    assert len(node["children"]) == 2
    restored = sk.decode(sk.encode(obj))
    assert restored.children[0].id == a.id and restored.extra.id == b.id


# normalise_legacy
def test_normalise_legacy_py_object():
    legacy = {"py/object": "pybamm.Scalar", "py/id": 7, "value": 1.0}
    out = sk.normalise_legacy(legacy)
    assert out[sk.TAG] == "pybamm.Scalar"
    # py/id and py/object are consumed, not carried forward
    assert "py/id" not in out and "py/object" not in out
    # read-only: the input node is not mutated
    assert legacy == {"py/object": "pybamm.Scalar", "py/id": 7, "value": 1.0}


def test_normalise_legacy_bare_type_resolves_to_pybamm():
    legacy = {"type": "Scalar", "value": 1.0}
    out = sk.normalise_legacy(legacy)
    assert sk._resolve_class(out[sk.TAG]) is pybamm.Scalar


def test_normalise_legacy_class_module():
    legacy = {
        "class": "Uniform1DSubMesh",
        "module": "pybamm.meshes.one_dimensional_submeshes",
    }
    out = sk.normalise_legacy(legacy)
    assert out[sk.TAG] == "type"
    assert out["class"] == "pybamm.meshes.one_dimensional_submeshes.Uniform1DSubMesh"


def test_normalise_legacy_passthrough_canonical():
    canonical = {sk.TAG: "pybamm.Scalar", "value": 1.0}
    assert sk.normalise_legacy(canonical) is canonical


# Anti-reintroduction trap tests
def test_trap_default_codec_unhandleable_required_arg_raises_loudly():
    """A Symbol subclass with a renamed attribute and no hook (-> DefaultCodec)
    must RAISE on encode, never silently drop. Red here = DefaultCodec softened
    back toward silent loss, breaking the safe-or-loud invariant.
    """

    class _Trap(pybamm.Symbol):
        def __init__(self, child, secret):
            super().__init__("trap", children=[child])
            self._hidden = secret  # not secret / _secret -> unresolvable

    with pytest.raises(sk.SerialisationError, match="secret"):
        sk.encode(_Trap(pybamm.Scalar(1.0), "x"))


def test_trap_hook_codec_inherited_base_drops_scalar_raises_loudly():
    """The inherited-hook leak (#5548's structural cause), locked in. A
    `UnaryOperator` subclass inherits `Symbol.to_json` (emits name/children/domains,
    not `knob`) and `UnaryOperator._from_json` (rebuilds via __new__, routes to
    HookCodec). The extra scalar `knob` is neither emitted, Symbol-valued, nor
    declared derived -> the HookCodec coverage guard must RAISE. This is exactly the
    shape `Magnitude.direction` had before its hook. Red here = the guard was
    weakened and silent field-drop is back.
    """

    class _TrapUnary(pybamm.UnaryOperator):
        def __init__(self, child, knob="a"):
            super().__init__("trap_unary", child)
            self.knob = knob

    with pytest.raises(sk.SerialisationError, match="knob"):
        sk.encode(_TrapUnary(pybamm.Scalar(1.0), knob="z"))


@pytest.mark.parametrize(
    "tree",
    [
        pybamm.Variable("v", scale=pybamm.Scalar(2.0)),
        pybamm.Variable("v", reference=pybamm.Scalar(1.0)),
        pybamm.VariableDot("v'", domains={"primary": ["negative electrode"]}),
        pybamm.SpatialVariableEdge("x", domain=["negative electrode"]),
    ],
)
def test_variable_family_round_trip(tree):
    assert _rt(tree).id == tree.id


@pytest.mark.parametrize(
    "tree",
    [
        pybamm.BoundaryGradient(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}), "left"
        ),
        pybamm.BoundaryMeshSize(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}), "left"
        ),
        pybamm.BoundaryIntegral(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}),
            region="negative tab",
        ),
        pybamm.UpwindDownwind2D(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}), "x", "y"
        ),
        pybamm.NodeToEdge2D(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}), "lr"
        ),
        pybamm.DeltaFunction(
            pybamm.Variable("u", domains={"primary": ["negative particle"]}),
            "left",
            "negative electrode",
        ),
        pybamm.OneDimensionalIntegral(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}),
            ["negative electrode"],
            "x",
        ),
        # carry scalar constructor args (side/order, vector_type) that must survive round-trip:
        pybamm.BoundaryValue(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}), "left"
        ),
        pybamm.DefiniteIntegralVector(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]})
        ),
    ],
)
def test_spatial_native_arg_round_trip(tree):
    assert _rt(tree).id == tree.id


def test_size_average_round_trip():
    child = pybamm.Variable("u", domains={"primary": ["negative particle size"]})
    assert (
        _rt(pybamm.SizeAverage(child, pybamm.Scalar(1.0))).id
        == pybamm.SizeAverage(child, pybamm.Scalar(1.0)).id
    )


@pytest.mark.parametrize(
    "tree",
    [
        pybamm.EvaluateAt(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}), 0.5
        ),
        # symbolic position (a Symbol, not a number): must route through children
        pybamm.EvaluateAt(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}),
            pybamm.Scalar(0.5),
        ),
        pybamm.BackwardIndefiniteIntegral(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}),
            pybamm.SpatialVariable("x", domain=["negative electrode"]),
        ),
        pybamm.ExplicitTimeIntegral(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}),
            pybamm.Scalar(0.0),
        ),
    ],
)
def test_symbol_valued_spatial_round_trip(tree):
    assert _rt(tree).id == tree.id


@pytest.mark.parametrize(
    "tree",
    [
        pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1.0), "negative electrode"),
        pybamm.SecondaryBroadcast(
            pybamm.PrimaryBroadcast(pybamm.Scalar(1.0), "negative particle"),
            "negative electrode",
        ),
        pybamm.TertiaryBroadcast(
            pybamm.Variable(
                "v",
                domains={
                    "primary": ["negative particle"],
                    "secondary": ["negative electrode"],
                },
            ),
            "current collector",
        ),
        pybamm.FullBroadcast(
            pybamm.Scalar(1.0),
            broadcast_domains={"primary": ["negative electrode"]},
            name="custom",
        ),
        pybamm.FullBroadcastToEdges(
            pybamm.Scalar(1.0),
            broadcast_domains={"primary": ["negative electrode"]},
        ),
        # custom name on a non-Full broadcast -- locks the "restore name on all" decision:
        pybamm.PrimaryBroadcast(
            pybamm.Scalar(1.0), "negative electrode", name="custom"
        ),
    ],
)
def test_broadcast_round_trip_preserves_subclass(tree):
    restored = _rt(tree)
    assert type(restored) is type(tree)
    assert restored.id == tree.id


def test_tensor_field_round_trip():
    tree = pybamm.TensorField([pybamm.Scalar(1.0), pybamm.Scalar(2.0)])
    assert _rt(tree).id == tree.id


def test_tensor_field_rank_2_round_trip():
    tree = pybamm.TensorField(
        [
            [pybamm.Scalar(1.0), pybamm.Scalar(2.0)],
            [pybamm.Scalar(3.0), pybamm.Scalar(4.0)],
        ]
    )
    restored = _rt(tree)
    assert restored.rank == 2
    assert restored.shape == (2, 2)
    assert restored.id == tree.id


def test_vector_field_round_trip():
    leaf = pybamm.Variable("u", domains={"primary": ["negative electrode"]})
    tree = pybamm.VectorField(leaf, leaf)
    restored = _rt(tree)
    assert type(restored) is pybamm.VectorField
    assert restored.id == tree.id


# Droppers + Scalar/Interpolant/Concatenation
@pytest.mark.parametrize(
    "tree",
    [
        pybamm.Magnitude(
            pybamm.Variable("u", domains={"primary": ["negative electrode"]}), "x"
        ),
        pybamm.Scalar(float("inf")),
        pybamm.Scalar(float("-inf")),
    ],
)
def test_dropper_and_scalar_round_trip(tree):
    assert _rt(tree).id == tree.id


def test_magnitude_restores_direction():
    tree = pybamm.Magnitude(
        pybamm.Variable("u", domains={"primary": ["negative electrode"]}), "x"
    )
    assert _rt(tree).direction == "x"


def test_discrete_time_data_round_trip():
    tree = pybamm.DiscreteTimeData(
        np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "dtd"
    )
    restored = _rt(tree)
    assert restored.id == tree.id
    assert np.array_equal(restored.y, tree.y)


def test_discrete_time_sum_round_trip():
    data = pybamm.DiscreteTimeData(np.array([0.0, 1.0]), np.array([1.0, 2.0]), "dtd")
    tree = pybamm.DiscreteTimeSum(data)
    restored = _rt(tree)
    assert restored.id == tree.id
    assert restored.data is not None  # __init__ ran, self.data re-derived


def _neg_sep_vars():
    return (
        pybamm.Variable("a", domains={"primary": ["negative electrode"]}),
        pybamm.Variable("b", domains={"primary": ["separator"]}),
    )


def _neg_sep_nonvars():
    return (
        pybamm.FullBroadcast(
            pybamm.Scalar(1.0), broadcast_domains={"primary": ["negative electrode"]}
        ),
        pybamm.FullBroadcast(
            pybamm.Scalar(2.0), broadcast_domains={"primary": ["separator"]}
        ),
    )


@pytest.mark.parametrize(
    "tree",
    [
        pybamm.Concatenation(*_neg_sep_nonvars()),
        pybamm.ConcatenationVariable(*_neg_sep_vars()),
        pybamm.SparseStack(*_neg_sep_vars()),
    ],
)
def test_concatenation_family_round_trip(tree):
    assert _rt(tree).id == tree.id


def test_concatenation_family_sparse_stack_rederives_concat_fun():
    tree = pybamm.SparseStack(*_neg_sep_vars())
    assert _rt(tree).concatenation_function is not None


@pytest.mark.parametrize(
    "tree",
    [
        pybamm.Index(pybamm.StateVector(slice(0, 1)), 0, check_size=False),
        pybamm.Array(np.array([[1.0, 2.0], [3.0, 4.0]])),
        pybamm.Matrix(np.array([[1.0, 0.0], [0.0, 1.0]])),
        pybamm.Vector(np.array([1.0, 2.0, 3.0])),
        pybamm.StateVector(slice(0, 2)),
        pybamm.StateVectorDot(slice(0, 2)),
        pybamm.Scalar(1.0) < pybamm.Scalar(2.0),  # EqualHeaviside / NotEqualHeaviside
    ],
)
def test_already_correct_classes_round_trip(tree):
    assert _rt(tree).id == tree.id


# Function operand opt-outs
@pytest.mark.parametrize(
    "tree",
    [
        pybamm.Arcsinh2(pybamm.Scalar(0.5), pybamm.Scalar(1.0)),
        pybamm.RegPower(pybamm.Scalar(0.5), pybamm.Scalar(2.0)),
    ],
)
def test_function_operands_carried_via_children_round_trip(tree):
    # a/b (Arcsinh2) and base/exponent/scale (RegPower) are Symbol operands passed
    # as children; eps/delta are emitted scalars. The guard must not false-positive
    # on the operand params (they are carried via children).
    assert _rt(tree).id == tree.id
