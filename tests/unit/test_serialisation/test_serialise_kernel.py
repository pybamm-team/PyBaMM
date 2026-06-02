from __future__ import annotations

import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.operations import serialise_kernel as sk


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


# ---------------------------------------------------------------------------
# Synthetic helper classes for DefaultCodec tests
# ---------------------------------------------------------------------------


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
    # The old switch always wrote a "name" key; a clean class whose __init__ has
    # no `name` param must still reconstruct (the key is ignored, not forwarded
    # as an unexpected kwarg). Guards legacy compatibility (#5548 spec).
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


# ---------------------------------------------------------------------------
# HookCodec + codec lookup (Task 1.5)
# ---------------------------------------------------------------------------


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
