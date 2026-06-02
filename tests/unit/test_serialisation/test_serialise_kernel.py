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
