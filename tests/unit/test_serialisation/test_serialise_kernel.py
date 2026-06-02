from __future__ import annotations

import pytest

import pybamm
from pybamm.expression_tree.operations import serialise_kernel as sk


def test_class_path_round_trips_a_pybamm_class():
    path = sk._class_path(pybamm.Scalar)
    assert sk._resolve_class(path) is pybamm.Scalar


def test_resolve_unknown_class_raises():
    with pytest.raises(sk.SerialisationError):
        sk._resolve_class("pybamm.NoSuchClassXYZ")
