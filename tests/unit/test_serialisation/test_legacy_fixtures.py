"""JSON written by the previous serialiser must keep loading (#5548).

Fixtures are pinned bytes captured from the previous on-disk format, so a
decode regression shows up here.
"""

from __future__ import annotations

import json
import pathlib

import pybamm
from pybamm.expression_tree.operations.serialise import (
    Serialise,
    convert_symbol_from_json,
)

_FIX = pathlib.Path(__file__).parent / "fixtures" / "legacy"


def _load(name):
    return json.loads((_FIX / name).read_text())


def test_legacy_compact_symbols_load():
    cases = _load("compact_symbols.json")
    assert cases  # non-empty
    for label, node in cases.items():
        tree = convert_symbol_from_json(node)
        assert isinstance(tree, pybamm.Symbol), label


def test_legacy_submesh_types_loads():
    restored = Serialise.load_submesh_types(_load("submesh_types.json"))
    assert isinstance(restored, dict) and restored
    assert all(isinstance(v, pybamm.MeshGenerator) for v in restored.values())
