"""JSON written by the previous serialiser must keep loading (#5548).

Fixtures are pinned bytes captured from the previous on-disk format, so a
decode regression shows up here.
"""

from __future__ import annotations

import gzip
import json
import pathlib

import pytest

import pybamm
from pybamm.expression_tree.operations.serialise import (
    Serialise,
    convert_symbol_from_json,
)

_FIX = pathlib.Path(__file__).parent / "fixtures" / "legacy"


def _load(name):
    path = _FIX / name
    if name.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


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


def _assert_legacy_spm_loads(data):
    model = Serialise().load_model(data, battery_model=pybamm.lithium_ion.SPM())
    assert isinstance(model, pybamm.lithium_ion.SPM)
    assert model.concatenated_rhs is not None
    assert model.concatenated_initial_conditions is not None


def test_legacy_discretised_model_coarse_loads():
    """The pre-refactor py/object discretised model must still load through the
    kernel-based load_model + legacy relocation.

    Coarse-mesh variant, small enough to track in git (gzipped), so this path
    is always exercised, including CI.
    """
    _assert_legacy_spm_loads(_load("discretised_model_coarse.json.gz"))


def test_legacy_discretised_model_loads():
    """Default-mesh variant of the fixture above.

    ~19 MB and git-ignored; regenerate from a pre-refactor PyBaMM checkout (the
    legacy writer no longer exists -- see generate.py for the recipe). Skipped
    when the local fixture is absent (e.g. CI).
    """
    if not (_FIX / "discretised_model.json").exists():
        pytest.skip("discretised_model.json fixture absent (git-ignored, local-only)")
    _assert_legacy_spm_loads(_load("discretised_model.json"))
