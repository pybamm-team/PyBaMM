from __future__ import annotations

import json

import pytest

import pybamm
from pybamm.expression_tree.operations.serialise import Serialise


@pytest.fixture(scope="module")
def built_spm():
    model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(model)
    sim.build()
    return sim


def test_serialise_model_emits_canonical_tags(built_spm):
    out = Serialise().serialise_model(built_spm.built_model, mesh=built_spm.mesh)
    # top-level model identity + every serialised tree uses $type, never py/object
    assert "$type" in out
    assert "py/object" not in json.dumps(out)
    # the symbol trees are real kernel nodes
    assert out["concatenated_rhs"]["$type"].startswith("pybamm")
    assert out["mesh"]["$type"].endswith("Mesh")
    # round-trips as plain JSON text (no non-native leftovers)
    json.loads(json.dumps(out))
