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


def test_serialise_then_load_round_trips_spm(built_spm, tmp_path):
    s = Serialise()
    model_json = s.serialise_model(built_spm.built_model, mesh=built_spm.mesh)
    # dict in-memory path
    reloaded = s.load_model(
        json.loads(json.dumps(model_json)), battery_model=pybamm.lithium_ion.SPM()
    )
    assert reloaded.concatenated_rhs.id == built_spm.built_model.concatenated_rhs.id
    assert reloaded.concatenated_initial_conditions.id == (
        built_spm.built_model.concatenated_initial_conditions.id
    )
    assert len(reloaded.events) == len(built_spm.built_model.events)
    for a, b in zip(reloaded.events, built_spm.built_model.events, strict=True):
        assert a.expression.id == b.expression.id


def test_load_model_resolves_class_from_canonical_tag(built_spm):
    # No battery_model passed -> class resolved from the $type tag.
    s = Serialise()
    model_json = s.serialise_model(built_spm.built_model, mesh=built_spm.mesh)
    reloaded = s.load_model(json.loads(json.dumps(model_json)))
    assert isinstance(reloaded, pybamm.lithium_ion.SPM)
