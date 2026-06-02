from __future__ import annotations

import json

import pytest

import pybamm
from pybamm.expression_tree.operations import serialise_kernel as sk


class _DictBacked(dict):
    """A dict subclass with a hook -- stands in for Mesh in the engine-ordering test."""

    def to_json(self):
        return {"marker": "kept"}

    @classmethod
    def _from_json(cls, snippet):
        inst = cls()
        inst["marker"] = snippet["marker"]
        return inst


def test_registered_dict_subclass_dispatches_to_codec_not_dict_branch(monkeypatch):
    # Register _DictBacked as a known base for the duration of the test.
    monkeypatch.setattr(sk, "_KNOWN_BASES", (*sk._KNOWN_BASES, _DictBacked))
    obj = _DictBacked()
    node = sk.encode(obj)
    assert node[sk.TAG] == sk._class_path(_DictBacked)
    assert node["marker"] == "kept"


def test_plain_dict_still_recurses():
    assert sk.encode({"a": [1, 2]}) == {"a": [1, 2]}
    assert sk.decode(sk.encode({"a": [1, 2]})) == {"a": [1, 2]}


def test_known_bases_resolve_to_hook_codec():
    # Event/Mesh/SubMesh all define their own to_json/_from_json -> HookCodec.
    for cls in (pybamm.Event, pybamm.Mesh, pybamm.Uniform1DSubMesh):
        assert isinstance(sk._lookup_codec(cls), sk.HookCodec), cls


def test_unregistered_non_symbol_returns_none():
    # An unregistered class is not a known base -> no codec (encode then raises
    # the generic "no codec registered" message elsewhere).
    class _NoHookBase:
        pass

    assert sk._lookup_codec(_NoHookBase) is None


def test_registered_non_symbol_base_without_hook_raises(monkeypatch):
    # A registered non-Symbol base with no hook must raise loudly rather than fall
    # through to the Symbol-only DefaultCodec.
    class _NoHookBase:
        pass

    monkeypatch.setattr(sk, "_KNOWN_BASES", (*sk._KNOWN_BASES, _NoHookBase))
    with pytest.raises(sk.SerialisationError, match="no to_json"):
        sk._lookup_codec(_NoHookBase)


def test_event_round_trip_through_kernel():
    expr = pybamm.Variable(
        "u", domains={"primary": ["negative electrode"]}
    ) - pybamm.Scalar(1.0)
    event = pybamm.Event("my event", expr, pybamm.EventType.TERMINATION)
    restored = sk.decode(json.loads(json.dumps(sk.encode(event))))
    assert isinstance(restored, pybamm.Event)
    assert restored.name == "my event"
    assert restored.event_type == pybamm.EventType.TERMINATION
    assert restored.expression.id == expr.id


def test_event_from_json_tolerates_legacy_expression_field():
    # Old discretised files carried "expression" as a sibling, not in children.
    expr = pybamm.Scalar(2.0)
    snippet = {
        "name": "legacy",
        "event_type": ["EventType.TERMINATION", 0],
        "expression": expr,
    }  # decoded expression, legacy shape
    restored = pybamm.Event._from_json(snippet)
    assert restored.expression.id == expr.id
