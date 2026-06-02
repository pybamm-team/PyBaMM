from __future__ import annotations

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
