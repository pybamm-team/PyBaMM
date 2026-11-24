#
# Flat (hash-consed) node-table serialisation of expression trees
#
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence

import numpy as np

from pybamm.expression_tree.operations import serialise_kernel
from pybamm.expression_tree.operations.serialise_kernel import TAG

_REF = "$ref"


class FlatEncoder:
    """
    Encode objects into a table of unique nodes.

    Each distinct term is stored once in ``nodes``; every use of it is encoded as
    ``{"$ref": index}``. Nodes only reference earlier entries, and their type tag
    and domains are indices into the shared ``types`` and ``domains`` tables.

    Parameters
    ----------
    default : callable, optional
        Fallback passed to ``json.dumps`` for values that are not JSON-native.
    """

    def __init__(self, default: Callable | None = None):
        self.nodes: list[dict] = []
        self._default = default
        self._types: dict[str, int] = {}
        self._domains: dict[str, int] = {}
        self._interned_domains: dict[int, int] = {}
        self._by_object: dict[int, dict] = {}
        self._by_content: dict[bytes, int] = {}
        # keeps memoised objects alive so their id() cannot be reused
        self._seen: list = []

    def encode(self, obj):
        """Encode ``obj``; registered objects become a reference into the table."""
        if obj is None or isinstance(obj, bool | int | str):
            return obj
        if isinstance(obj, float | np.ndarray | slice | type):
            return serialise_kernel.encode(obj)

        reference = self._by_object.get(id(obj))
        if reference is not None:
            return reference
        codec = serialise_kernel._lookup_codec(type(obj))
        if codec is not None:
            node = codec.to_json(obj, self.encode)
            path = serialise_kernel._class_path(type(obj))
            node[TAG] = self._types.setdefault(path, len(self._types))
            if "domains" in node:
                node["domains"] = self._domains_index(obj, node["domains"])
            # a digest, so that nodes holding large arrays are not kept twice
            content = json.dumps(node, sort_keys=True, default=self._default)
            key = hashlib.blake2b(content.encode(), digest_size=16).digest()
            position = self._by_content.get(key)
            if position is None:
                position = self._by_content[key] = len(self.nodes)
                self.nodes.append(node)
            reference = self._by_object[id(obj)] = {_REF: position}
            self._seen.append(obj)
            return reference

        if isinstance(obj, tuple):
            return {TAG: "builtins.tuple", "items": [self.encode(x) for x in obj]}
        if isinstance(obj, list):
            return [self.encode(x) for x in obj]
        if isinstance(obj, dict):
            return {key: self.encode(value) for key, value in obj.items()}
        return serialise_kernel.encode(obj)  # raises the kernel's no-codec error

    def _domains_index(self, obj, domains) -> int:
        # symbol domains are interned objects, so identity finds the entry
        interned = getattr(obj, "_domains", None)
        if interned is not None and interned == domains:
            position = self._interned_domains.get(id(interned))
            if position is not None:
                return position
        key = json.dumps(domains)
        position = self._domains.setdefault(key, len(self._domains))
        if interned is not None and interned == domains:
            self._interned_domains[id(interned)] = position
        return position

    def tables(self) -> dict:
        """The ``types``, ``domains`` and ``nodes`` tables built so far."""
        return {
            "types": list(self._types),
            "domains": [json.loads(key) for key in self._domains],
            "nodes": self.nodes,
        }


def resolve_reference(reference, nodes: Sequence):
    """
    The entry of ``nodes`` that ``{"$ref": reference}`` points to.

    Raises
    ------
    SerialisationError
        If ``reference`` is not the index of an entry of ``nodes``. Negative
        indices and booleans are rejected rather than read as Python indices.
    """
    if type(reference) is not int or not 0 <= reference < len(nodes):
        raise serialise_kernel.SerialisationError(
            f"Invalid node reference {reference!r}: expected an index below "
            f"{len(nodes)}"
        )
    return nodes[reference]


def decode_nodes(document: dict) -> list:
    """
    Build every node of a flat document, each exactly once.

    Parameters
    ----------
    document : dict
        A document holding the ``types``, ``domains`` and ``nodes`` tables
        written by :class:`FlatEncoder`.

    Returns
    -------
    list
        The decoded object for each entry of ``nodes``, so ``{"$ref": i}`` is
        entry ``i``. Empty if the document has no node table.
    """
    if "nodes" not in document:
        return []
    try:
        types, domains = document["types"], document["domains"]
    except KeyError as error:
        raise serialise_kernel.SerialisationError(
            f"Flat document is missing the {error.args[0]!r} table"
        ) from error
    classes = [serialise_kernel._resolve_class(path) for path in types]
    objects: list = []

    def decode(raw):
        if isinstance(raw, list):
            return [decode(item) for item in raw]
        if not isinstance(raw, dict):
            return raw
        if _REF in raw:
            return resolve_reference(raw[_REF], objects)
        tag = raw.get(TAG)
        if tag == "builtins.tuple":
            return tuple(decode(item) for item in raw["items"])
        if tag is None:
            return {key: decode(value) for key, value in raw.items()}
        return serialise_kernel.decode(raw)

    for node in document["nodes"]:
        node = dict(node)
        cls = classes[node[TAG]]
        node[TAG] = types[node[TAG]]
        if "domains" in node:
            node["domains"] = domains[node["domains"]]
        codec = serialise_kernel._lookup_codec(cls)
        if codec is None:
            raise serialise_kernel.SerialisationError(
                f"No codec to decode '{node[TAG]}'"
            )
        objects.append(codec.from_json(node, decode, cls))
    return objects


def to_nested(document: dict):
    """
    Expand a flat document into the nested kernel JSON it is equivalent to.

    Parameters
    ----------
    document : dict
        A flat document: ``types``, ``domains`` and ``nodes`` tables plus any
        other entries, which may contain references.

    Returns
    -------
    dict
        The other entries with every reference replaced by its nested node.
    """
    types, domains = document["types"], document["domains"]
    expanded: list[dict] = []

    def resolve(value):
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, dict):
            return value
        if _REF in value:
            return resolve_reference(value[_REF], expanded)
        return {key: resolve(item) for key, item in value.items()}

    for node in document["nodes"]:
        node = resolve(node)
        node[TAG] = types[node[TAG]]
        if "domains" in node:
            node["domains"] = domains[node["domains"]]
        expanded.append(node)
    return {
        key: resolve(value)
        for key, value in document.items()
        if key not in ("types", "domains", "nodes")
    }
