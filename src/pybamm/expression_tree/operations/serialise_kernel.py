"""Unified safe-or-loud serialisation kernel.

One encode/decode engine for pybamm serialisable objects. Dispatches by type to
per-class ``to_json``/``_from_json`` hooks or an introspection default, and
raises :class:`SerialisationError` rather than silently dropping any value.
"""

from __future__ import annotations

import importlib

import numpy as np

TAG = "$type"  # canonical key holding a dotted "module.ClassName" path


class SerialisationError(Exception):
    """Raised when a value cannot be serialised or reconstructed safely."""


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_class(path: str) -> type:
    module_name, _, qualname = path.rpartition(".")
    try:
        obj: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError, ValueError) as err:
        raise SerialisationError(f"Cannot resolve class '{path}'") from err
    if not isinstance(obj, type):
        raise SerialisationError(f"'{path}' did not resolve to a class")
    return obj


_FLOAT_SENTINELS = {float("inf"): "Infinity", float("-inf"): "-Infinity"}
_FLOAT_SENTINELS_INV = {
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
    "NaN": float("nan"),
}


def _encode_float(value: float):
    if value != value:  # NaN
        return {TAG: "builtins.float", "value": "NaN"}
    if value in _FLOAT_SENTINELS:
        return {TAG: "builtins.float", "value": _FLOAT_SENTINELS[value]}
    return value


def _encode_ndarray(value: np.ndarray) -> dict:
    return {TAG: "numpy.ndarray", "data": value.tolist(), "dtype": str(value.dtype)}


def _encode_slice(value: slice) -> dict:
    return {
        TAG: "builtins.slice",
        "start": value.start,
        "stop": value.stop,
        "step": value.step,
    }


def _decode_leaf(node):
    """Decode a leaf node previously produced by an _encode_* helper.

    Accepts already-native floats unchanged.
    """
    if not isinstance(node, dict):
        return node
    tag = node.get(TAG)
    if tag == "builtins.float":
        return _FLOAT_SENTINELS_INV[node["value"]]
    if tag == "numpy.ndarray":
        return np.array(node["data"], dtype=node["dtype"])
    if tag == "builtins.slice":
        return slice(node["start"], node["stop"], node["step"])
    if tag == "builtins.tuple":
        return tuple(decode(x) for x in node["items"])  # noqa: F821 - decode added in Task 1.3
    raise SerialisationError(f"Not a leaf node: {tag!r}")


_LEAF_TAGS = frozenset(
    {"builtins.float", "numpy.ndarray", "builtins.slice", "builtins.tuple"}
)
