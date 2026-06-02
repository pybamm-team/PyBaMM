"""Unified safe-or-loud serialisation kernel.

One encode/decode engine for pybamm serialisable objects. Dispatches by type to
per-class ``to_json``/``_from_json`` hooks or an introspection default, and
raises :class:`SerialisationError` rather than silently dropping any value.
"""

from __future__ import annotations

import importlib
import inspect

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
        value = node["value"]
        if value not in _FLOAT_SENTINELS_INV:
            raise SerialisationError(f"Unknown float sentinel: {value!r}")
        return _FLOAT_SENTINELS_INV[value]
    if tag == "numpy.ndarray":
        return np.array(node["data"], dtype=node["dtype"])
    if tag == "builtins.slice":
        return slice(node["start"], node["stop"], node["step"])
    if tag == "builtins.tuple":
        return tuple(decode(x) for x in node["items"])
    raise SerialisationError(f"Not a leaf node: {tag!r}")


_LEAF_TAGS = frozenset(
    {"builtins.float", "numpy.ndarray", "builtins.slice", "builtins.tuple"}
)


def _encode_class_ref(cls: type) -> dict:
    return {TAG: "type", "class": _class_path(cls)}


def encode(obj):
    """Encode any serialisable object into a JSON-compatible structure.

    Raises SerialisationError for objects with no codec (safe-or-loud).
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return _encode_float(obj)
    if isinstance(obj, np.ndarray):
        return _encode_ndarray(obj)
    if isinstance(obj, slice):
        return _encode_slice(obj)
    if isinstance(obj, tuple):
        return {TAG: "builtins.tuple", "items": [encode(x) for x in obj]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, type):
        return _encode_class_ref(obj)

    codec = _lookup_codec(type(obj))
    if codec is None:
        raise SerialisationError(
            f"Cannot serialise {type(obj).__module__}.{type(obj).__qualname__}: "
            f"no codec registered. Register the class or add a to_json/_from_json hook."
        )
    node = codec.to_json(obj, encode)
    node[TAG] = _class_path(type(obj))
    return node


def decode(node):
    """Inverse of :func:`encode`."""
    if node is None or isinstance(node, (bool, int, str, float)):
        return node
    if isinstance(node, list):
        return [decode(x) for x in node]
    if isinstance(node, dict):
        node = normalise_legacy(node)
        tag = node.get(TAG)
        if tag in _LEAF_TAGS:
            return _decode_leaf(node)
        if tag == "type":
            return _resolve_class(node["class"])
        if tag is None:
            return {k: decode(v) for k, v in node.items()}
        cls = _resolve_class(tag)
        codec = _lookup_codec(cls)
        if codec is None:
            raise SerialisationError(f"No codec to decode '{tag}'")
        return codec.from_json(node, decode, cls)
    raise SerialisationError(f"Cannot decode value of type {type(node)}")


def normalise_legacy(node: dict) -> dict:
    return node  # replaced in Task 1.6


# ---------------------------------------------------------------------------
# DefaultCodec
# ---------------------------------------------------------------------------

_CHILD_PARAMS = frozenset({"child", "children", "child_input", "left", "right"})
_SKIP_PARAMS = frozenset({"self", "domain", "domains", "auxiliary_domains"})
_MISSING = object()


def _read_attr(obj, name):
    for candidate in (name, f"_{name}"):
        if hasattr(obj, candidate):
            return getattr(obj, candidate)
    return _MISSING


class DefaultCodec:
    """Introspection-based codec: derive fields from ``__init__``.

    Captures every resolvable non-child parameter (including defaulted ones);
    raises if a *required* parameter cannot be read from the instance.
    """

    def to_json(self, obj, encode) -> dict:
        node: dict = {}
        node["domains"] = obj.domains
        sig = inspect.signature(type(obj).__init__)
        # Validate and collect non-child params first so that missing-required-param
        # errors surface before any encoding attempt (fail-fast, predictable order).
        extra: dict = {}
        for name, param in sig.parameters.items():
            if name in _SKIP_PARAMS or name in _CHILD_PARAMS:
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise SerialisationError(
                    f"{type(obj).__name__}.__init__ uses *args/**kwargs ({name}); "
                    f"add a to_json/_from_json hook."
                )
            value = _read_attr(obj, name)
            if value is _MISSING:
                if param.default is inspect.Parameter.empty:
                    raise SerialisationError(
                        f"{type(obj).__name__}.__init__ requires {name!r}; no "
                        f"attribute {name}/_{name} found -- add a to_json/_from_json "
                        f"hook or register a codec."
                    )
                continue
            extra[name] = value
        # Encode children and collected params only after validation succeeds.
        if obj.children:
            node["children"] = [encode(c) for c in obj.children]
        for name, value in extra.items():
            node[name] = encode(value)
        return node

    def from_json(self, node, decode, cls):
        children = [decode(c) for c in node.get("children", [])]
        param_names = set(inspect.signature(cls.__init__).parameters)
        # Forward only real __init__ params; ignore bookkeeping/legacy extras
        # (e.g. the old switch's "name") rather than passing them as kwargs.
        # "domains" is excluded here and handled separately below.
        kwargs = {
            key: decode(raw)
            for key, raw in node.items()
            if key not in (TAG, "children", "domains") and key in param_names
        }
        # to_json always writes node["domains"]; map it onto whichever param the
        # signature accepts -- `domains` (full dict), `domain` (primary list), or
        # neither (class infers domain from children). Prevents the silent
        # domain-drop for `domain`-param classes like CoupledVariable.
        domains = node.get("domains")
        if domains:
            if "domains" in param_names:
                kwargs.setdefault("domains", domains)
            elif "domain" in param_names:
                kwargs.setdefault("domain", domains.get("primary") or None)
        return cls(*children, **kwargs)


_default_codec = DefaultCodec()


def _lookup_codec(cls: type):
    import pybamm  # local import to avoid circular dependency at module level

    if issubclass(cls, pybamm.Symbol):
        return _default_codec
    return None  # extended in Task 1.5 with hook detection
