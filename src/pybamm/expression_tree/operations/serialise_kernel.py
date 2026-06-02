"""Unified safe-or-loud serialisation kernel.

One encode/decode engine for pybamm serialisable objects. Dispatches by type to
per-class ``to_json``/``_from_json`` hooks or an introspection default, and
raises :class:`SerialisationError` rather than silently dropping any value.
"""

from __future__ import annotations

import importlib

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
