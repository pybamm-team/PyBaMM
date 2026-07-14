from __future__ import annotations

import inspect

import numpy as np
import pytest

import pybamm

_MISSING = object()

# Per-step attributes that must round-trip. ``_experiments_equal`` skips
# attrs missing on both sides, so ``value`` is fine for the Rest case.
_STEP_ATTRS = (
    "duration",
    "input_duration",
    "uses_default_duration",
    "value",
    "temperature",
    "termination",
    "period",
    "tags",
    "description",
    "direction",
    "start_time",
    "skip_ok",
)
_EXPERIMENT_TOP_LEVEL_ATTRS = ("period", "temperature", "termination_string")


def _attr(obj, name):
    """Read constructor-arg-named attribute, trying both ``name`` and ``_name``."""
    for candidate in (name, f"_{name}"):
        if hasattr(obj, candidate):
            return getattr(obj, candidate)
    return _MISSING


def _values_equal(a, b):
    """Strict equality with bool-vs-int discrimination and float tolerance.

    Recurses into dicts, lists/tuples, and ``BaseSolver`` instances so a
    nested ``root_method`` is compared structurally.
    """
    if isinstance(a, pybamm.BaseSolver) or isinstance(b, pybamm.BaseSolver):
        return _solver_init_args_equal(a, b)
    # bool must match bool exactly so True doesn't round-trip as 1 (#5495).
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return a == pytest.approx(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_values_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, type(a))
            and len(a) == len(b)
            and all(_values_equal(x, y) for x, y in zip(a, b, strict=False))
        )
    return a == b


def _solver_init_args_equal(a, b):
    """Two solvers are equal if same class and every ``__init__`` arg matches."""
    if type(a) is not type(b):
        return False
    sig = inspect.signature(type(a).__init__)
    for name in sig.parameters:
        if name == "self":
            continue
        va = _attr(a, name)
        vb = _attr(b, name)
        if va is _MISSING and vb is _MISSING:
            continue
        if not _values_equal(va, vb):
            return False
    return True


def _experiments_equal(a, b):
    """Equality across both top-level experiment attrs and per-step attrs."""
    for attr in _EXPERIMENT_TOP_LEVEL_ATTRS:
        if not _values_equal(getattr(a, attr, None), getattr(b, attr, None)):
            return False
    if len(a.steps) != len(b.steps):
        return False
    for orig, new in zip(a.steps, b.steps, strict=True):
        if type(orig) is not type(new):
            return False
        for attr in _STEP_ATTRS:
            if not hasattr(orig, attr) and not hasattr(new, attr):
                continue
            if not _values_equal(getattr(orig, attr, None), getattr(new, attr, None)):
                return False
    return True
