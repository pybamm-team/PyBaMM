"""Compatibility support for deprecated in-place symbol mutation."""

from __future__ import annotations

import gc
import operator
import os
import sys
import warnings
from typing import Any

from pybamm.expression_tree.tree_util import _TRANSIENT_SLOTS


class SymbolMutationDeprecationWarning(DeprecationWarning):
    """An in-place symbol update is deprecated."""


# set by the test suites; fixed at import so every process agrees
MUTATION_FORBIDDEN = os.environ.get("PYBAMM_TEST_FORBID_SYMBOL_MUTATION") == "1"
_initial_frozen_count = gc.get_freeze_count()

# Display-only metadata, not part of identity or evaluation
_DISPLAY_ATTRIBUTES = frozenset(
    {"_print_name", "_raw_print_name", "domain", "print_name"}
)


def _warn_mutation(operation: str, replacement: str) -> None:
    message = (
        f"In-place symbol mutation ({operation}) is deprecated. {replacement} "
        "Use pybamm.replace(expression, {old_symbol: new_symbol}) to update an "
        "expression."
    )
    if MUTATION_FORBIDDEN:
        raise AttributeError(message)
    if gc.get_freeze_count() > _initial_frozen_count:
        raise RuntimeError(
            "Cannot safely mutate symbols while gc.freeze() is active. "
            "Use an out-of-place replacement or call gc.unfreeze() first."
        )
    warnings.warn(message, SymbolMutationDeprecationWarning, stacklevel=3)
    from pybamm.expression_tree.symbol import Symbol

    for obj in gc.get_objects():
        if isinstance(obj, Symbol):
            for slot in _TRANSIENT_SLOTS:
                object.__setattr__(obj, slot, None)


def legacy_property(slot: str, replacement: str, convert=None, doc=None):
    """A read-only property over ``slot`` whose setter is a deprecated mutation."""

    def fset(self, value):
        _warn_mutation(slot.lstrip("_"), replacement)
        object.__setattr__(self, slot, value if convert is None else convert(value))

    return property(operator.attrgetter(slot), fset, doc=doc)


def _process_bounds(values):
    from pybamm.expression_tree.variable import _process_bounds

    return _process_bounds(values)


bounds_property = legacy_property(
    "_bounds",
    "Use symbol = symbol.create_copy(bounds=value).",
    convert=_process_bounds,
    doc="Physical bounds on the variable.",
)


def _intern_domains(domains):
    from pybamm.expression_tree.symbol import _intern_domains

    return _intern_domains(domains)


class _DomainList(list):
    """A list view supporting deprecated edits to its owning symbol."""

    __slots__ = ("_level", "_owner", "_source")

    def __init__(self, owner: Any, level: str):
        self._owner, self._level = owner, level
        self._source = self._read_values()
        super().__init__(self._source)

    def _refresh(self) -> None:
        values = self._read_values()
        if values is not self._source:
            list.__setitem__(self, slice(None), values)
            self._source = values

    def _read_values(self):
        if self._owner is not None and self._level not in self._owner._domains:
            self._owner = None
        return (
            self._owner._domains[self._level]
            if self._owner is not None
            else list.copy(self)
        )

    def _write_values(self, values) -> None:
        if self._owner is None:
            return
        domains = dict(self._owner._domains)
        domains[self._level] = values
        object.__setattr__(self._owner, "_domains", _intern_domains(domains))

    def __reduce__(self):
        return (list, (list(self),))


def _list_mutator(name):
    def mutate(self, *args, **kwargs):
        replacement = (
            "Build a new symbol with the desired children or inputs."
            if isinstance(self, _SymbolList)
            else "Build a new symbol with with_domains()."
        )
        _warn_mutation(f"{self._level}.{name}", replacement)
        values = list(self)
        result = getattr(values, name)(*args, **kwargs)
        self._write_values(values)
        list.__setitem__(self, slice(None), values)
        return self if name in ("__iadd__", "__imul__") else result

    return mutate


for _method in (
    "__setitem__",
    "__delitem__",
    "__iadd__",
    "__imul__",
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "clear",
    "sort",
    "reverse",
):
    setattr(_DomainList, _method, _list_mutator(_method))


class _SymbolList(_DomainList):
    """A list view for legacy edits of symbol children and input names."""

    __slots__ = ()

    def _read_values(self):
        return getattr(self._owner, self._level)

    def _write_values(self, values) -> None:
        object.__setattr__(self._owner, self._level, values)


class _DomainsDict(dict):
    """A dictionary view supporting deprecated edits to its owning symbol."""

    __slots__ = ("_owner", "_source")

    def __init__(self, owner: Any):
        super().__init__((level, _DomainList(owner, level)) for level in owner._domains)
        self._owner = owner
        self._source = owner._domains

    def _refresh(self) -> None:
        if self._owner._domains is not self._source:
            dict.clear(self)
            dict.update(
                self,
                (
                    (level, _DomainList(self._owner, level))
                    for level in self._owner._domains
                ),
            )
            self._source = self._owner._domains

    def __reduce__(self):
        return (dict, ({level: list(names) for level, names in self.items()},))


def _dict_mutator(name):
    def mutate(self, *args, **kwargs):
        self._refresh()
        domains = {level: list(names) for level, names in dict.items(self)}
        _warn_mutation(f"domains.{name}", "Build a new symbol with with_domains().")
        result = getattr(domains, name)(*args, **kwargs)
        object.__setattr__(self._owner, "_domains", _intern_domains(domains))
        dict.clear(self)
        dict.update(
            self, ((level, _DomainList(self._owner, level)) for level in domains)
        )
        self._source = self._owner._domains
        if name == "setdefault":
            return self[args[0]]
        return self if name == "__ior__" else result

    return mutate


for _method in (
    "__setitem__",
    "__delitem__",
    "clear",
    "pop",
    "popitem",
    "setdefault",
    "update",
    "__ior__",
):
    setattr(_DomainsDict, _method, _dict_mutator(_method))


def _domain_reader(method):
    def read(self, *args, **kwargs):
        self._refresh()
        for arg in args:
            if isinstance(arg, _DomainList | _DomainsDict):
                arg._refresh()
        return method(self, *args, **kwargs)

    return read


for _method in (
    "__getitem__",
    "__iter__",
    "__len__",
    "__contains__",
    "__repr__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__add__",
    "__mul__",
    "__rmul__",
    "__reversed__",
    "count",
    "index",
    "copy",
):
    setattr(_DomainList, _method, _domain_reader(getattr(list, _method)))

for _method in (
    "__getitem__",
    "__iter__",
    "__len__",
    "__contains__",
    "__repr__",
    "__eq__",
    "__ne__",
    "__or__",
    "__ror__",
    "__reversed__",
    "get",
    "items",
    "keys",
    "values",
    "copy",
):
    setattr(_DomainsDict, _method, _domain_reader(getattr(dict, _method)))


def _in_own_constructor(symbol) -> bool:
    """Whether an ``__init__`` of ``symbol`` is on the calling thread's stack."""
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name == "__init__" and frame.f_locals.get("self") is symbol:
            return True
        frame = frame.f_back
    return False


def _frozen_setattr(self, name, value):
    """``Symbol.__setattr__`` while mutation is forbidden."""
    if (
        name not in _TRANSIENT_SLOTS
        and name not in _DISPLAY_ATTRIBUTES
        and hasattr(self, "_id")  # set once Symbol.__init__ has run
        and not _in_own_constructor(self)
    ):
        raise AttributeError(
            f"Cannot set '{name}' on {type(self).__name__} '{self.name}': symbols "
            "are immutable once constructed. Build a new symbol instead (e.g. "
            "`create_copy`, `with_domains`)."
        )
    object.__setattr__(self, name, value)
