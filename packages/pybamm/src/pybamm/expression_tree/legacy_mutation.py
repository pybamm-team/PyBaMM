"""Compatibility support for deprecated in-place symbol mutation."""

from __future__ import annotations

import gc
import operator
import os
import warnings

from pybamm.expression_tree.tree_util import _TRANSIENT_SLOTS


class SymbolMutationDeprecationWarning(DeprecationWarning):
    """An in-place symbol update is deprecated."""


# set by the test suites; fixed at import so every process agrees
MUTATION_FORBIDDEN = os.environ.get("PYBAMM_TEST_FORBID_SYMBOL_MUTATION") == "1"
_initial_frozen_count = gc.get_freeze_count()


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


# attributes stored under a different slot name before symbols were slotted
_RENAMED_PICKLED_FIELDS = {
    "mesh": "_mesh",
    "secondary_mesh": "_secondary_mesh",
    "tertiary_mesh": "_tertiary_mesh",
    "bounds": "_bounds",
}


def upgrade_pickled_state(cls: type, state: dict) -> dict:
    """Convert the ``__dict__`` of a symbol pickled before symbols were slotted
    into the current slot state of ``cls``."""
    from pybamm.expression_tree.symbol import Symbol
    from pybamm.expression_tree.tree_util import layout

    leaf_fields, state_fields, _ = layout(cls)
    slots = {*leaf_fields, *state_fields}
    state = {
        _RENAMED_PICKLED_FIELDS.get(key, key): value for key, value in state.items()
    }
    upgraded = dict.fromkeys(slots)
    upgraded.update((key, value) for key, value in state.items() if key in slots)
    upgraded["_domains"] = Symbol._normalise_domains(state["_domains"])
    if upgraded.get("_raw_print_name") is None:
        upgraded["_raw_print_name"] = state.get("_print_name")
    return upgraded
