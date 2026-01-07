from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

import pybamm

if TYPE_CHECKING:
    from collections.abc import Mapping


class ParameterCategory(Enum):
    """Categories for grouping battery parameters."""

    NEGATIVE_ELECTRODE = "negative electrode"
    POSITIVE_ELECTRODE = "positive electrode"
    SEPARATOR = "separator"
    ELECTROLYTE = "electrolyte"
    THERMAL = "thermal"
    KINETIC = "kinetic"
    GEOMETRIC = "geometric"
    ELECTRICAL = "electrical"
    OTHER = "other"


@dataclass
class ParameterInfo:
    """
    Metadata about a parameter.

    Attributes
    ----------
    name : str
        The parameter name (key).
    value : Any
        The parameter value.
    units : str | None
        Units parsed from the parameter name (e.g., "K" from "Temperature [K]").
    category : str | None
        Category of the parameter (e.g., "negative electrode", "thermal").
    is_function : bool
        True if the value is callable.
    is_input : bool
        True if the value is an InputParameter.
    """

    name: str
    value: Any
    units: str | None
    category: str | None
    is_function: bool
    is_input: bool


@dataclass
class ParameterDiff:
    """
    Result of comparing two parameter sets.

    Attributes
    ----------
    added : dict[str, Any]
        Parameters in `other` but not in `self`.
    removed : dict[str, Any]
        Parameters in `self` but not in `other`.
    changed : dict[str, tuple[Any, Any]]
        Parameters with different values: (self_value, other_value).
    """

    added: dict[str, Any]
    removed: dict[str, Any]
    changed: dict[str, tuple[Any, Any]]


# Regex to parse units from parameter names like "Temperature [K]"
_UNITS_RE = re.compile(r"\[([^\]]+)\]\s*$")

# Keywords for category detection
_CATEGORY_KEYWORDS: dict[ParameterCategory, list[str]] = {
    ParameterCategory.NEGATIVE_ELECTRODE: ["negative electrode", "negative particle"],
    ParameterCategory.POSITIVE_ELECTRODE: ["positive electrode", "positive particle"],
    ParameterCategory.SEPARATOR: ["separator"],
    ParameterCategory.ELECTROLYTE: ["electrolyte"],
    ParameterCategory.THERMAL: [
        "thermal",
        "temperature",
        "heat",
        "cooling",
        "conductivity",
    ],
    ParameterCategory.KINETIC: [
        "exchange-current",
        "reaction",
        "kinetic",
        "transfer coefficient",
    ],
    ParameterCategory.GEOMETRIC: [
        "thickness",
        "length",
        "width",
        "height",
        "radius",
        "area",
        "volume",
        "porosity",
    ],
    ParameterCategory.ELECTRICAL: [
        "conductivity",
        "current",
        "voltage",
        "capacity",
        "resistance",
    ],
}


def _parse_units(name: str) -> str | None:
    """Extract units from a parameter name like 'Temperature [K]' -> 'K'."""
    match = _UNITS_RE.search(name)
    return match.group(1) if match else None


def _detect_category(name: str) -> str | None:
    """Detect the category of a parameter based on keywords in its name."""
    name_lower = name.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category.value
    return ParameterCategory.OTHER.value


class ParameterStore:
    """
    Manages parameter key-value storage with FuzzyDict lookup.

    This class provides a clean interface for storing and retrieving parameters
    with explicit control over update behavior.

    Parameters
    ----------
    initial_data : dict | None
        Initial parameter data. If None, starts empty.

    Examples
    --------
    >>> store = ParameterStore({"Temperature [K]": 298.15})
    >>> store["Temperature [K]"]
    298.15
    >>> store.set("New param", 42, allow_new=True)
    >>> store.get_info("Temperature [K]")
    ParameterInfo(name='Temperature [K]', value=298.15, units='K', ...)
    """

    def __init__(self, initial_data: dict[str, Any] | None = None) -> None:
        self._data: pybamm.FuzzyDict = pybamm.FuzzyDict(initial_data or {})

    def __getitem__(self, key: str) -> Any:
        """Get a parameter value by key."""
        try:
            return self._data[key]
        except KeyError as e:
            # Re-raise with more context
            raise KeyError(
                f"Parameter '{key}' not found. {e.args[0] if e.args else ''}"
            ) from e

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a parameter value (always allows new parameters)."""
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete a parameter."""
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check if a parameter exists."""
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value, returning default if not found.

        Example
        -------
        >>> store = ParameterStore({"a": 1})
        >>> store.get("a")
        1
        >>> store.get("missing", default=0)
        0
        """
        try:
            return self._data[key]
        except KeyError:
            return default

    def set(self, key: str, value: Any, *, allow_new: bool = True) -> None:
        """
        Set a parameter value.

        Parameters
        ----------
        key : str
            Parameter name.
        value : Any
            Parameter value.
        allow_new : bool
            If False, raises KeyError when key doesn't exist.
            If True (default), allows adding new parameters.

        Raises
        ------
        KeyError
            If allow_new=False and the key doesn't exist.

        Example
        -------
        >>> store = ParameterStore({"a": 1})
        >>> store.set("a", 10)  # Update existing
        >>> store.set("b", 2)   # Add new (allow_new=True by default)
        """
        if not allow_new and key not in self._data:
            best_matches = self._data.get_best_matches(key)
            raise KeyError(
                f"Parameter '{key}' does not exist. "
                f"Use allow_new=True to add new parameters. "
                f"Best matches: {best_matches}"
            )
        self._data[key] = value

    def update(
        self,
        values: Mapping[str, Any],
        *,
        allow_new: bool = True,
        conflict: Literal["raise", "warn", "ignore"] = "ignore",
    ) -> None:
        """
        Bulk update parameters.

        Parameters
        ----------
        values : Mapping[str, Any]
            Dictionary of parameter values to update.
        allow_new : bool
            If False, raises KeyError for unknown parameters.
            If True (default), allows adding new parameters.
        conflict : {"raise", "warn", "ignore"}
            How to handle conflicts when a parameter already exists with a
            different value:
            - "raise": Raise ValueError
            - "warn": Emit a warning and update
            - "ignore": Silently update (default)

        Example
        -------
        >>> store = ParameterStore({"a": 1, "b": 2})
        >>> store.update({"a": 10, "c": 3})
        >>> store["a"], store["c"]
        (10, 3)
        """
        for key, value in values.items():
            # Check if key exists
            if not allow_new and key not in self._data:
                best_matches = self._data.get_best_matches(key)
                raise KeyError(
                    f"Parameter '{key}' does not exist. "
                    f"Use allow_new=True to add new parameters. "
                    f"Best matches: {best_matches}"
                )

            # Check for conflicts
            if conflict != "ignore" and key in self._data:
                existing = self._data[key]
                if existing != value:
                    msg = (
                        f"Parameter '{key}' already exists with value "
                        f"'{existing}', updating to '{value}'"
                    )
                    if conflict == "raise":
                        raise ValueError(msg)
                    elif conflict == "warn":
                        warn(msg, stacklevel=2)

            self._data[key] = value

    def keys(self):
        """Return parameter keys."""
        return self._data.keys()

    def values(self):
        """Return parameter values."""
        return self._data.values()

    def items(self):
        """Return parameter items."""
        return self._data.items()

    def pop(self, key: str, *args) -> Any:
        """
        Remove and return a parameter value.

        Example
        -------
        >>> store = ParameterStore({"a": 1, "b": 2})
        >>> store.pop("a")
        1
        >>> "a" in store
        False
        """
        return self._data.pop(key, *args)

    def copy(self) -> ParameterStore:
        """
        Return a shallow copy of the store.

        Example
        -------
        >>> store = ParameterStore({"a": 1})
        >>> store_copy = store.copy()
        >>> store_copy["a"] = 99
        >>> store["a"]  # Original unchanged
        1
        """
        return ParameterStore(dict(self._data))

    def search(self, key: str, print_values: bool = True) -> None:
        """
        Search for parameters containing the given key.

        Example
        -------
        >>> store = ParameterStore({"Temperature [K]": 298.15})
        >>> store.search("Temperature")  # Prints matching parameters
        """
        return self._data.search(key, print_values)

    def get_info(self, key: str) -> ParameterInfo:
        """
        Get metadata about a parameter.

        Parameters
        ----------
        key : str
            The parameter name.

        Returns
        -------
        ParameterInfo
            Metadata including value, units, category, and type information.

        Examples
        --------
        >>> store = ParameterStore({"Maximum concentration [mol.m-3]": 51765})
        >>> info = store.get_info("Maximum concentration [mol.m-3]")
        >>> info.units
        'mol.m-3'
        >>> info.is_function
        False
        """
        value = self[key]
        return ParameterInfo(
            name=key,
            value=value,
            units=_parse_units(key),
            category=_detect_category(key),
            is_function=callable(value),
            is_input=isinstance(value, pybamm.InputParameter),
        )

    def list_by_category(self, category: ParameterCategory | str) -> list[str]:
        """
        Return all parameter names in a given category.

        Parameters
        ----------
        category : ParameterCategory or str
            The category to filter by. Can be a ParameterCategory enum value
            or a string like "negative electrode".

        Returns
        -------
        list[str]
            List of parameter names in the category.

        Example
        -------
        >>> store = ParameterStore({"Negative electrode thickness [m]": 1e-4})
        >>> store.list_by_category("negative electrode")
        ['Negative electrode thickness [m]']
        """
        if isinstance(category, ParameterCategory):
            category_str = category.value
        else:
            category_str = category.lower()

        return [key for key in self._data if _detect_category(key) == category_str]

    def categories(self) -> dict[str, list[str]]:
        """
        Return all parameters grouped by category.

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping category names to lists of parameter names.

        Example
        -------
        >>> store = ParameterStore({"Temperature [K]": 298.15})
        >>> cats = store.categories()
        >>> "thermal" in cats
        True
        """
        result: dict[str, list[str]] = {}
        for key in self._data:
            cat = _detect_category(key) or ParameterCategory.OTHER.value
            if cat not in result:
                result[cat] = []
            result[cat].append(key)
        return result

    def diff(self, other: ParameterStore, *, rtol: float = 0.0) -> ParameterDiff:
        """
        Compare this store with another and return differences.

        Parameters
        ----------
        other : ParameterStore
            The other parameter store to compare against.
        rtol : float, optional
            Relative tolerance for numerical comparisons. Differences smaller
            than ``rtol * max(|a|, |b|)`` are ignored. Default is 0.0 (exact
            comparison). Set to e.g. 1e-6 to ignore tiny floating-point
            differences.

        Returns
        -------
        ParameterDiff
            Object containing added, removed, and changed parameters.

        Examples
        --------
        >>> store1 = ParameterStore({"a": 1, "b": 2})
        >>> store2 = ParameterStore({"b": 3, "c": 4})
        >>> diff = store1.diff(store2)
        >>> diff.added
        {'c': 4}
        >>> diff.removed
        {'a': 1}
        >>> diff.changed
        {'b': (2, 3)}

        With tolerance to ignore small differences:

        >>> store1 = ParameterStore({"x": 1.0})
        >>> store2 = ParameterStore({"x": 1.0 + 1e-10})
        >>> diff = store1.diff(store2, rtol=1e-9)
        >>> diff.changed  # Empty because difference is within tolerance
        {}
        """
        self_keys = set(self._data.keys())
        other_keys = set(other._data.keys())

        added = {k: other._data[k] for k in other_keys - self_keys}
        removed = {k: self._data[k] for k in self_keys - other_keys}

        changed = {}
        for key in self_keys & other_keys:
            self_val = self._data[key]
            other_val = other._data[key]
            if not _values_equal(self_val, other_val, rtol=rtol):
                changed[key] = (self_val, other_val)

        return ParameterDiff(added=added, removed=removed, changed=changed)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a plain dictionary copy of the parameters.

        Example
        -------
        >>> store = ParameterStore({"a": 1, "b": 2})
        >>> store.to_dict()
        {'a': 1, 'b': 2}
        """
        return dict(self._data)


def _values_equal(a: Any, b: Any, *, rtol: float = 0.0) -> bool:
    """
    Compare two values for equality, handling special cases.

    Parameters
    ----------
    a, b : Any
        Values to compare.
    rtol : float
        Relative tolerance for numerical comparisons.
    """
    import numpy as np

    # Handle numpy arrays
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            if rtol > 0:
                return np.allclose(a, b, rtol=rtol, atol=0)
            return np.array_equal(a, b)
        except (TypeError, ValueError):
            return False

    # Handle numeric types with tolerance
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if rtol > 0:
            # Check if difference is within relative tolerance
            max_val = max(abs(a), abs(b))
            if max_val == 0:
                return a == b
            return abs(a - b) <= rtol * max_val
        return a == b

    # Handle callables - compare by identity
    if callable(a) and callable(b):
        return a is b

    # Handle pybamm symbols
    if isinstance(a, pybamm.Symbol) and isinstance(b, pybamm.Symbol):
        return a == b

    # Default comparison
    try:
        return a == b
    except (TypeError, ValueError):
        return False
