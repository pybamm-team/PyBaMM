"""Checking a model's dependencies against the installed distribution metadata.

A model declares its dependencies once, as an extra of the distribution that
ships it, so the metadata read here is exactly what ``pip`` installs for that
extra. ``packaging`` is imported where it is used, because the docs generator
imports the registry without it.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, requires, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.requirements import Requirement


def unsatisfied(distribution: str, extras: Iterable[str]) -> list[str]:
    """Requirements of ``distribution[extras]`` that are not installed.

    Nested extras are followed, so ``pybamm[fem]`` reports scikit-fem when PyBaMM
    is installed without it. A distribution that is not installed reports nothing.

    Parameters
    ----------
    distribution : str
        The distribution declaring the extras, e.g. ``"pybamm-model-zoo"``.
    extras : iterable of str
        The extras to check.

    Returns
    -------
    list of str
        Each unsatisfied requirement, without its marker.
    """
    from packaging.utils import canonicalize_name

    extras = frozenset(extras)
    return _unsatisfied(
        distribution, extras, {(canonicalize_name(distribution), extras)}
    )


def _unsatisfied(
    distribution: str, extras: frozenset[str], seen: set[tuple[str, frozenset[str]]]
) -> list[str]:
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    missing = []
    for requirement in _extra_requirements(distribution, extras):
        requirement.marker = None
        try:
            installed = Version(version(requirement.name))
        except PackageNotFoundError:
            missing.append(str(requirement))
            continue
        if not requirement.specifier.contains(installed, prereleases=True):
            missing.append(str(requirement))
            continue
        key = (canonicalize_name(requirement.name), frozenset(requirement.extras))
        if requirement.extras and key not in seen:
            seen.add(key)
            missing.extend(_unsatisfied(requirement.name, key[1], seen))
    return missing


def _extra_requirements(distribution: str, extras: frozenset[str]) -> list[Requirement]:
    """The requirements ``extras`` add to ``distribution`` on this platform."""
    from packaging.requirements import Requirement

    try:
        declared = requires(distribution) or []
    except PackageNotFoundError:
        return []
    added = []
    for item in declared:
        requirement = Requirement(item)
        marker = requirement.marker
        # A marker that holds with no extra selected is a base dependency, which
        # installing the distribution already satisfied.
        if marker is None or marker.evaluate({"extra": ""}):
            continue
        if any(marker.evaluate({"extra": extra}) for extra in extras):
            added.append(requirement)
    return added
