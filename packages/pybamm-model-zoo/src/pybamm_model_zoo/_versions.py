"""Ordering PyBaMM releases, and choosing which ones the zoo is tested against.

Shared by the compatibility matrix the weekly job runs and the status tables the
docs generator renders, so a release window is decided in one tested place
rather than once per consumer.

Neither PyBaMM nor ``packaging`` is imported at module scope: both consumers run
in environments with no PyBaMM install, and the docs generator has no
``packaging`` either, which only :func:`pybamm_specifier` and :func:`window`
reach for.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from pybamm_model_zoo._exceptions import ZooError
from pybamm_model_zoo._paths import ZOO_PYPROJECT
from pybamm_model_zoo._registry import read_manifest

#: Final CalVer releases only: no prereleases, no yanked-empty entries.
CALVER = re.compile(r"^\d+(\.\d+)*$")
#: The checkout itself, which has no release number to match a specifier against.
MAIN = "main"


def version_key(version: str) -> tuple[int, list[int]]:
    """Sort CalVer releases numerically, and sort anything else (``main``) last."""
    try:
        return (0, [int(part) for part in version.split(".")])
    except ValueError:
        return (1, [])


def sorted_releases(versions: Iterable[str]) -> list[str]:
    """Every final CalVer release among ``versions``, oldest first."""
    return sorted((v for v in versions if CALVER.match(v)), key=version_key)


def pybamm_specifier(pyproject: Path = ZOO_PYPROJECT) -> str:
    """The PyBaMM versions the zoo's ``pybamm`` dependency admits, e.g. ``>=26.10``.

    Raises
    ------
    ZooError
        If ``pyproject`` declares no ``pybamm`` dependency.
    """
    from packaging.requirements import Requirement

    for item in read_manifest(pyproject).get("project", {}).get("dependencies", []):
        requirement = Requirement(item)
        if requirement.name == "pybamm":
            return str(requirement.specifier)
    raise ZooError(f"{pyproject}: no pybamm dependency")


def window(releases: Iterable[str], specifier: str, count: int) -> list[str]:
    """The oldest release ``specifier`` admits and its ``count`` newest, oldest first.

    The oldest is kept whatever ``count`` is: it is the floor every model is
    installable against, so it is the release most likely to break unnoticed.

    Parameters
    ----------
    releases : iterable of str
        Final releases, oldest first.
    specifier : str
        The PyBaMM versions to admit.
    count : int
        How many of the newest admitted releases to keep besides the oldest.
    """
    from packaging.specifiers import SpecifierSet

    admitted = list(SpecifierSet(specifier).filter(releases))
    if not admitted:
        return []
    # `[-0:]` is the whole list, so an empty window has to be spelled out.
    newest = admitted[-count:] if count else []
    return sorted({admitted[0], *newest}, key=version_key)
