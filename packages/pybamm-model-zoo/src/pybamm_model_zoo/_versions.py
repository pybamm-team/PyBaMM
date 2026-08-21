"""Ordering PyBaMM releases, and choosing which ones a model is tested against.

Shared by the compatibility matrix the weekly job runs and the status tables the
docs generator renders, so a release window is decided in one tested place
rather than once per consumer.

Neither PyBaMM nor ``packaging`` is imported at module scope: both consumers run
in environments with no PyBaMM install, and the docs generator has no
``packaging`` either, which only :func:`window_for` reaches for.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pybamm_model_zoo._registry import ModelEntry

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


def window_for(entry: ModelEntry, releases: Iterable[str], count: int) -> list[str]:
    """The ``count`` newest releases ``entry`` admits, oldest first.

    The filtering has to come before the window, not after: taking the newest
    releases globally and then dropping the ones a model excludes leaves a model
    with an upper bound no released cells at all, even though older releases it
    does support are right there.
    """
    return [release for release in releases if entry.admits(release)][-count:]
