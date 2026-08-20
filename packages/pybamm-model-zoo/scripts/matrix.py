"""Emit the model zoo compatibility matrix for the weekly status workflow.

Prints a GitHub Actions ``include`` list: one ``{model, version}`` cell per pair.
A pair a manifest's ``pybamm_requires`` excludes is left out, so a badge never
reports "failing" on a release the model never claimed to support.

    uv run --with packaging python packages/pybamm-model-zoo/scripts/matrix.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pybamm_model_zoo._registry import Registry

PYPI_URL = "https://pypi.org/pypi/pybamm/json"
#: Final CalVer releases only: no prereleases, no yanked-empty entries.
CALVER = re.compile(r"^\d+(\.\d+)*$")
#: The checkout itself, which has no release number to match a specifier against.
MAIN = "main"


def version_order(version: str) -> tuple[int, list[int]]:
    """Sort releases numerically, and sort ``main`` last."""
    try:
        return (0, [int(part) for part in version.split(".")])
    except ValueError:
        return (1, [])


def released_versions(count: int) -> list[str]:
    """The ``count`` most recent final PyBaMM releases on PyPI, oldest first."""
    with urllib.request.urlopen(PYPI_URL) as response:  # nosec B310 - literal https URL
        releases = json.load(response)["releases"]
    published = sorted(
        (
            version
            for version, files in releases.items()
            if files and CALVER.match(version)
        ),
        key=version_order,
    )
    return published[-count:]


def matrix(versions: list[str]) -> list[dict[str, str]]:
    """One cell per (model, version) pair the model's declared range admits."""
    cells = []
    for entry in sorted(Registry().values(), key=lambda entry: entry.slug):
        for version in versions:
            if version == MAIN or entry.admits(version):
                cells.append({"model": entry.slug, "version": version})
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--releases",
        type=int,
        default=2,
        help="how many of the most recent PyPI releases to test against",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="print `include=` and `versions=` lines for $GITHUB_OUTPUT",
    )
    args = parser.parse_args(argv)

    cells = matrix([*released_versions(args.releases), MAIN])
    if args.github_output:
        # The workflow matrixes on version and loops models inside the cell, so
        # it needs the versions that survived filtering as well as the pairs.
        versions = sorted({cell["version"] for cell in cells}, key=version_order)
        print(f"include={json.dumps(cells)}")
        print(f"versions={json.dumps(versions)}")
    else:
        print(json.dumps(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
