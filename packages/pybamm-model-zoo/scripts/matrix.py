"""Emit the model zoo compatibility matrix for the weekly status workflow.

Prints a GitHub Actions ``include`` list: one ``{model, version}`` cell per pair.
Every model is paired with the oldest release the zoo's ``pybamm`` dependency
admits, the newest releases, and ``main``.

    uv run --with packaging python packages/pybamm-model-zoo/scripts/matrix.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pybamm_model_zoo._registry import Registry
from pybamm_model_zoo._versions import MAIN, pybamm_specifier, sorted_releases, window

PYPI_URL = "https://pypi.org/pypi/pybamm/json"


def released_versions() -> list[str]:
    """Every final PyBaMM release on PyPI, oldest first."""
    with urllib.request.urlopen(PYPI_URL) as response:  # nosec B310 - literal https URL
        releases = json.load(response)["releases"]
    return sorted_releases(version for version, files in releases.items() if files)


def matrix(releases: list[str], count: int) -> list[dict[str, str]]:
    """One cell per model per release in the zoo's window, plus ``main``."""
    versions = [*window(releases, pybamm_specifier(), count), MAIN]
    return [
        {"model": entry.slug, "version": version}
        for entry in sorted(Registry().values(), key=lambda entry: entry.slug)
        for version in versions
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--releases",
        type=int,
        default=2,
        help="how many of the newest releases to test, besides the oldest supported",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="print `include=` and `versions=` lines for $GITHUB_OUTPUT",
    )
    args = parser.parse_args(argv)

    cells = matrix(released_versions(), args.releases)
    if args.github_output:
        # The workflow matrixes on version and loops models inside the cell, so
        # it needs the versions that survived filtering as well as the pairs.
        versions = [*sorted_releases({cell["version"] for cell in cells}), MAIN]
        print(f"include={json.dumps(cells)}")
        print(f"versions={json.dumps(versions)}")
    else:
        print(json.dumps(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
