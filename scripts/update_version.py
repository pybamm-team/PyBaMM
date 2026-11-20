"""
Automatically update version numbers in various files for releases
"""

import argparse
import os
import re
import subprocess
from datetime import date
from pathlib import Path

import pybamm

# Repo-root files (e.g. CHANGELOG.md) live at the monorepo root; this script
# sits in <repo-root>/scripts/, so the repo root is its parent's parent.
# Package-level files (e.g. CITATION.cff) live with the pybamm package and are
# located via pybamm.root_dir() (which returns packages/pybamm in the monorepo).
REPO_ROOT = Path(__file__).resolve().parent.parent
TOWNCRIER = "towncrier==26.9.0"


def update_version(release_version):
    """
    Updates version numbers and release information across project files
    """
    release_date = date.today()

    # CITATION.cff (package-level file)
    with open(os.path.join(pybamm.root_dir(), "CITATION.cff"), "r+") as file:
        output = file.read()
        replace_version = re.sub('(?<=version: ")(.+)(?=")', release_version, output)
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    # CHANGELOG.md (repo-root file): towncrier builds the changelog.d/ fragments
    # into a dated block, titled as configured in pyproject.toml, and deletes them.
    subprocess.run(
        [
            "uvx",
            "--from",
            TOWNCRIER,
            "towncrier",
            "build",
            "--version",
            release_version,
            "--date",
            release_date.isoformat(),
            "--yes",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update version numbers for a PyBaMM release"
    )
    parser.add_argument(
        "version",
        help=(
            "Release version in YY.MM.N.P form, e.g. 27.1.0.0 for a feature "
            "release or 27.1.0.1 for a patch. See RELEASE.md for the version "
            "scheme."
        ),
    )
    args = parser.parse_args()
    update_version(args.version)
