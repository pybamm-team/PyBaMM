"""
Automatically update version numbers in various files for releases
"""

import argparse
import os
import re
from datetime import date
from pathlib import Path

import pybamm

# Repo-root files are at parent's parent; package files located via pybamm.root_dir()
REPO_ROOT = Path(__file__).resolve().parent.parent


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

    # CHANGELOG.md (repo-root file). The release tag is `pybamm-v<version>`
    # (monorepo tag namespace), so the changelog link targets that ref.
    changelog_line1 = "# [Unreleased](https://github.com/pybamm-team/PyBaMM/)\n"
    changelog_line2 = f"# [v{release_version}](https://github.com/pybamm-team/PyBaMM/tree/pybamm-v{release_version}) - {release_date}\n\n"

    with open(os.path.join(REPO_ROOT, "CHANGELOG.md"), "r+") as file:
        output_list = file.readlines()
        output_list[0] = changelog_line1
        output_list.insert(2, changelog_line2)
        file.truncate(0)
        file.seek(0)
        file.writelines(output_list)


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
