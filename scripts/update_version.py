"""
Automatically update version numbers in various files for releases
"""

import os
import re
from datetime import date

import pybamm
from pybamm._version import __version__ as release_version


def update_version():
    """
    Updates version numbers and release information across project files
    """
    release_date = date.today()

    # CITATION.cff
    with open(os.path.join(pybamm.root_dir(), "CITATION.cff"), "r+") as file:
        output = file.read()
        replace_version = re.sub('(?<=version: ")(.+)(?=")', release_version, output)
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    # CHANGELOG.md
    changelog_line1 = "# [Unreleased](https://github.com/pybamm-team/PyBaMM/)\n"
    changelog_line2 = f"# [v{release_version}](https://github.com/pybamm-team/PyBaMM/tree/v{release_version}) - {release_date}\n\n"

    with open(os.path.join(pybamm.root_dir(), "CHANGELOG.md"), "r+") as file:
        output_list = file.readlines()
        output_list[0] = changelog_line1
        output_list.insert(2, changelog_line2)
        file.truncate(0)
        file.seek(0)
        file.writelines(output_list)


if __name__ == "__main__":
    update_version()
