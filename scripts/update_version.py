"""
Automatically update the version number
"""

import os
import re
from datetime import date
import pybamm


def update_version():
    """
    Opens file and updates the version number
    """
    release_version = os.getenv("VERSION")[1:]
    release_date = date.today()

    # pybamm/version.py
    with open(
        os.path.join(pybamm.root_dir(), "src", "pybamm", "version.py"), "r+"
    ) as file:
        output = file.read()
        replace_version = re.sub(
            '(?<=__version__ = ")(.+)(?=")', release_version, output
        )
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    # pyproject.toml
    with open(os.path.join(pybamm.root_dir(), "pyproject.toml"), "r+") as file:
        output = file.read()
        replace_version = re.sub(
            r'(?<=\bversion = ")(.+)(?=")', release_version, output
        )
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    # CITATION.cff
    with open(os.path.join(pybamm.root_dir(), "CITATION.cff"), "r+") as file:
        output = file.read()
        replace_version = re.sub('(?<=version: ")(.+)(?=")', release_version, output)
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    changelog_line1 = "# [Unreleased](https://github.com/pybamm-team/PyBaMM/)\n"
    changelog_line2 = f"# [v{release_version}](https://github.com/pybamm-team/PyBaMM/tree/v{release_version}) - {release_date}\n\n"

    # CHANGELOG.md
    with open(os.path.join(pybamm.root_dir(), "CHANGELOG.md"), "r+") as file:
        output_list = file.readlines()
        output_list[0] = changelog_line1
        output_list.insert(2, changelog_line2)
        file.truncate(0)
        file.seek(0)
        file.writelines(output_list)


if __name__ == "__main__":
    update_version()
