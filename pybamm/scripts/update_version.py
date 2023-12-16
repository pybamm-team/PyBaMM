"""
Automatically update the version number
"""

import json
import os
import re
from datetime import date
from dateutil.relativedelta import relativedelta


import pybamm


def update_version():
    """
    Opens file and updates the version number
    """
    release_version = os.getenv("VERSION")[1:]
    last_day_of_month = date.today() + relativedelta(day=31)


    # pybamm/version.py
    with open(os.path.join(pybamm.root_dir(), "pybamm", "version.py"), "r+") as file:
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
            '(?<=version = ")(.+)(?=")', release_version, output
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

    # vcpkg.json
    with open(os.path.join(pybamm.root_dir(), "vcpkg.json"), "r+") as file:
        output = file.read()
        json_version_string = json.loads(output)["version-string"]
        replace_version = output.replace(json_version_string, release_version)
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    # Get latest commit id from pybamm-team/sundials-vcpkg-registry
    cmd = "git ls-remote https://github.com/pybamm-team/sundials-vcpkg-registry | grep refs/heads/main | cut -f 1 | tr -d '\n'"
    latest_commit_id = os.popen(cmd).read()

    # vcpkg-configuration.json
    with open(
        os.path.join(pybamm.root_dir(), "vcpkg-configuration.json"), "r+"
    ) as file:
        output = file.read()
        json_commit_id = json.loads(output)["registries"][0]["baseline"]
        replace_commit_id = output.replace(json_commit_id, latest_commit_id)
        file.truncate(0)
        file.seek(0)
        file.write(replace_commit_id)

    changelog_line1 = "# [Unreleased](https://github.com/pybamm-team/PyBaMM/)\n"
    changelog_line2 = f"# [v{release_version}](https://github.com/pybamm-team/PyBaMM/tree/v{release_version}) - {last_day_of_month}\n\n"

    # CHANGELOG.md
    with open(os.path.join(pybamm.root_dir(), "CHANGELOG.md"), "r+") as file:
        output_list = file.readlines()
        output_list[0] = changelog_line1
        if "rc0" in release_version:
            output_list.insert(2, changelog_line2)
        else:
            output_list[2] = changelog_line2
        file.truncate(0)
        file.seek(0)
        file.writelines(output_list)


if __name__ == "__main__":
    update_version()
