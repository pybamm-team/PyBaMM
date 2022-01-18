"""
Automatically update the version number
"""

import json
import os
import re
from datetime import date, datetime

import dateutil.relativedelta

import pybamm


def update_version():
    """
    Opens file and updates the version number
    """
    current_year = datetime.now().strftime("%y")
    current_month = datetime.now().month

    release_version1 = f"{current_year}, {current_month}"
    release_version2 = f"{current_year}.{current_month}"

    # pybamm/version
    with open(os.path.join(pybamm.root_dir(), "pybamm", "version"), "r+") as file:
        output = file.read()
        replace_version = output.replace(output, release_version1)
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    # docs/conf.py
    with open(os.path.join(pybamm.root_dir(), "docs", "conf.py"), "r+") as file:
        output = file.read()
        replace_version = re.sub('(?<=version = ")(.+)(?=")', release_version2, output)
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    # CITATION.cff
    with open(os.path.join(pybamm.root_dir(), "CITATION.cff"), "r+") as file:
        output = file.read()
        replace_version = re.sub('(?<=version: ")(.+)(?=")', release_version2, output)
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    # vcpkg.json
    with open(os.path.join(pybamm.root_dir(), "vcpkg.json"), "r+") as file:
        output = file.read()
        json_version_string = json.loads(output)["version-string"]
        replace_version = output.replace(json_version_string, release_version2)
        file.truncate(0)
        file.seek(0)
        file.write(replace_version)

    # Get latest commit id from pybamm-team/sundials-vcpkg-registry
    cmd = "git ls-remote https://github.com/pybamm-team/sundials-vcpkg-registry | grep refs/heads/main | cut -f 1 | tr -d '\n'"  # noqa: E501
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
    changelog_line2 = f"# [v{release_version2}](https://github.com/pybamm-team/PyBaMM/tree/v{release_version2}) - {date.today()}\n\n"  # noqa: E501

    # CHANGELOG.md
    with open(os.path.join(pybamm.root_dir(), "CHANGELOG.md"), "r+") as file:
        output_list = file.readlines()
        output_list[0] = changelog_line1
        output_list.insert(2, changelog_line2)
        file.truncate(0)
        file.seek(0)
        file.writelines(output_list)


def get_changelog():
    """
    Opens CHANGELOG.md and overrides the changelog with the latest version.
    Used in GitHub workflow to create the changelog for the GitHub release.
    """
    # This month
    now = datetime.now()
    current_year = now.strftime("%y")
    current_month = now.month

    # Previous month
    previous_date = datetime.now() + dateutil.relativedelta.relativedelta(months=-1)
    previous_year = previous_date.strftime("%y")
    previous_month = previous_date.month

    current_version = re.escape(f"# [v{current_year}.{current_month}]")
    previous_version = re.escape(f"# [v{previous_year}.{previous_month}]")

    # Open CHANGELOG.md and keep the relevant lines
    with open(os.path.join(pybamm.root_dir(), "CHANGELOG.md"), "r+") as file:
        output = file.read()
        re_changelog = f"{current_version}.*?(##.*)(?={previous_version})"
        release_changelog = re.findall(re_changelog, output, re.DOTALL)[0]
        print(release_changelog)
        file.truncate(0)
        file.seek(0)
        file.write(release_changelog)


if __name__ == "__main__":
    update_version()
