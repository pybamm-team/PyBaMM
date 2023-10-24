"""
Automatically update the version number
"""

import json
import os
import re

import pybamm


def update_version():
    """
    Opens file and updates the version number
    """
    release_version = os.getenv("VERSION")[1:]


    # pybamm/version.py
    with open(os.path.join(pybamm.root_dir(), "pybamm", "version.py"), "r+") as file:
        output = file.read()
        replace_version = re.sub(
            '(?<=__version__ = ")(.+)(?=")', release_version, output
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

    # docs/_static/versions.json for readthedocs build
    if "rc" not in release_version:
        with open(
            os.path.join(pybamm.root_dir(), "docs", "_static", "versions.json"),
            "r+",
        ) as file:
            output = file.read()
            json_data = json.loads(output)
            json_data.insert(
                2,
                {
                    "name": f"v{release_version}",
                    "version": f"{release_version}",
                    "url": f"https://docs.pybamm.org/en/v{release_version}/",
                },
            )
            file.truncate(0)
            file.seek(0)
            file.write(json.dumps(json_data, indent=4))

    # vcpkg.json
    with open(os.path.join(pybamm.root_dir(), "vcpkg.json"), "r+") as file:
        output = file.read()
        json_version_string = json.loads(output)["version-string"]
        replace_version = output.replace(json_version_string, release_version)
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


if __name__ == "__main__":
    update_version()
