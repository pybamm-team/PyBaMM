"""
Automatically update the baseline of vcpkg-configuration.json
"""

import json
import os

import pybamm


def update_baseline():
    """
    Opens vcpkg-configuration.json and updates the baseline with the latest commit id
    """
    # Get latest commit id from pybamm-team/sundials-vcpkg-registry
    cmd = "git ls-remote https://github.com/pybamm-team/sundials-vcpkg-registry | grep refs/heads/main | cut -f 1 | tr -d '\n'"  # noqa: E501
    commit_id = os.popen(cmd).read()

    # Open file and write it
    with open(
        os.path.join(pybamm.root_dir(), "vcpkg-configuration.json"), "r+"
    ) as file:
        output = file.read()
        json_commit_id = json.loads(output)["registries"][0]["baseline"]
        output = output.replace(json_commit_id, commit_id)
        file.truncate(0)
        file.seek(0)
        file.write(output)


if __name__ == "__main__":
    update_baseline()
