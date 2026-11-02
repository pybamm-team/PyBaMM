"""Print the likely number of the next PyBaMM pull request.

The result is a prediction. An issue, pull request, or discussion created before
the pull request will take the predicted number first.
"""

from __future__ import annotations

import json
import subprocess

REPO = "pybamm-team/PyBaMM"
QUERY = """
query ($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    issues(orderBy: {field: CREATED_AT, direction: DESC}, first: 1) {
      nodes { number }
    }
    pullRequests(orderBy: {field: CREATED_AT, direction: DESC}, first: 1) {
      nodes { number }
    }
    discussions(orderBy: {field: CREATED_AT, direction: DESC}, first: 1) {
      nodes { number }
    }
  }
}
"""


def next_pr_number() -> int:
    """Return the likely number of the next PyBaMM pull request."""
    owner, name = REPO.split("/")
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                "graphql",
                "-f",
                f"query={QUERY}",
                "-f",
                f"owner={owner}",
                "-f",
                f"name={name}",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise SystemExit("gh is not installed; see https://cli.github.com") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "unknown error"
        raise SystemExit(f"`gh api graphql` failed: {message}") from error

    connections = json.loads(result.stdout)["data"]["repository"].values()
    newest = [item["nodes"][0]["number"] for item in connections if item["nodes"]]
    return max(newest, default=0) + 1


if __name__ == "__main__":
    print(next_pr_number())
