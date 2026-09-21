"""Report comment slop on the paths pre-commit hands over.

At a commit that is the staged files, compared against `HEAD`. On a pull
request every tracked file matches `HEAD` in CI's clean checkout, so the
comparison is made against the base ref instead; an unresolvable base ref fails
rather than passing having scanned nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys

from comment_slop import main as report_slop

# comment-slop runs `git` from each file's own directory, where an inherited
# GIT_DIR makes that the work tree and the narrowing to changed lines fails.
GIT_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)


def resolves(ref: str) -> bool:
    """Whether `ref` names a commit in this checkout."""
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        return 0
    for var in GIT_LOCATION_VARS:
        os.environ.pop(var, None)
    base = os.environ.get("GITHUB_BASE_REF")
    if not base:
        return report_slop(paths)
    ref = f"origin/{base}"
    if not resolves(ref):
        print(
            f"comment-slop: cannot resolve {ref}. The checkout needs "
            "`fetch-depth: 0` for the base ref to exist.",
            file=sys.stderr,
        )
        return 1
    return report_slop(["--since", ref, *paths])


if __name__ == "__main__":
    sys.exit(main())
