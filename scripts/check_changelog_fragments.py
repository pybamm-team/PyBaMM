"""Check the changelog fragments in changelog.d/.

Every fragment must be named ``<number>.<type>.md`` or ``<number>.<type>.<n>.md``
and hold entry text without a leading bullet or PR link. With ``--pr``, fragments
the pull request adds must also be named after it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAGMENT_DIR = "changelog.d"
NOT_FRAGMENTS = {"README.md", "template.md"}
TYPES = ("breaking", "deprecated", "feature", "bugfix", "optimization")
NAME = re.compile(rf"^(?P<number>\d+)\.(?:{'|'.join(TYPES)})(?:\.\d+)?\.md$")
PR_LINK = re.compile(r"\(\[#\d+\]\(https://github\.com/pybamm-team/PyBaMM/pull/\d+\)\)")


def check_fragment(path: Path) -> list[str]:
    """Return the problems with one fragment file.

    Parameters
    ----------
    path : Path
        The fragment file.

    Returns
    -------
    list[str]
        One message per problem, empty if the fragment is valid.
    """
    if not NAME.match(path.name):
        return [
            f"{path.name}: name it <PR number>.<type>.md, with <type> one of "
            f"{', '.join(TYPES)}"
        ]
    text = path.read_text().strip()
    if not text:
        return [f"{path.name}: the fragment is empty"]
    problems = []
    if text.startswith(("- ", "* ")):
        problems.append(f"{path.name}: drop the leading bullet, towncrier adds it")
    if PR_LINK.search(text.splitlines()[0]):
        problems.append(
            f"{path.name}: drop the PR link, towncrier adds it from the file name"
        )
    return problems


def changed_paths(base: str) -> list[str]:
    """Return the paths changed between ``base`` and ``HEAD``.

    Parameters
    ----------
    base : str
        The git ref to compare ``HEAD`` with.

    Returns
    -------
    list[str]
        Lines of ``git diff --name-status``: a status letter, a tab and a
        repository-relative path.
    """
    result = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", base, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def check_pr_numbers(pr: int, base: str) -> list[str]:
    """Return problems with the numbers of fragments a pull request adds.

    Pull requests that edit ``CHANGELOG.md`` itself, such as releases, are exempt.

    Parameters
    ----------
    pr : int
        The pull request number.
    base : str
        The git ref of the pull request's base branch.

    Returns
    -------
    list[str]
        One message per misnamed fragment.
    """
    changes = [line.split("\t", 1) for line in changed_paths(base)]
    if any(path == "CHANGELOG.md" for _, path in changes):
        return []
    problems = []
    for status, path in changes:
        name = Path(path).name
        if status != "A" or Path(path).parent.name != FRAGMENT_DIR:
            continue
        match = NAME.match(name)
        if match and int(match["number"]) != pr:
            problems.append(f"{name}: rename it to start with this PR's number, {pr}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=int, help="the pull request number")
    parser.add_argument(
        "--base", default="origin/main", help="the pull request's base ref"
    )
    args = parser.parse_args(argv)

    problems = []
    for path in sorted((REPO_ROOT / FRAGMENT_DIR).iterdir()):
        if path.name not in NOT_FRAGMENTS:
            problems.extend(check_fragment(path))
    if args.pr is not None:
        problems.extend(check_pr_numbers(args.pr, args.base))

    for problem in problems:
        print(f"{FRAGMENT_DIR}/{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
