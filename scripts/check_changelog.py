from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ALLOWED_SUBSECTIONS = {
    "## Breaking changes",
    "## Deprecated",
    "## Features",
    "## Optimizations",
    "## Bug fixes",
}
TRAILING_PR_LINK = re.compile(
    r"\(\[#\d+\]\(https://github\.com/pybamm-team/PyBaMM/(?:pull|issues)/\d+\)\)\s*$"
)
MIGRATION_REQUIRED_SECTIONS = {"## Breaking changes", "## Deprecated"}


def _validate_bullet(
    path: Path, line_number: int, subsection: str | None, text: str
) -> list[str]:
    if subsection not in ALLOWED_SUBSECTIONS:
        return [
            f"{path}:{line_number}: unreleased bullet must be inside an "
            "allowed subsection"
        ]

    if TRAILING_PR_LINK.search(text) is None:
        return [f"{path}:{line_number}: bullet must end with a PR link"]

    content = TRAILING_PR_LINK.sub("", text).strip()
    if subsection in MIGRATION_REQUIRED_SECTIONS and not content:
        return [
            f"{path}:{line_number}: breaking/deprecated bullet needs a migration note"
        ]

    return []


def _violations(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    errors: list[str] = []
    in_unreleased = False
    subsection: str | None = None

    index = 0
    while index < len(lines):
        line = lines[index]

        if line.startswith("# [Unreleased]"):
            in_unreleased = True
            subsection = None
            index += 1
            continue

        if in_unreleased and line.startswith("# ["):
            break

        if not in_unreleased:
            index += 1
            continue

        if line.startswith("## "):
            subsection = line
            index += 1
            continue

        if not line.startswith("- "):
            index += 1
            continue

        # Join wrapped continuation lines (indented, non-bullet) into one
        # logical bullet so the trailing PR link can sit on a later line.
        bullet_line = index + 1
        text = line[2:]
        index += 1
        while index < len(lines):
            continuation = lines[index]
            stripped = continuation.strip()
            if (
                continuation[:1].isspace()
                and stripped
                and not stripped.startswith(("- ", "* "))
            ):
                text += " " + stripped
                index += 1
                continue
            break

        errors.extend(_validate_bullet(path, bullet_line, subsection, text))

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)

    errors: list[str] = []
    for raw_path in args.paths:
        errors.extend(_violations(Path(raw_path)))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
