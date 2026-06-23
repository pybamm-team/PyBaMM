from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ALLOWED_SUBSECTIONS = {
    "## Breaking changes",
    "## Deprecated",
    "## Features",
    "## Bug fixes",
}
TRAILING_PR_LINK = re.compile(
    r"\(\[#\d+\]\(https://github\.com/pybamm-team/PyBaMM/pull/\d+\)\)$"
)
MIGRATION_REQUIRED_SECTIONS = {"## Breaking changes", "## Deprecated"}


def _violations(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    errors: list[str] = []
    in_unreleased = False
    subsection: str | None = None

    for line_number, line in enumerate(lines, start=1):
        if line.startswith("# [Unreleased]"):
            in_unreleased = True
            subsection = None
            continue

        if in_unreleased and line.startswith("# [") and not line.startswith("## "):
            break

        if not in_unreleased:
            continue

        if line.startswith("## "):
            subsection = line
            continue

        if not line.startswith("- "):
            continue

        if subsection not in ALLOWED_SUBSECTIONS:
            errors.append(
                f"{path}:{line_number}: unreleased bullet must be inside an "
                "allowed subsection"
            )
            continue

        if TRAILING_PR_LINK.search(line) is None:
            errors.append(f"{path}:{line_number}: bullet must end with a PR link")
            continue

        content = TRAILING_PR_LINK.sub("", line[2:]).strip()
        if subsection in MIGRATION_REQUIRED_SECTIONS and not content:
            errors.append(
                f"{path}:{line_number}: breaking/deprecated bullet needs a "
                "migration note"
            )

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
