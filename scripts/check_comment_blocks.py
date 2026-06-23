from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from itertools import pairwise
from pathlib import Path

MAX_COMMENT_BLOCK_LINES = 2
ENCODING_COMMENT = re.compile(r"^#.*coding[:=]\s*([-\w.]+)")
PYTHON_SUFFIXES = {".py", ".pyi"}


def _is_comment_only_line(source_line: str, column: int) -> bool:
    return source_line[:column].strip() == ""


def _is_ignored_header_comment(line_number: int, comment: str) -> bool:
    stripped = comment.strip()
    return (line_number == 1 and stripped.startswith("#!")) or (
        line_number <= 2 and ENCODING_COMMENT.match(stripped) is not None
    )


def _python_comment_only_lines(path: Path) -> list[int]:
    with tokenize.open(path) as handle:
        source = handle.read()

    comment_lines: list[int] = []
    source_lines = source.splitlines()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            line_number, column = token.start
            if _is_ignored_header_comment(line_number, token.string):
                continue
            if _is_comment_only_line(source_lines[line_number - 1], column):
                comment_lines.append(line_number)
    except (tokenize.TokenError, SyntaxError):
        # Unparseable source (e.g. mid-edit); ruff reports syntax errors
        # separately, so skip comment-block analysis rather than crash.
        pass

    return comment_lines


def _generic_comment_only_lines(path: Path) -> list[int]:
    comment_lines: list[int] = []
    for line_number, source_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = source_line.strip()
        if not stripped.startswith("#"):
            continue
        if _is_ignored_header_comment(line_number, stripped):
            continue
        comment_lines.append(line_number)

    return comment_lines


def _comment_only_lines(path: Path) -> list[int]:
    if path.suffix in PYTHON_SUFFIXES:
        return _python_comment_only_lines(path)
    return _generic_comment_only_lines(path)


def _violations(path: Path) -> list[int]:
    comment_lines = _comment_only_lines(path)
    if not comment_lines:
        return []

    block_start = comment_lines[0]
    block_length = 1
    violations: list[int] = []
    for previous_line, current_line in pairwise(comment_lines):
        if current_line == previous_line + 1:
            block_length += 1
            continue
        if block_length > MAX_COMMENT_BLOCK_LINES:
            violations.append(block_start)
        block_start = current_line
        block_length = 1

    if block_length > MAX_COMMENT_BLOCK_LINES:
        violations.append(block_start)
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)

    has_violations = False
    for raw_path in args.paths:
        path = Path(raw_path)
        for line_number in _violations(path):
            has_violations = True
            print(
                f"{path}:{line_number}: comment block exceeds "
                f"{MAX_COMMENT_BLOCK_LINES} lines",
                file=sys.stderr,
            )

    return 1 if has_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
