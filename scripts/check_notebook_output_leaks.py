from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REDACTED = "<path>"
# The boundary exempts a URL host segment (".com/home/") but still catches a path
# after an ANSI colour code ("\x1b[..m"); vscode is first so its URI is one match.
_BOUNDARY = r"(?:\x1b\[[0-9;]*m|[^\w.\-]|^)"
LEAK_TOKEN = re.compile(
    _BOUNDARY + r"(?:"
    r"(?P<vscode>vscode-notebook-cell:[^\s'\"<>`]*)"
    r"|(?P<users>/Users/[^\s'\"<>`:]*)"
    r"|(?P<home>/home/[^\s'\"<>`:]*)"
    r"|(?P<windows>C:\\Users\\[^\s'\"<>`]*)"
    r")"
)
LEAK_LABELS = {
    "vscode": "vscode-notebook-cell:",
    "users": "/Users/",
    "home": "/home/",
    "windows": "C:\\Users\\",
}


def _redact_token(match: re.Match[str]) -> tuple[str, str]:
    name = match.lastgroup
    token = match.group(name)
    if name == "vscode":
        return token, REDACTED
    separator = "\\" if name == "windows" else "/"
    _, found, basename = token.rpartition(separator)
    return token, REDACTED + found + basename


def _json_body(text: str, ensure_ascii: bool) -> str:
    # The substring as it appears inside a JSON string literal; non-ASCII may be
    # stored escaped (\uXXXX) or raw, so callers try both ensure_ascii settings.
    return json.dumps(text, ensure_ascii=ensure_ascii)[1:-1]


def _coalesce(value: object) -> str | None:
    # nbformat stores an output string as either a list of lines or one string.
    if isinstance(value, list):
        return "".join(value)
    return value if isinstance(value, str) else None


def _iter_output_texts(output: dict) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []

    for field in ("text", "traceback", "evalue"):
        joined = _coalesce(output.get(field))
        if joined is not None:
            texts.append((field, joined))

    data = output.get("data")
    if isinstance(data, dict):
        for key, value in data.items():
            joined = _coalesce(value)
            if joined is not None:
                texts.append((f"data:{key}", joined))

    return texts


def _violations(path: Path, notebook: dict) -> list[str]:
    messages: list[str] = []

    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        for output_index, output in enumerate(outputs, start=1):
            for field, text in _iter_output_texts(output):
                seen: list[str] = []
                for match in LEAK_TOKEN.finditer(text):
                    label = LEAK_LABELS[match.lastgroup]
                    if label in seen:
                        continue
                    seen.append(label)
                    messages.append(
                        f"{path}: cell {cell_index}, output {output_index}, "
                        f"field {field}: leaked local path pattern {label!r}"
                    )

    return messages


def _redact(raw: str, notebook: dict) -> str:
    replacements: dict[str, str] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            for _field, text in _iter_output_texts(output):
                for match in LEAK_TOKEN.finditer(text):
                    token, redacted = _redact_token(match)
                    if redacted != token:
                        replacements[token] = redacted

    # Rewrite longest first so a path is never rewritten through a shorter token
    # that prefixes it; try both escapings since either may appear in the file.
    updated = raw
    for token in sorted(replacements, key=len, reverse=True):
        redacted = replacements[token]
        for ensure_ascii in (True, False):
            updated = updated.replace(
                _json_body(token, ensure_ascii),
                _json_body(redacted, ensure_ascii),
            )
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="redact leaked local paths in place, keeping the file basename",
    )
    args = parser.parse_args(argv)

    fixed: list[str] = []
    errors: list[str] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        raw = path.read_text(encoding="utf-8")
        notebook = json.loads(raw)
        if args.fix:
            updated = _redact(raw, notebook)
            if updated != raw:
                path.write_text(updated, encoding="utf-8")
                fixed.append(str(path))
                notebook = json.loads(updated)
        errors.extend(_violations(path, notebook))

    if fixed:
        print("redacted leaked paths in:\n" + "\n".join(fixed), file=sys.stderr)
    if errors:
        print("\n".join(errors), file=sys.stderr)

    # A fixer signals a non-clean tree so pre-commit reports the rewrite and the
    # contributor restages; unfixable leaks also fail.
    return 1 if (fixed or errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
