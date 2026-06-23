from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Absolute-path prefixes are anchored so a path segment inside a URL (e.g.
# "https://host/home/x") is not mistaken for a leaked local path.
LEAK_PATTERNS = (
    ("/Users/", re.compile(r"(?<![\w.\-])/Users/")),
    ("/home/", re.compile(r"(?<![\w.\-])/home/")),
    ("vscode-notebook-cell:", re.compile(r"vscode-notebook-cell:")),
    ("C:\\Users\\", re.compile(r"C:\\Users\\")),
)


def _iter_output_texts(output: dict) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []

    text = output.get("text")
    if isinstance(text, list):
        texts.append(("text", "".join(text)))
    elif isinstance(text, str):
        texts.append(("text", text))

    traceback = output.get("traceback")
    if isinstance(traceback, list):
        texts.append(("traceback", "".join(traceback)))
    elif isinstance(traceback, str):
        texts.append(("traceback", traceback))

    evalue = output.get("evalue")
    if isinstance(evalue, list):
        texts.append(("evalue", "".join(evalue)))
    elif isinstance(evalue, str):
        texts.append(("evalue", evalue))

    data = output.get("data")
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                texts.append((f"data:{key}", "".join(value)))
            elif isinstance(value, str):
                texts.append((f"data:{key}", value))

    return texts


def _violations(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    messages: list[str] = []

    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        for output_index, output in enumerate(outputs, start=1):
            for field, text in _iter_output_texts(output):
                for label, pattern in LEAK_PATTERNS:
                    if pattern.search(text):
                        messages.append(
                            f"{path}: cell {cell_index}, output {output_index}, "
                            f"field {field}: leaked local path pattern {label!r}"
                        )

    return messages


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
