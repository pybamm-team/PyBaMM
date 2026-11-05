"""Reading a model folder's ``CITATION.bib`` and crediting it through PyBaMM."""

from __future__ import annotations

import re
from pathlib import Path

from pybamm_model_zoo._exceptions import ManifestError

CITATION_FILE = "CITATION.bib"
_ENTRY_START = re.compile(r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,\s}]+)\s*,")


def parse_bibtex(text: str) -> dict[str, str]:
    """Split BibTeX source into ``{key: entry source}``.

    A brace-matching scan rather than a full parser: it keeps the zoo free of a
    hard ``pybtex`` dependency, and PyBaMM parses the entry properly when the
    citation is printed.
    """
    entries: dict[str, str] = {}
    for match in _ENTRY_START.finditer(text):
        start = match.start()
        depth = 0
        for index in range(text.index("{", start), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    entries[match.group("key")] = text[start : index + 1]
                    break
    return entries


def read_citations(directory: Path) -> dict[str, str]:
    """Parse the ``CITATION.bib`` in ``directory``.

    Raises
    ------
    ManifestError
        If the file is missing.
    """
    path = Path(directory) / CITATION_FILE
    if not path.is_file():
        raise ManifestError(f"{path}: no such file")
    return parse_bibtex(path.read_text(encoding="utf-8"))
