"""Filesystem locations the zoo's tooling shares.

The ``Path(__file__).parents[N]`` walks that tie this package to the repository
layout live here only, so moving a directory is one edit rather than four. The
repository-relative paths are meaningful only in a checkout; the in-tree contract
checks that use them are scoped accordingly.
"""

from __future__ import annotations

from pathlib import Path

#: The directory holding the in-tree model folders.
PACKAGE_ROOT = Path(__file__).parent
ZOO_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = ZOO_ROOT.parents[1]

ZOO_PYPROJECT = ZOO_ROOT / "pyproject.toml"
TEMPLATE_ROOT = ZOO_ROOT / "template"
BADGES_DIR = ZOO_ROOT / "badges"
STATUS_FILE = ZOO_ROOT / "status.json"
DOCS_DIR = REPO_ROOT / "docs" / "source" / "model_zoo"
CODEOWNERS = REPO_ROOT / ".github" / "CODEOWNERS"


def codeowners_folder(slug: str) -> str:
    """The repository-relative folder a model's CODEOWNERS line must cover."""
    return f"/{(PACKAGE_ROOT / slug).relative_to(REPO_ROOT).as_posix()}/"
