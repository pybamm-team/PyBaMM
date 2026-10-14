"""Rendering the new-model template.

Shared by ``scripts/new_model.py`` and ``tests/test_template.py``, so the
skeleton a contributor gets is exactly the one CI proves compliant.

Placeholders use :class:`string.Template`'s ``$name`` syntax rather than braces,
because the rendered files include BibTeX and MyST, both of which use braces
themselves.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from string import Template

from pybamm_model_zoo._exceptions import ZooError
from pybamm_model_zoo._paths import TEMPLATE_ROOT, codeowners_folder
from pybamm_model_zoo._registry import (
    NAME_PATTERN,
    SLUG_PATTERN,
    extra_name,
    usable_identifier,
)

TEMPLATE_SUFFIX = ".in"
#: Matches what ``string.Template`` would substitute, for asserting none is left.
PLACEHOLDER_PATTERN = Template.pattern


def template_root() -> Path:
    """The directory holding the token-substituted skeleton."""
    if not TEMPLATE_ROOT.is_dir():
        raise ZooError(
            f"{TEMPLATE_ROOT}: no template directory. The template ships with the "
            f"repository, so render it from a checkout rather than an installed "
            f"wheel."
        )
    return TEMPLATE_ROOT


def citation_key_for(author: str, year: int) -> str:
    """A BibTeX key from an author's surname and a year, e.g. ``Author2026``."""
    words = author.split()
    surname = re.sub(r"[^A-Za-z]", "", words[-1]) if words else "Model"
    return f"{surname.capitalize()}{year}"


def tokens(
    *,
    slug: str,
    name: str,
    author: str,
    github: str,
    tier: str = "community",
    year: int | None = None,
    added: str | None = None,
) -> dict[str, str]:
    """Build the substitution map, validating the contributor's inputs."""
    if not usable_identifier(slug, SLUG_PATTERN):
        raise ZooError(
            f"slug '{slug}' must be lower_snake_case and not a Python keyword, "
            f"e.g. 'my_new_model'"
        )
    if not usable_identifier(name, NAME_PATTERN):
        raise ZooError(
            f"name '{name}' must be a valid Python identifier and not a Python "
            f"keyword, e.g. 'MyModel'"
        )
    today = datetime.date.today()
    year = year if year is not None else today.year
    return {
        "slug": slug,
        "ModelName": name,
        "Author": author,
        "github": github.lstrip("@"),
        "Year": str(year),
        "CitationKey": citation_key_for(author, year),
        "extra": extra_name(slug),
        "tier": tier,
        "added": added or today.isoformat(),
    }


def substitute(text: str, values: dict[str, str]) -> str:
    """Fill in every ``$placeholder``, refusing to leave one unresolved."""
    try:
        return Template(text).substitute(values)
    except (KeyError, ValueError) as error:
        raise ZooError(f"could not render template: {error}") from error


def planned(destination: Path, values: dict[str, str]) -> dict[Path, Path]:
    """Map each template file to the path it renders to under ``destination``."""
    root = template_root()
    return {
        source: Path(destination)
        / substitute(source.relative_to(root).with_suffix("").as_posix(), values)
        for source in sorted(root.rglob(f"*{TEMPLATE_SUFFIX}"))
    }


def render(destination: Path, values: dict[str, str]) -> list[Path]:
    """Render the template into ``destination``, returning the files written."""
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ZooError(f"{destination}: already exists and is not empty")
    written = []
    for source, target in planned(destination, values).items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            substitute(source.read_text(encoding="utf-8"), values), encoding="utf-8"
        )
        written.append(target)
    return written


def codeowners_line(slug: str, github: str) -> str:
    """The ``.github/CODEOWNERS`` line that makes a contributor their own reviewer."""
    return f"{codeowners_folder(slug)} @{github.lstrip('@')}"
