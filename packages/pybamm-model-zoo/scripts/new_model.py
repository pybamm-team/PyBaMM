"""Create a new model zoo entry from the template.

    uv run python packages/pybamm-model-zoo/scripts/new_model.py \
        --slug my_model --name MyModel --author "A. Author" --github ahandle

Also available as ``nox -s zoo-new -- --slug ... --name ...``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from a checkout without installing: the zoo's src/ is a sibling of scripts/.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pybamm_model_zoo import _template
from pybamm_model_zoo._exceptions import ZooError
from pybamm_model_zoo._paths import CODEOWNERS, PACKAGE_ROOT, REPO_ROOT
from pybamm_model_zoo._registry import TIERS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slug", required=True, help="folder name, lower_snake_case")
    parser.add_argument("--name", required=True, help="registry key and class name")
    parser.add_argument("--author", required=True, help="maintainer's name")
    parser.add_argument("--github", required=True, help="maintainer's GitHub handle")
    parser.add_argument("--tier", default="community", choices=TIERS)
    parser.add_argument("--license", default="BSD-3-Clause", help="SPDX identifier")
    parser.add_argument(
        "--pybamm-requires",
        default=None,
        metavar="SPECIFIER",
        help=(
            "the PyBaMM versions this model supports, e.g. '>=26.8'; defaults to a "
            "floor of the installed release"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be written"
    )
    return parser.parse_args(argv)


def append_codeowners(slug: str, github: str, *, dry_run: bool) -> str:
    line = _template.codeowners_line(slug, github)
    if dry_run:
        return line
    text = CODEOWNERS.read_text(encoding="utf-8")
    if line in text:
        return line
    separator = "" if text.endswith("\n") else "\n"
    CODEOWNERS.write_text(f"{text}{separator}{line}\n", encoding="utf-8")
    return line


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    values = _template.tokens(
        slug=args.slug,
        name=args.name,
        author=args.author,
        github=args.github,
        tier=args.tier,
        license=args.license,
        pybamm_requires=args.pybamm_requires,
    )
    destination = PACKAGE_ROOT / args.slug
    if args.dry_run:
        print(f"would render the template into {destination}")
        for target in _template.planned(destination, values).values():
            print(f"  {target}")
        print(f"would add to {CODEOWNERS}:")
        print(f"  {append_codeowners(args.slug, args.github, dry_run=True)}")
        return 0

    written = _template.render(destination, values)
    line = append_codeowners(args.slug, args.github, dry_run=False)

    print(f"Created {len(written)} files in {destination.relative_to(REPO_ROOT)}:")
    for path in written:
        print(f"  {path.relative_to(REPO_ROOT)}")
    print(f"\nAdded to .github/CODEOWNERS:\n  {line}")
    print(
        "\nNext:\n"
        f"  1. Replace the TODOs in {args.slug}/model.toml, README.md, and "
        "CITATION.bib.\n"
        "  2. Put your physics in model.py, and a test that pins a physical\n"
        "     result in tests/.\n"
        "  3. If your model needs third-party packages, add a\n"
        f"     '{values['extra']}' extra to\n"
        "     packages/pybamm-model-zoo/pyproject.toml and list it in 'zoo-all'.\n"
        "  4. uv run python packages/pybamm-model-zoo/scripts/generate.py\n"
        f"  5. nox -s zoo -- --zoo-model={args.slug}\n"
        "  6. Add a CHANGELOG.md bullet in packages/pybamm-model-zoo/."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ZooError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
