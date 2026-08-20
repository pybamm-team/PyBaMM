"""Regenerate the model zoo docs pages and status badges from the manifests.

A thin command line over :mod:`pybamm_model_zoo._docs`, which owns the rendering
so the ``docs`` contract check can compare a page against what it should contain.

    uv run python packages/pybamm-model-zoo/scripts/generate.py [--check]
    uv run python packages/pybamm-model-zoo/scripts/generate.py --collect results/
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

# Run from a checkout without installing: the zoo's src/ is a sibling of scripts/.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pybamm_model_zoo import _docs, _paths
from pybamm_model_zoo._registry import Registry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report out-of-date files without writing them",
    )
    parser.add_argument(
        "--collect",
        metavar="DIR",
        type=Path,
        help=(
            "fold one JSON file per compatibility-matrix cell into status.json "
            "before rendering"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.collect:
        generated = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        status = _docs.collect_results(args.collect, generated)
        _paths.STATUS_FILE.write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {_paths.STATUS_FILE.relative_to(_paths.REPO_ROOT)}")

    entries = sorted(Registry().values(), key=lambda entry: entry.slug)
    files = _docs.all_files(entries)
    outdated = [
        path
        for path, content in files.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    removed = _docs.stale(files)

    if args.check:
        for path in outdated + removed:
            print(f"out of date: {path.relative_to(_paths.REPO_ROOT)}")
        if outdated or removed:
            print("run `nox -s zoo-docs`", file=sys.stderr)
            return 1
        return 0

    for path in outdated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[path], encoding="utf-8")
        print(f"wrote {path.relative_to(_paths.REPO_ROOT)}")
    for path in removed:
        path.unlink()
        print(f"removed {path.relative_to(_paths.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
