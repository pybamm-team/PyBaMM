"""Check that an installed pybamm-model-zoo ships every committed package file.

Run it with the interpreter of an environment the built wheel was installed into,
from the checkout it was built from:

    python packages/pybamm-model-zoo/scripts/check_wheel.py --tag pybamm-model-zoo-v0.1.0

A model folder is data as much as code (its manifest, README, citation, examples,
and tests), so a wheel that drops one of those files still imports cleanly.
"""

from __future__ import annotations

import argparse
import subprocess  # nosec B404 - lists the checkout's own tracked files
from pathlib import Path

ZOO_ROOT = Path(__file__).parents[1]
SOURCE = ZOO_ROOT / "src" / "pybamm_model_zoo"
TAG_PREFIX = "pybamm-model-zoo-v"


def tracked_files(root: Path) -> list[str]:
    """Every file git tracks under ``root``, relative to it."""
    listing = subprocess.run(  # nosec B603 B607 - fixed git invocation
        ["git", "ls-files", "-z", "--", "."],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(name for name in listing.split("\0") if name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tag",
        help=f"a release tag, which must be '{TAG_PREFIX}' plus the pyproject version",
    )
    args = parser.parse_args(argv)

    import pybamm_model_zoo as zoo
    from pybamm_model_zoo._registry import Registry, read_manifest

    installed = Path(zoo.__file__).parent
    expected_version = read_manifest(ZOO_ROOT / "pyproject.toml")["project"]["version"]
    errors = []

    if installed.resolve() == SOURCE.resolve():
        errors.append(
            f"imported the source tree at {installed}, not an installed wheel"
        )
    if missing := [f for f in tracked_files(SOURCE) if not (installed / f).is_file()]:
        errors.append(f"{installed} is missing {len(missing)} file(s): {missing}")
    if zoo.__version__ != expected_version:
        errors.append(
            f"installed version {zoo.__version__} is not the pyproject version "
            f"{expected_version}"
        )
    if args.tag is not None and args.tag != f"{TAG_PREFIX}{expected_version}":
        errors.append(
            f"release tag '{args.tag}' does not match the pyproject version; "
            f"expected '{TAG_PREFIX}{expected_version}'"
        )
    source_models = sorted(Registry([SOURCE], external_paths=[]))
    if zoo.list_models() != source_models:
        errors.append(
            f"installed registry lists {zoo.list_models()}, the checkout {source_models}"
        )

    for error in errors:
        print(f"error: {error}")
    if not errors:
        print(
            f"pybamm-model-zoo {zoo.__version__} from {installed}: "
            f"{len(source_models)} models, {', '.join(source_models)}"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
