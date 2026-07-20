#!/usr/bin/env python3
"""Mirror the pybammsolvers version from version.py into its static copies.

version.py is the single source of truth for the package version (bumped at
release, guarded by the release workflow). Two other files carry a static copy:

- ``pyproject.toml`` needs a static ``[project] version`` so build-tool-less
  resolvers such as Dependabot can read the workspace metadata without invoking
  the scikit-build-core backend.
- ``vcpkg.json`` carries a ``version-string`` for the vcpkg manifest.

This script mirrors version.py into both so a release bumps only one file. It
runs as a pre-commit hook: it rewrites any drifted copy and exits non-zero (the
pre-commit auto-fix convention), otherwise it exits zero. With ``--check`` it
verifies without rewriting (used by the release workflow's drift guard).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PACKAGE_ROOT / "src" / "pybammsolvers" / "version.py"
PYPROJECT_FILE = PACKAGE_ROOT / "pyproject.toml"
VCPKG_FILE = PACKAGE_ROOT / "vcpkg.json"

_VERSION_PY_RE = re.compile(r"""__version__\s*=\s*["'](?P<value>[^"']+)["']""")
# The [project] version line, anchored to the start of a line so it never
# matches dependency specifiers or `cmake.version` inside [tool.scikit-build].
_PROJECT_VERSION_RE = re.compile(r'^version\s*=\s*"(?P<value>[^"]*)"', re.MULTILINE)
# The vcpkg manifest's top-level version-string (dependencies use version>=).
_VCPKG_VERSION_RE = re.compile(r'"version-string"\s*:\s*"(?P<value>[^"]*)"')


def read_source_version(version_file: Path) -> str:
    """Return ``__version__`` parsed from a version.py file."""
    match = _VERSION_PY_RE.search(version_file.read_text())
    if match is None:
        raise ValueError(f"could not find __version__ in {version_file}")
    return match.group("value")


def sync_target(
    path: Path,
    pattern: re.Pattern[str],
    template: str,
    source_version: str,
    *,
    check: bool = False,
) -> bool:
    """Bring the single version declaration in ``path`` up to ``source_version``.

    ``pattern`` must expose a ``value`` group capturing the current version;
    ``template`` is the full replacement string with a ``{version}`` field.
    Returns ``True`` if ``path`` differs from ``source_version``; unless
    ``check`` is set, a differing ``path`` is rewritten in place.

    Raises
    ------
    ValueError
        If ``path`` does not contain exactly one match for ``pattern``.
    """
    text = path.read_text()
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one version declaration in {path}, found {len(matches)}"
        )
    if matches[0].group("value") == source_version:
        return False
    if not check:
        path.write_text(
            pattern.sub(template.format(version=source_version), text, count=1)
        )
    return True


def sync_pyproject(
    pyproject_file: Path, source_version: str, *, check: bool = False
) -> bool:
    """Mirror ``source_version`` into pyproject.toml's ``[project] version``."""
    return sync_target(
        pyproject_file,
        _PROJECT_VERSION_RE,
        'version = "{version}"',
        source_version,
        check=check,
    )


def sync_vcpkg(vcpkg_file: Path, source_version: str, *, check: bool = False) -> bool:
    """Mirror ``source_version`` into vcpkg.json's ``version-string``."""
    return sync_target(
        vcpkg_file,
        _VCPKG_VERSION_RE,
        '"version-string": "{version}"',
        source_version,
        check=check,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync pybammsolvers version mirrors.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the mirrors match version.py without rewriting; exit non-zero on drift",
    )
    check = parser.parse_args().check
    try:
        source_version = read_source_version(VERSION_FILE)
        drifted = [
            path.name
            for path, sync in (
                (PYPROJECT_FILE, sync_pyproject),
                (VCPKG_FILE, sync_vcpkg),
            )
            if sync(path, source_version, check=check)
        ]
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if drifted:
        joined = ", ".join(drifted)
        if check:
            print(
                f"error: {joined} out of sync with version.py ({source_version}); "
                "run the sync-pybammsolvers-version pre-commit hook",
                file=sys.stderr,
            )
        else:
            print(f"synced {joined} version -> {source_version} (from version.py)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
