"""Hatchling build hook that compiles the Rust extension into the pybamm wheel."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# One wheel per platform for CPython 3.10 and later, at a small per-FFI-call cost.
FEATURES = "pyo3/abi3-py310"

CRATE_MANIFEST = Path("pybamm-python") / "Cargo.toml"
DESTINATION = Path("src") / "pybamm" / "rust"
EXTENSION_SUFFIXES = (".so", ".pyd")

CARGO_MISSING = """\
Building PyBaMM from source requires a Rust toolchain (cargo >= 1.89), which was
not found on PATH. Install it from https://rustup.rs, or install a prebuilt wheel
with `pip install --only-binary pybamm pybamm`.
"""


def wheel_tag(wheel_name: str) -> str:
    """Return the ``<python>-<abi>-<platform>`` tag from a wheel filename."""
    parts = wheel_name.removesuffix(".whl").split("-")
    if len(parts) < 5:
        raise RuntimeError(f"Cannot parse a wheel tag from {wheel_name!r}")
    return "-".join(parts[-3:])


def extract_extension(wheel: Path, destination: Path) -> Path:
    """Copy the single ``_core`` extension out of ``wheel`` into ``destination``."""
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name
            for name in archive.namelist()
            if Path(name).name.startswith("_core") and name.endswith(EXTENSION_SUFFIXES)
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"Expected exactly one _core extension in {wheel.name}, found {members}"
            )

        destination.mkdir(parents=True, exist_ok=True)
        # A stale version-specific .so shadows a new abi3 one; keep the checked-in _core.pyi.
        for stale in destination.glob("_core*"):
            if stale.name.endswith(EXTENSION_SUFFIXES):
                stale.unlink()

        target = destination / Path(members[0]).name
        staged = target.with_name(f"{target.name}.tmp")
        with archive.open(members[0]) as source, staged.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        staged.replace(target)

    return target


class RustBuildHook(BuildHookInterface):
    """Build the Rust extension into ``pybamm/rust/`` and tag the wheel for the platform."""

    # Matches what hatchling's `custom` plugin sets on the instance anyway.
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        """Build the extension and register it with the wheel being assembled."""
        artifact, tag = self._build_extension(self._crate_root())

        # Otherwise the wheel is pure-Python and .gitignore's `*.so` hides the artifact.
        build_data["pure_python"] = False
        build_data["artifacts"].append(f"/{DESTINATION.as_posix()}/{artifact.name}")

        if version != "editable":
            build_data["tag"] = tag

    def _crate_root(self) -> Path:
        """Locate the Cargo workspace in the monorepo or an sdist."""
        root = Path(self.root)
        for candidate in (root.parent / "pybamm-rust", root / "pybamm-rust"):
            if (candidate / CRATE_MANIFEST).is_file():
                return candidate
        raise RuntimeError(
            f"Could not find {CRATE_MANIFEST} under {root.parent / 'pybamm-rust'} "
            f"or {root / 'pybamm-rust'}"
        )

    def _build_extension(self, crate_root: Path) -> tuple[Path, str]:
        if shutil.which("cargo") is None:
            raise RuntimeError(CARGO_MISSING)

        command = [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(CRATE_MANIFEST),
            "--features",
            FEATURES,
        ]
        if sys.platform.startswith("linux"):
            # Let auditwheel own manylinux compliance and retagging downstream.
            command += ["--compatibility", "linux", "--auditwheel", "skip"]

        with tempfile.TemporaryDirectory() as staging:
            # CRATE_MANIFEST is relative to the Cargo workspace root.
            subprocess.run([*command, "--out", staging], cwd=crate_root, check=True)
            wheel = next(Path(staging).glob("*.whl"))
            return (
                extract_extension(wheel, Path(self.root) / DESTINATION),
                wheel_tag(wheel.name),
            )
