"""Hatchling build hook that compiles the Rust extension into the pybamm wheel.

maturin builds the ``pybamm-rust`` crate into its own wheel; this hook runs that
build, lifts the ``_core`` extension out of the result into ``pybamm/rust/``, and
re-tags the pybamm wheel as platform-specific so the extension is not shipped as
pure Python.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Buys one wheel per platform for CPython 3.10-3.14, at +20 ns per FFI call.
FEATURES = "pyo3/abi3-py310,diffsol"

CRATE_MANIFEST = Path("pybamm-python") / "Cargo.toml"
DESTINATION = Path("src") / "pybamm" / "rust"
EXTENSION_SUFFIXES = (".so", ".pyd")

CARGO_MISSING = """\
Building PyBaMM from source requires a Rust toolchain (cargo >= 1.89), which was
not found on PATH. Install it from https://rustup.rs, or install a prebuilt wheel
with `pip install --only-binary pybamm pybamm`.
"""


def wheel_tag(wheel_name: str) -> str:
    """Return the ``<python>-<abi>-<platform>`` tag from a wheel filename.

    Parameters
    ----------
    wheel_name : str
        Basename of a wheel, e.g. ``pybamm-1.0-cp310-abi3-win_amd64.whl``.

    Returns
    -------
    str
        The trailing three tag components.

    Raises
    ------
    RuntimeError
        If the name has too few components to carry a tag.
    """
    parts = wheel_name.removesuffix(".whl").split("-")
    if len(parts) < 5:
        raise RuntimeError(f"Cannot parse a wheel tag from {wheel_name!r}")
    return "-".join(parts[-3:])


def extract_extension(wheel: Path, destination: Path) -> Path:
    """Copy the ``_core`` extension out of ``wheel`` into ``destination``.

    Parameters
    ----------
    wheel : Path
        Wheel produced by maturin.
    destination : Path
        Directory to write the extension into.

    Returns
    -------
    Path
        The written extension.

    Raises
    ------
    RuntimeError
        If the wheel does not hold exactly one ``_core`` extension.
    """
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
        # A stale version-specific .so shadows a new abi3 one (importlib checks it
        # first); only binaries are stale — the checked-in _core.pyi must survive.
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

    # Local hook files load through hatchling's `custom` plugin, which overwrites
    # PLUGIN_NAME on the instance anyway. Declaring it matching avoids confusion.
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        """Build the extension and register it with the wheel being assembled.

        Parameters
        ----------
        version : str
            Hatchling's build version; ``"editable"`` keeps hatchling's own tag,
            since an editable install is not redistributed.
        build_data : dict
            Mutated in place to mark the wheel platform-specific, list the
            extension as an artifact, and set the wheel tag.
        """
        artifact, tag = self._build_extension(self._crate_root())

        # Without this the wheel is declared pure-Python, and .gitignore's `*.so`
        # would hide the artifact from hatchling's file selection.
        build_data["pure_python"] = False
        build_data["artifacts"].append(f"/{DESTINATION.as_posix()}/{artifact.name}")

        if version != "editable":
            build_data["tag"] = tag

    def _crate_root(self) -> Path:
        """Locate the Cargo workspace: ``../pybamm-rust`` in the monorepo, ``./pybamm-rust`` in an sdist."""
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
            # cwd must be the crate root: cargo discovers .cargo/config.toml from
            # its cwd, not from --manifest-path.
            subprocess.run([*command, "--out", staging], cwd=crate_root, check=True)
            wheel = next(Path(staging).glob("*.whl"))
            return (
                extract_extension(wheel, Path(self.root) / DESTINATION),
                wheel_tag(wheel.name),
            )
