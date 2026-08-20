"""Unit tests for the maturin build hook (``packages/pybamm/hatch_build.py``).

Synthetic wheels stand in for maturin's output, so none of this needs a Rust
toolchain.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

HOOK_PATH = Path(__file__).resolve().parents[2] / "hatch_build.py"


def load_hook_module():
    spec = importlib.util.spec_from_file_location("pybamm_hatch_build", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_hook(root):
    """Build a hook instance. ``root`` is a read-only property, so it must go
    through ``__init__``; the other arguments are unused by the paths tested."""
    module = load_hook_module()
    hook = module.RustBuildHook(
        root=str(root),
        config={},
        build_config=None,
        metadata=None,
        directory="",
        target_name="wheel",
    )
    return module, hook


def make_wheel(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name in members:
            archive.writestr(name, b"\x7fELF fake")
    return path


class TestWheelTag:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (
                "pybamm_rust-0.1.0-cp310-abi3-macosx_11_0_arm64.whl",
                "cp310-abi3-macosx_11_0_arm64",
            ),
            (
                "pybamm_rust-0.1.0-cp314-cp314-linux_x86_64.whl",
                "cp314-cp314-linux_x86_64",
            ),
            ("pybamm-26.7.1.1-cp310-abi3-win_amd64.whl", "cp310-abi3-win_amd64"),
        ],
    )
    def test_extracts_the_trailing_three_components(self, name, expected):
        module = load_hook_module()
        assert module.wheel_tag(name) == expected

    def test_rejects_a_malformed_name(self):
        module = load_hook_module()
        with pytest.raises(RuntimeError, match=r"Cannot parse a wheel tag"):
            module.wheel_tag("not-a-wheel.whl")


class TestExtractExtension:
    def test_copies_the_single_extension(self, tmp_path):
        module = load_hook_module()
        wheel = make_wheel(
            tmp_path / "w.whl", ["_core/_core.abi3.so", "_core/__init__.py"]
        )
        target = module.extract_extension(wheel, tmp_path / "dest")
        assert target.name == "_core.abi3.so"
        assert target.read_bytes() == b"\x7fELF fake"

    def test_removes_a_stale_version_specific_artifact(self, tmp_path):
        module = load_hook_module()
        destination = tmp_path / "dest"
        destination.mkdir()
        stale = destination / "_core.cpython-314-darwin.so"
        stale.write_bytes(b"old")
        (destination / "__init__.py").write_text("keep me\n")

        wheel = make_wheel(tmp_path / "w.whl", ["_core/_core.abi3.so"])
        module.extract_extension(wheel, destination)

        assert not stale.exists()
        assert sorted(p.name for p in destination.glob("_core*")) == ["_core.abi3.so"]
        assert (destination / "__init__.py").read_text() == "keep me\n"

    def test_keeps_the_checked_in_stub_file(self, tmp_path):
        module = load_hook_module()
        destination = tmp_path / "dest"
        destination.mkdir()
        stub = destination / "_core.pyi"
        stub.write_text("keep me\n")

        wheel = make_wheel(tmp_path / "w.whl", ["_core/_core.abi3.so"])
        module.extract_extension(wheel, destination)

        assert stub.read_text() == "keep me\n"

    def test_leaves_no_staging_file(self, tmp_path):
        module = load_hook_module()
        wheel = make_wheel(tmp_path / "w.whl", ["_core/_core.abi3.so"])
        destination = tmp_path / "dest"
        module.extract_extension(wheel, destination)
        assert list(destination.glob("*.tmp")) == []

    def test_rejects_a_wheel_with_no_extension(self, tmp_path):
        module = load_hook_module()
        wheel = make_wheel(tmp_path / "w.whl", ["_core/__init__.py"])
        with pytest.raises(RuntimeError, match=r"Expected exactly one _core extension"):
            module.extract_extension(wheel, tmp_path / "dest")

    def test_rejects_a_wheel_with_two_extensions(self, tmp_path):
        module = load_hook_module()
        wheel = make_wheel(
            tmp_path / "w.whl",
            ["_core/_core.abi3.so", "_core/_core.cpython-314-darwin.so"],
        )
        with pytest.raises(RuntimeError, match=r"Expected exactly one _core extension"):
            module.extract_extension(wheel, tmp_path / "dest")


class TestCrateRootResolution:
    def test_finds_crate_in_monorepo_layout(self, tmp_path):
        (tmp_path / "packages" / "pybamm-rust" / "pybamm-python").mkdir(parents=True)
        (tmp_path / "packages" / "pybamm-rust" / "pybamm-python" / "Cargo.toml").touch()
        package = tmp_path / "packages" / "pybamm"
        package.mkdir()
        _, hook = make_hook(package)
        assert hook._crate_root() == tmp_path / "packages" / "pybamm-rust"

    def test_finds_crate_in_sdist_layout(self, tmp_path):
        (tmp_path / "pybamm-rust" / "pybamm-python").mkdir(parents=True)
        (tmp_path / "pybamm-rust" / "pybamm-python" / "Cargo.toml").touch()
        _, hook = make_hook(tmp_path)
        assert hook._crate_root() == tmp_path / "pybamm-rust"

    def test_raises_when_crate_is_absent(self, tmp_path):
        _, hook = make_hook(tmp_path)
        with pytest.raises(RuntimeError, match=r"Could not find"):
            hook._crate_root()


class TestBuildInvocation:
    def test_missing_cargo_raises_actionable_error(self, tmp_path, monkeypatch):
        module, hook = make_hook(tmp_path)
        monkeypatch.setattr(module.shutil, "which", lambda _: None)
        with pytest.raises(RuntimeError, match=r"requires a Rust toolchain"):
            hook._build_extension(tmp_path)

    def test_maturin_is_invoked_through_the_interpreter_with_locked(
        self, tmp_path, monkeypatch
    ):
        module, hook = make_hook(tmp_path)
        monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/cargo")
        recorded = {}

        def fake_run(command, cwd, check):
            recorded["command"] = command
            recorded["cwd"] = cwd
            out = Path(command[command.index("--out") + 1])
            make_wheel(
                out / "pybamm_rust-0.1.0-cp310-abi3-linux_x86_64.whl",
                ["_core/_core.abi3.so"],
            )

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        artifact, tag = hook._build_extension(tmp_path)

        assert recorded["command"][:4] == [sys.executable, "-m", "maturin", "build"]
        assert "--locked" in recorded["command"]
        assert "--release" in recorded["command"]
        assert recorded["cwd"] == tmp_path
        assert tag == "cp310-abi3-linux_x86_64"
        assert artifact.name == "_core.abi3.so"


class TestBuildDataWiring:
    def _initialize(self, tmp_path, monkeypatch, version):
        _, hook = make_hook(tmp_path)
        monkeypatch.setattr(hook, "_crate_root", lambda: tmp_path)
        monkeypatch.setattr(
            hook,
            "_build_extension",
            lambda _: (Path("_core.abi3.so"), "cp310-abi3-linux_x86_64"),
        )
        build_data = {
            "artifacts": [],
            "force_include": {},
            "infer_tag": False,
            "pure_python": True,
        }
        hook.initialize(version, build_data)
        return build_data

    def test_standard_build_sets_the_tag_and_artifact(self, tmp_path, monkeypatch):
        build_data = self._initialize(tmp_path, monkeypatch, "standard")
        assert build_data["tag"] == "cp310-abi3-linux_x86_64"
        assert build_data["pure_python"] is False
        assert build_data["artifacts"] == ["/src/pybamm/rust/_core.abi3.so"]

    def test_editable_build_does_not_set_a_tag(self, tmp_path, monkeypatch):
        build_data = self._initialize(tmp_path, monkeypatch, "editable")
        assert "tag" not in build_data
        assert build_data["artifacts"] == ["/src/pybamm/rust/_core.abi3.so"]
